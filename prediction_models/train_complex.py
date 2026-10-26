"""Native PyG ComplEx baseline for the yeast link-prediction experiment.

Unlike the GNN in :mod:`train`, ``torch_geometric.nn.kge.ComplEx`` operates on
one entity table.  This script gives every heterogeneous node type a disjoint
range in that table and treats each typed edge (including reverse edges) as a
relation.  The held-out ``hasChemStressResistance_Decreased`` edges use the
same folds and OGB evaluation negatives as ``train.py``.

Run, for example::

    python train_complex.py --epochs 500 --hidden-channels 64

PyG is installed by ``pyg_recipe.def`` in the project container.
"""

import argparse
import copy
import json
import pickle
import time
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn.functional as F
from ogb.linkproppred import Evaluator
from torch_geometric.nn.kge import ComplEx


CHEMICALS = {
    "Hydrogen peroxide": "CHEBI_16240",
    "Tunicamycin": "CHEBI_29699",
    "Caffeine": "CHEBI_27732",
    "Table salt": "CHEBI_26710",
    "Ethanol": "CHEBI_16236",
    "Acetic acid": "CHEBI_15366",
}
TARGET_EDGE_TYPE = ("genes", "hasChemStressResistance_Decreased", "mat_ent")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="kg.pkl")
    parser.add_argument("--folds", default="prediction_models/saved_folds.json")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--eval-batch-size", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=70)
    parser.add_argument("--finetune-epochs", type=int, default=50,
                        help="Target-chemical-only fine-tuning epochs (0 disables this stage).")
    parser.add_argument("--finetune-lr", type=float, default=5e-5)
    parser.add_argument("--finetune-negative-ratio", type=int, default=100,
                        help="Fixed-chemical gene negatives per target positive.")
    parser.add_argument("--finetune-batch-size", type=int, default=2_048,
                        help="Number of positive target triples per fine-tuning batch.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Defaults to cuda when available.")
    parser.add_argument("--negative-ratio", type=int, default=60,
                        help="Base number of negatives per positive for each corruption strategy.")
    parser.add_argument("--positive-weight", type=float, default=400.0,
                        help="Weight applied to positive BCE losses during general training.")
    parser.add_argument("--target-batch-ratio", type=float, default=0.4,
                        help="Fraction of positive triples in each batch involving target chemicals.")
    return parser.parse_args()


def score_edges(model, heads, relation, tails, batch_size):
    """Score an edge list without making the validation tensor GPU-resident."""
    scores = []
    for start in range(0, heads.numel(), batch_size):
        end = start + batch_size
        scores.append(model(heads[start:end], relation[start:end], tails[start:end]))
    return torch.cat(scores)


def evaluate_mrr(model, val_edges, target_relation, evaluator, batch_size):
    model.eval()
    with torch.no_grad():
        pos = score_edges(model, val_edges[0], target_relation.expand(val_edges[0].numel()),
                          val_edges[1], batch_size)
        neg = score_edges(model, val_edges["neg"][0],
                          target_relation.expand(val_edges["neg"].size(1)),
                          val_edges["neg"][1], batch_size)
    # OGB BioKG negatives are laid out as a contiguous block for each positive.
    neg = neg.view(pos.numel(), -1)
    result = evaluator.eval({"y_pred_pos": pos, "y_pred_neg": neg})
    return result, pos, neg


def chemical_metrics(model, val_edges, train_target_edges, target_relation,
                     chemical_ids, gene_ids, evaluator, batch_size):
    """Filtered per-chemical precision@5/@10, matching train.py's intent."""
    precisions = {5: [], 10: []}
    per_chemical = {}
    model.eval()
    with torch.no_grad():
        for name, chemical in chemical_ids.items():
            positives = val_edges[0, val_edges[1] == chemical].unique()
            if positives.numel() == 0:
                continue
            known = train_target_edges[0, train_target_edges[1] == chemical].unique()
            candidates = gene_ids[~torch.isin(gene_ids, known)]
            tails = torch.full_like(candidates, chemical)
            scores = score_edges(model, candidates,
                                 target_relation.expand(candidates.numel()),
                                 tails, batch_size)
            top = candidates[scores.topk(min(10, scores.numel())).indices]
            for k in (5, 10):
                size = min(k, top.numel())
                precision = torch.isin(top[:size], positives).float().mean().item()
                precisions[k].append(precision)
                if k == 5:
                    per_chemical[name] = precision
    return {k: float(np.mean(v)) if v else float("nan") for k, v in precisions.items()}, per_chemical


def balanced_triple_batches(target_indices, other_indices, batch_size,
                            target_ratio=0.4):
    target_indices = target_indices[
        torch.randperm(target_indices.numel())
    ]
    other_indices = other_indices[
        torch.randperm(other_indices.numel())
    ]

    n_target = max(1, round(batch_size * target_ratio))
    n_other = batch_size - n_target

    # Number of batches needed to cover the non-target pool.
    num_batches = max(
        1,
        int(np.ceil(other_indices.numel() / n_other)),
    )

    target_ptr = 0
    other_ptr = 0

    for _ in range(num_batches):
        # Take non-target triples without replacement.
        other_batch = other_indices[
            other_ptr:other_ptr + n_other
        ]
        other_ptr += other_batch.numel()

        # Reuse and reshuffle target triples if necessary.
        target_parts = []
        remaining = n_target

        while remaining > 0:
            if target_ptr >= target_indices.numel():
                target_indices = target_indices[
                    torch.randperm(target_indices.numel())
                ]
                target_ptr = 0

            take = min(
                remaining,
                target_indices.numel() - target_ptr,
            )

            target_parts.append(
                target_indices[target_ptr:target_ptr + take]
            )
            target_ptr += take
            remaining -= take

        target_batch = torch.cat(target_parts)

        batch = torch.cat([target_batch, other_batch])
        batch = batch[torch.randperm(batch.numel())]

        yield batch


def sample_target_negatives(positive_edges, known_edges, gene_ids, num_nodes,
                            negative_ratio):
    """Corrupt gene heads while keeping each chemical fixed."""

    num_negative = positive_edges.size(1) * negative_ratio

    # Sample exclusively from the gene entity IDs.
    sampled_indices = torch.randint(
        gene_ids.numel(),
        (num_negative,),
        device=positive_edges.device,
    )
    heads = gene_ids[sampled_indices]

    # Repeat each positive's chemical for its negative samples.
    tails = positive_edges[1].repeat_interleave(negative_ratio)

    # Encode (head, tail) pairs for efficient membership checks.
    known_keys = known_edges[0] * num_nodes + known_edges[1]

    invalid = torch.isin(heads * num_nodes + tails, known_keys)

    while invalid.any():
        sampled_indices = torch.randint(
            gene_ids.numel(),
            (invalid.sum().item(),),
            device=positive_edges.device,
        )
        heads[invalid] = gene_ids[sampled_indices]
        invalid = torch.isin(heads * num_nodes + tails, known_keys)

    return torch.stack((heads, tails))


def sample_typed_negatives(pos_h, pos_r, pos_t, relation_edge_types,
                           entity_ids_by_type, known_edges_by_relation,
                           num_nodes, neg_ratio):
    """GNN-style random, fixed-head, and fixed-tail corruption.

    All replacements respect the relation's source/destination node types.
    Known positive triples are rejected for the same relation.
    """
    all_neg_h, all_neg_r, all_neg_t = [], [], []

    for relation_id in torch.unique(pos_r).tolist():
        mask = pos_r == relation_id
        h = pos_h[mask]
        r = pos_r[mask]
        t = pos_t[mask]

        edge_type = relation_edge_types[int(relation_id)]
        head_pool = entity_ids_by_type[edge_type[0]]
        tail_pool = entity_ids_by_type[edge_type[2]]

        true_edges = known_edges_by_relation[int(relation_id)]
        true_keys = true_edges[0] * num_nodes + true_edges[1]

        # Mirror the GNN's three corruption groups:
        # random: corrupt both; fixed head: corrupt tail;
        # fixed tail: corrupt head.
        for mode, ratio in (
            ("random", neg_ratio),
            ("fixed_head", neg_ratio),
            ("fixed_tail", 8 * neg_ratio),
        ):
            n = h.numel() * ratio

            if mode == "random":
                neg_h = head_pool[
                    torch.randint(head_pool.numel(), (n,), device=pos_h.device)
                ]
                neg_t = tail_pool[
                    torch.randint(tail_pool.numel(), (n,), device=pos_h.device)
                ]
            elif mode == "fixed_head":
                neg_h = h.repeat_interleave(ratio)
                neg_t = tail_pool[
                    torch.randint(tail_pool.numel(), (n,), device=pos_h.device)
                ]
            else:
                neg_h = head_pool[
                    torch.randint(head_pool.numel(), (n,), device=pos_h.device)
                ]
                neg_t = t.repeat_interleave(ratio)

            # Reject known true triples; resample only the corrupted side(s).
            invalid = torch.isin(neg_h * num_nodes + neg_t, true_keys)

            while invalid.any():
                n_invalid = int(invalid.sum().item())

                if mode in ("random", "fixed_tail"):
                    neg_h[invalid] = head_pool[
                        torch.randint(
                            head_pool.numel(),
                            (n_invalid,),
                            device=pos_h.device,
                        )
                    ]

                if mode in ("random", "fixed_head"):
                    neg_t[invalid] = tail_pool[
                        torch.randint(
                            tail_pool.numel(),
                            (n_invalid,),
                            device=pos_h.device,
                        )
                    ]

                invalid = torch.isin(neg_h * num_nodes + neg_t, true_keys)

            all_neg_h.append(neg_h)
            all_neg_r.append(r[:1].expand(n))
            all_neg_t.append(neg_t)

    return (
        torch.cat(all_neg_h),
        torch.cat(all_neg_r),
        torch.cat(all_neg_t),
    )


def weighted_bce(scores, labels, positive_weight=1.0):
    """BCE for probability scores, with optional positive-class weighting."""
    scores = scores.clamp(min=1e-7, max=1.0 - 1e-7)

    losses = F.binary_cross_entropy_with_logits(
        scores,
        labels,
        reduction="none",
    )

    weights = torch.where(
        labels == 1,
        torch.as_tensor(
            positive_weight, dtype=losses.dtype, device=losses.device
        ),
        torch.ones_like(losses),
    )

    return (losses * weights).sum()


def main():
    BASE = os.path.dirname(os.path.abspath(__file__))
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # base = Path(__file__).resolve().parent

    with open(BASE / args.graph, "rb") as handle:
        saved = pickle.load(handle)
    graph, index = saved["graph"].contiguous(), saved["index"]
    for edge_type in list(graph.edge_types):
        if graph[edge_type].edge_index.size(1) < 1000:
            del graph[edge_type]
    # The unsuffixed relation contains the same experimental signal at a
    # coarser label level.  ``train.py`` removes it before fitting a decreased
    # resistance model; retaining it would leak held-out target links here.
    base_relation = ("genes", "hasChemStressResistance", "mat_ent")
    base_reverse = ("mat_ent", "rev_hasChemStressResistance", "genes")
    for edge_type in (base_relation, base_reverse):
        if edge_type in graph.edge_types:
            del graph[edge_type]
    if TARGET_EDGE_TYPE not in graph.edge_types:
        raise KeyError(f"Target edge type {TARGET_EDGE_TYPE} is not in the graph")

    # ``ComplEx`` has one entity embedding matrix, so concatenate type-local IDs.
    offsets, total_nodes = {}, 0
    for node_type in graph.node_types:
        offsets[node_type] = total_nodes
        total_nodes += len(graph.node_id_dict[node_type])
    relation_ids = {edge_type: i for i, edge_type in enumerate(graph.edge_types)}
    target_relation = torch.tensor(relation_ids[TARGET_EDGE_TYPE], device=device)
    gene_ids = torch.arange(len(graph.node_id_dict["genes"]), device=device) + offsets["genes"]
    chemical_ids = {
        name: index["mat_ent"][f"http://purl.obolibrary.org/obo/{chebi}"] + offsets["mat_ent"]
        for name, chebi in CHEMICALS.items()
    }

    with open(os.path.join(BASE, args.folds)) as handle:
        folds = json.load(handle)
    fold_key = "___".join(TARGET_EDGE_TYPE)
    if any(fold_key not in fold for fold in folds):
        raise KeyError(f"{args.folds} does not contain folds for {TARGET_EDGE_TYPE}")

    evaluator = Evaluator(name="ogbl-biokg")
    all_fold_metrics, chemical_precision = [], {name: [] for name in CHEMICALS}
    checkpoints = []
    print("\nComplEx training configuration")
    print("=" * 34)
    for name, value in vars(args).items():
        print(f"{name}: {value}")
    print("\nDerived dataset/model settings")
    print(f"device: {device}")
    print(f"entities: {total_nodes}")
    print(f"relations: {len(relation_ids)}")
    print(f"folds: {len(folds)}")
    print(f"target edge type: {TARGET_EDGE_TYPE}")
    print(f"target relation ID: {relation_ids[TARGET_EDGE_TYPE]}")
    # ComplEx has one entity embedding table and one relation embedding table;
    # this count makes its capacity directly comparable across baseline runs.
    parameter_count = sum(
        parameter.numel()
        for parameter in ComplEx(total_nodes, len(relation_ids), args.hidden_channels).parameters()
        if parameter.requires_grad
    )
    print(f"trainable parameters: {parameter_count:,}")
    # assert False
    print(f"training objective: PyG ComplEx native negative-sampling loss")
    print(f"evaluation: OGB BioKG MRR and filtered chemical Precision@5/@10")
    print("=" * 34, flush=True)

    for fold_number, fold in enumerate(folds):
        split = fold[fold_key]
        target_train = torch.as_tensor(split["train"], dtype=torch.long).t().contiguous()
        target_reverse = torch.as_tensor(split["rev"], dtype=torch.long).t().contiguous()
        val_local = torch.as_tensor(split["val"], dtype=torch.long).t().contiguous()
        neg_local = torch.as_tensor(split["neg"], dtype=torch.long).contiguous()

        # Train on every relation in the KG.  Only the target relation and its
        # reverse are replaced with their fold-specific training edges, avoiding
        # message/triple leakage from validation links.
        heads, relations, tails = [], [], []
        reverse_target = (TARGET_EDGE_TYPE[2], f"rev_{TARGET_EDGE_TYPE[1]}", TARGET_EDGE_TYPE[0])
        for edge_type in graph.edge_types:
            if edge_type == TARGET_EDGE_TYPE:
                edge_index = target_train
            elif edge_type == reverse_target:
                edge_index = target_reverse
            else:
                edge_index = graph[edge_type].edge_index.cpu()
            heads.append(edge_index[0] + offsets[edge_type[0]])
            tails.append(edge_index[1] + offsets[edge_type[2]])
            relations.append(torch.full((edge_index.size(1),), relation_ids[edge_type], dtype=torch.long))
        heads, relations, tails = torch.cat(heads), torch.cat(relations), torch.cat(tails)

        relation_edge_types = {
            relation_ids[edge_type]: edge_type
            for edge_type in graph.edge_types
        }

        entity_ids_by_type = {
            node_type: (
                torch.arange(
                    len(graph.node_id_dict[node_type]),
                    device=device,
                ) + offsets[node_type]
            )
            for node_type in graph.node_types
        }

        # The flattened training triples already contain fold-specific target
        # edges and reverse edges, so use these as the known training positives.
        known_edges_by_relation = {}

        for relation_id in torch.unique(relations).tolist():
            mask = relations == relation_id
            known_edges_by_relation[int(relation_id)] = torch.stack(
                [heads[mask], tails[mask]]
            ).to(device)

        # Move the training triples once, rather than transferring every batch.
        # heads = heads.to(device)
        # relations = relations.to(device)
        # tails = tails.to(device)

        target_chemical_ids = torch.tensor(
            list(chemical_ids.values()),
            dtype=torch.long,
        )

        target_relation_id = relation_ids[TARGET_EDGE_TYPE]

        target_mask = (
            (relations == target_relation_id)
            & torch.isin(tails, target_chemical_ids)
        )

        target_indices = torch.where(target_mask)[0]
        other_indices = torch.where(~target_mask)[0]

        print(f"Target-chemical triples: {target_indices.numel():,}")
        print(f"Other triples: {other_indices.numel():,}")

        val_edges = torch.stack((val_local[0] + offsets["genes"], val_local[1] + offsets["mat_ent"])).to(device)
        val_edges = {0: val_edges[0], 1: val_edges[1], "neg": torch.stack((
            neg_local[0] + offsets["genes"], neg_local[1] + offsets["mat_ent"]
        )).to(device)}
        train_target_edges = torch.stack((target_train[0] + offsets["genes"],
                                          target_train[1] + offsets["mat_ent"])).to(device)

        model = ComplEx(total_nodes, len(relation_ids), args.hidden_channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_precision, stale_epochs, best_state = -np.inf, 0, None
        best_summary = None
        print(f"fold {fold_number + 1}/{len(folds)}: {heads.numel()} training triples")

        for epoch in range(1, args.epochs + 1):
            model.train()
            # permutation = torch.randperm(heads.numel())
            total_loss = 0.0
            # for batch in permutation.split(args.batch_size):
            # for batch in balanced_triple_batches(target_indices, other_indices,
            #                                      args.batch_size,
            #                                      target_ratio=0.4):
            #     optimizer.zero_grad()
            #     # This is PyG's native KGE negative-sampling loss.
            #     loss = model.loss(heads[batch].to(device), relations[batch].to(device), tails[batch].to(device))
            #     loss.backward()
            #     optimizer.step()
            #     total_loss += loss.detach().item()
            for batch in balanced_triple_batches(target_indices, other_indices,
                                        args.batch_size,
                                        target_ratio=args.target_batch_ratio):
                batch = batch.to(device)

                pos_h = heads[batch]
                pos_r = relations[batch]
                pos_t = tails[batch]

                neg_h, neg_r, neg_t = sample_typed_negatives(
                    pos_h,
                    pos_r,
                    pos_t,
                    relation_edge_types,
                    entity_ids_by_type,
                    known_edges_by_relation,
                    total_nodes,
                    args.negative_ratio,
                )

                pos_scores = model(pos_h, pos_r, pos_t).flatten()
                neg_scores = model(neg_h, neg_r, neg_t).flatten()

                scores = torch.cat([pos_scores, neg_scores])
                labels = torch.cat([
                    torch.ones_like(pos_scores),
                    torch.zeros_like(neg_scores),
                ])

                optimizer.zero_grad()

                loss = weighted_bce(scores, labels,
                                    positive_weight=args.positive_weight)

                loss.backward()
                optimizer.step()

                total_loss += loss.detach().item()

            if epoch % args.eval_every:
                continue
            mrr, _, _ = evaluate_mrr(model, val_edges, target_relation, evaluator, args.eval_batch_size)
            precision, per_chemical = chemical_metrics(
                model, torch.stack((val_edges[0], val_edges[1])),
                train_target_edges, target_relation, chemical_ids, gene_ids,
                evaluator, args.eval_batch_size,
            )
            print(f"  epoch={epoch:03d} loss={total_loss:.4f} mrr={mrr['mrr_list'].mean().item():.4f} "
                  f"P@5={precision[5]:.4f} P@10={precision[10]:.4f}", flush=True)
            if precision[5] > best_precision:
                best_precision, stale_epochs = precision[5], 0
                best_state = copy.deepcopy(model.state_dict())
                best_summary = (mrr["mrr_list"].mean().item(), precision, per_chemical)
            else:
                stale_epochs += args.eval_every
                if stale_epochs > args.patience:
                    print("  early stopping: Precision@5 did not improve")
                    break

        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())
            best_summary = (float("nan"), {5: float("nan"), 10: float("nan")}, {})

        # Stage 2 mirrors train.py's retrieval fine-tuning: train only on the
        # experimental target relation and corrupt genes while holding the
        # chemical fixed.  All ComplEx embeddings remain trainable because the
        # native PyG model has no separate encoder/decoder modules to freeze.
        model.load_state_dict(best_state)
        final_state, final_summary = best_state, best_summary
        if args.finetune_epochs:
            print("  target-chemical fine-tuning")
            finetune_optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay,
            )
            target_best_precision = best_summary[1][5]
            target_best_precision10 = best_summary[1][10]
            # pos_weight = torch.tensor([args.finetune_negative_ratio], device=device)
            for epoch in range(1, args.finetune_epochs + 1):
                model.train()
                permutation = torch.randperm(train_target_edges.size(1), device=device)
                total_loss = 0.0
                for batch in permutation.split(args.finetune_batch_size):
                    positives = train_target_edges[:, batch]
                    # negatives = sample_target_negatives(
                    #     positives, train_target_edges, total_nodes,
                    #     args.finetune_negative_ratio,
                    # )
                    negatives = sample_target_negatives(positives,
                                    train_target_edges, gene_ids, total_nodes,
                                    args.finetune_negative_ratio)
                    relation_pos = target_relation.expand(positives.size(1))
                    relation_neg = target_relation.expand(negatives.size(1))
                    scores = torch.cat((
                        model(positives[0], relation_pos, positives[1]),
                        model(negatives[0], relation_neg, negatives[1]),
                    ))
                    labels = torch.cat((
                        torch.ones(positives.size(1), device=device),
                        torch.zeros(negatives.size(1), device=device),
                    ))
                    finetune_optimizer.zero_grad()
                    scores = scores.clamp(min=1e-7, max=1.0 - 1e-7)

                    losses = F.binary_cross_entropy_with_logits(
                        scores,
                        labels,
                        reduction="none",
                    )

                    weights = torch.where(
                        labels == 1,
                        torch.full_like(labels, args.finetune_negative_ratio),
                        torch.ones_like(labels),
                    )

                    loss = (losses * weights).mean()
                    loss.backward()
                    finetune_optimizer.step()
                    total_loss += loss.detach().item()

                mrr, _, _ = evaluate_mrr(model, val_edges, target_relation, evaluator, args.eval_batch_size)
                precision, per_chemical = chemical_metrics(
                    model, torch.stack((val_edges[0], val_edges[1])), train_target_edges,
                    target_relation, chemical_ids, gene_ids, evaluator, args.eval_batch_size,
                )
                print(f"    epoch={epoch:03d} loss={total_loss:.4f} "
                      f"mrr={mrr['mrr_list'].mean().item():.4f} "
                      f"P@5={precision[5]:.4f} P@10={precision[10]:.4f}", flush=True)
                if (precision[5] > target_best_precision or
                        (precision[5] == target_best_precision and
                         precision[10] > target_best_precision10)):
                    target_best_precision, target_best_precision10 = precision[5], precision[10]
                    final_state = copy.deepcopy(model.state_dict())
                    final_summary = (mrr["mrr_list"].mean().item(), precision, per_chemical)

            print(f"  fine-tuning best P@5: {final_summary[1][5]:.4f} "
                  f"(pre-fine-tuning: {best_summary[1][5]:.4f})", flush=True)

        checkpoints.append(final_state)
        all_fold_metrics.append(final_summary)
        for name, value in final_summary[2].items():
            chemical_precision[name].append(value)

    print("\nComplEx cross-validation summary")
    print(f"MRR: {np.mean([x[0] for x in all_fold_metrics]):.4f} +- {np.std([x[0] for x in all_fold_metrics]):.4f}")
    print(f"Precision@5: {np.mean([x[1][5] for x in all_fold_metrics]):.4f} +- {np.std([x[1][5] for x in all_fold_metrics]):.4f}")
    print(f"Precision@10: {np.mean([x[1][10] for x in all_fold_metrics]):.4f} +- {np.std([x[1][10] for x in all_fold_metrics]):.4f}")
    for name, values in chemical_precision.items():
        if values:
            print(f"{name} Precision@5: {np.mean(values):.4f} +- {np.std(values):.4f}")

    output_dir = BASE / "trained_yeast_models"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / f"complex_cv_{time.strftime('%Y%m%d-%H%M%S')}.pt"
    torch.save({
        "model": "torch_geometric.nn.kge.ComplEx",
        "state_dicts": checkpoints,
        "offsets": offsets,
        "relation_ids": relation_ids,
        "target_edge_type": TARGET_EDGE_TYPE,
        "metrics": all_fold_metrics,
        "args": vars(args),
    }, output)
    print(f"Saved checkpoints to {output}")


if __name__ == "__main__":
    main()
