# %%
import torch
import json, os, pickle, copy, time
from torch.cuda import is_available
import tqdm
from torch.nn import BCEWithLogitsLoss
from ogb.linkproppred import Evaluator
import numpy as np
from model import Classifier, MultiDomainComplEx, MultiDomainRotatE
from torch_geometric.transforms import RandomLinkSplit
import torch.nn.functional as F
from itertools import chain

from loss_functions import box_loss

if is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)
torch.set_float32_matmul_precision('high')
# %%
BASE = os.path.dirname(os.path.abspath(__file__))
print(BASE)

# %%
MIN_EDGE_SIZE = 1000
# %%
chemicals = {'Hydrogen peroxide': 'CHEBI_16240',
'Tunicamycin': 'CHEBI_29699',
'Caffeine': 'CHEBI_27732',
'Table salt': 'CHEBI_26710',
'Ethanol': 'CHEBI_16236',
'Acetic acid': 'CHEBI_15366'}
rev_chemicals = {v:k for k,v in chemicals.items()}
# %%
with open(os.path.join(BASE, 'kg.pkl'), 'rb') as fi:
    g = pickle.load(fi)
    graph = g['graph'].contiguous()#.to(device)
    gci0 = g['gci0']
    index = g['index']

print(f"Number of edges before ignoring fewer than {MIN_EDGE_SIZE}: {len(graph.edge_types)}")
for e in graph.edge_types:
    if graph[e].edge_index.shape[1] < MIN_EDGE_SIZE:
        del graph[e]

print(f"Number of edges after ignoring fewer than {MIN_EDGE_SIZE}: {len(graph.edge_types)}")

for v in gci0.values():
    v = v.to(device)
    # valid_graph = g['valid'].contiguous().to(device)

# %%
chem_indices = {}
for c in chemicals.values():
    chem_indices[c] = index['mat_ent']['http://purl.obolibrary.org/obo/'+c]
# %%
TARGET_CHEMICALS = set(chem_indices.values())
OVERSAMPLE_FACTOR = 10
# Stage 2 specializes the general model for the actual experimental queries.
# Negatives keep the chemical fixed, so the decoder must separate genes for the
# same chemical rather than distinguish arbitrary gene--chemical pairs.
FINETUNE_EPOCHS = 50
FINETUNE_LR = 5e-5
FINETUNE_NEG_RATIO = 100
FINETUNE_MESSAGE_RATIO = 0.9
# Residual chemical-specific heads augment, rather than replace, the shared
# decoder. Set this to False to reproduce the shared-decoder-only model.
USE_TARGET_CHEMICAL_HEADS = True
#USE_TARGET_CHEMICAL_HEADS = False
TARGET_HEAD_CHANNELS = None  # None reuses the shared decoder's nn_channels
TARGET_HEAD_CHANNELS = [16, 8, 4]  # None reuses the shared decoder's nn_channels
# TARGET_HEAD_CHANNELS = [8, 4]
print(f"use target heads: {USE_TARGET_CHEMICAL_HEADS}")
print(f"target head channels: {TARGET_HEAD_CHANNELS}")
# %%
def generate_folds(graph, edge_types, file_path=None, num_folds=None):
    splits = {}
    device = graph[edge_types[0]].edge_index.device
    assert file_path is not None or num_folds is not None, "Either file_path or num_folds must be provided"
    if file_path is not None:
        print(f"Loading folds from {file_path}...")
        with open(file_path, 'r') as fi:
            saved_folds = json.load(fi)
        print(f"Number of folds: {len(saved_folds)}")
        # assert
        assert num_folds is None or len(saved_folds) == num_folds, f"Number of saved folds {len(saved_folds)} does not match number of folds {num_folds}"
        for fold in saved_folds:
            fold_info = {}
            for k, v in fold.items():
                kt = tuple(k.split('___'))
                assert kt in edge_types, f"Edge type {kt} not found in saved folds" 
                fold_info[kt] = {'train': torch.tensor(v['train'], device=device),
                                 'val': torch.tensor(v['val'], device=device),
                                 'rev': torch.tensor(v['rev'], device=device),
                                 'neg': torch.tensor(v['neg'], device=device)}
            yield fold_info
    else:
        for e in edge_types:
            N = graph[e].edge_index.shape[1]
            # splits[e] = torch.tensor_split(torch.arange(N), num_folds)
            perm = torch.randperm(N)
            splits[e] = torch.tensor_split(perm, num_folds)
            
        for i in range(num_folds):
            fold_edges = {}
            for e in edge_types:
                val = graph[e].edge_index.T[splits[e][i]]
                train_idx = torch.cat([splits[e][j] for j in range(num_folds) if j != i])
                train = graph[e].edge_index.T[train_idx]
                rev = graph[e[2], f'rev_{e[1]}', e[0]].edge_index.T[train_idx]
                fold_edges[e] = {'train': train, 'val': val, 'rev': rev}
            yield fold_edges


# %%

def sample_neg_ex(num_pos, max_h, max_t, neg_ratio=10, device='cpu',
                  head=None, tail=None, true_edges=None):
    """
    Negative sampling for KG edges.

    Args:
        num_pos: number of positive edges
        max_h: number of head nodes
        max_t: number of tail nodes
        neg_ratio: negatives per positive
        head: optional tensor of heads to FIX (shape [1, num_neg] or [1, num_pos])
        tail: optional tensor of tails to FIX (same logic)

    Returns:
        Tensor of shape [2, num_neg]
    """

    num_neg = neg_ratio * num_pos

    # Normalize shape → [1, N]
    if head is not None:
        head = head.to(device)
        if head.dim() == 1:
            head = head.unsqueeze(0)

        if head.shape[1] == num_pos:
            head = head.repeat_interleave(neg_ratio, dim=1)

        heads = head
    else:
        heads = torch.randint(0, max_h, (1, num_neg), device=device)

    if tail is not None:
        tail = tail.to(device)
        if tail.dim() == 1:
            tail = tail.unsqueeze(0)

        if tail.shape[1] == num_pos:
            tail = tail.repeat_interleave(neg_ratio, dim=1)

        tails = tail
    else:
        tails = torch.randint(0, max_t, (1, num_neg), device=device)

    # resample any true negative
    if true_edges is not None:
        true_edges = true_edges.to(device)
        true_ids = true_edges[0] * max_t + true_edges[1]
        while True:
            neg_ids = heads.squeeze(0) * max_t + tails.squeeze(0)
            invalid = torch.isin(neg_ids, true_ids)
            if not invalid.any():
                break

            n_invalid = invalid.sum().item()
            # Only resample invalid entries
            if head is None:
                heads[0, invalid] = torch.randint(
                    0,
                    max_h,
                    (n_invalid,),
                    device=device
                )
            if tail is None:
                tails[0, invalid] = torch.randint(
                    0,
                    max_t,
                    (n_invalid,),
                    device=device
                )
    return torch.cat([heads, tails], dim=0)



def get_index_splits(idx_lens, message_ratio):
    nbr_splits = int(np.ceil(1 / (1-message_ratio)))
    
    prev = {k: 0 for k in idx_lens.keys()}
    perm = {k: torch.randperm(idx_len) for k, idx_len in idx_lens.items()}
    for _ in range(nbr_splits):
        indices = {}
        for k, idx_len in idx_lens.items():
            split_size = int(np.ceil(idx_len * (1-message_ratio)))
            label_indices = perm[k][prev[k]:prev[k]+split_size]
            prev[k] = prev[k]+split_size
            mask = torch.ones(idx_len, dtype=torch.bool)
            mask[label_indices] = False
            message_indices = torch.arange(idx_len)[mask]
            indices[k] = (message_indices, label_indices)

        yield indices


def get_balanced_index_splits(idx_lens,
                              message_ratio,
                              graph,
                              predicted_edge_types,
                              target_chemicals,
                              target_ratio=0.4):
    """
    Similar to get_index_splits(), but each label split contains a fixed
    fraction of target-chemical examples.

    target_ratio:
        Fraction of label edges that should involve target chemicals.
        Example:
            0.4 -> 40% target chemicals, 60% all others.
    """

    nbr_splits = int(np.ceil(1 / (1 - message_ratio)))

    # ------------------------------------------------------------------
    # Build independent pools for target / non-target edges
    # ------------------------------------------------------------------

    pools = {}

    for k in predicted_edge_types:

        edge_index = graph[k].edge_index

        target_mask = torch.isin(
            edge_index[1],
            torch.tensor(list(target_chemicals),
                         device=edge_index.device)
        )

        target_idx = torch.where(target_mask)[0]
        other_idx = torch.where(~target_mask)[0]

        target_idx = target_idx[torch.randperm(len(target_idx))]
        other_idx = other_idx[torch.randperm(len(other_idx))]

        pools[k] = {
            "target": target_idx,
            "other": other_idx,
            "target_ptr": 0,
            "other_ptr": 0
        }

    # ------------------------------------------------------------------
    # Create batches
    # ------------------------------------------------------------------

    for _ in range(nbr_splits):

        indices = {}

        for k, idx_len in idx_lens.items():

            split_size = int(np.ceil(idx_len * (1 - message_ratio)))

            n_target = int(round(split_size * target_ratio))
            n_other = split_size - n_target

            pool = pools[k]

            # If we run out of either pool, reshuffle and restart.
            if pool["target_ptr"] + n_target > len(pool["target"]):
                pool["target"] = pool["target"][torch.randperm(len(pool["target"]))]
                pool["target_ptr"] = 0

            if pool["other_ptr"] + n_other > len(pool["other"]):
                pool["other"] = pool["other"][torch.randperm(len(pool["other"]))]
                pool["other_ptr"] = 0

            label_indices = torch.cat([
                pool["target"][pool["target_ptr"]:pool["target_ptr"] + n_target],
                pool["other"][pool["other_ptr"]:pool["other_ptr"] + n_other]
            ])

            pool["target_ptr"] += n_target
            pool["other_ptr"] += n_other

            # Shuffle inside batch
            label_indices = label_indices[
                torch.randperm(len(label_indices))
            ]

            # Message edges are simply all remaining edges
            mask = torch.ones(idx_len, dtype=torch.bool)
            mask[label_indices] = False

            message_indices = torch.arange(idx_len)[mask]

            indices[k] = (message_indices, label_indices)

        yield indices

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %%
def eval_full(model, val_graph):
    with torch.no_grad():
        val_data = {'y_pred_pos': torch.tensor([], device=device),
                    'y_pred_neg': torch.tensor([], device=device)}
        for k in val_graph.edge_label_index_dict.keys():
            pos = val_graph[k].edge_label_index
            pos_pred = model(val_graph, edges_to_predict={k: pos},
                             return_embs=False)
            val_data['y_pred_pos'] = torch.cat((val_data['y_pred_pos'],
                                                pos_pred), dim=0)
            neg = val_graph[k].neg_edge_label_index
            val_batches = int(np.ceil(neg.shape[1]/VAL_BATCH_SIZE))
            idx = 0
            # print('neg:', neg.shape)
            for i in range(val_batches):
                neg_data = neg[:,idx:idx+VAL_BATCH_SIZE]#,:].reshape((2, -1))
                neg_pred = model(val_graph, edges_to_predict={k: neg_data},
                                 return_embs=False)
                val_data['y_pred_neg'] = torch.cat((val_data['y_pred_neg'],
                                neg_pred.reshape((-1,NEG_VAL_RATIO))), dim=0)
                idx += VAL_BATCH_SIZE
        val_data['y_pred_pos'] = val_data['y_pred_pos'].squeeze()

        val = ev.eval(val_data)
        print(f"MRR: {val['mrr_list'].mean()}")
        print(f"hits@1: {val['hits@1_list'].mean()}")
        print(f"hits@3: {val['hits@3_list'].mean()}")
        print(f"hits@10: {val['hits@10_list'].mean()}", flush=True)

    return val


def eval_chemicals(model, val_graph, chem_indices):
    precisions_at_5 = []
    precision_at = {5: [], 10:[]}
    chemical_precision_at_5 = {}
    with torch.no_grad():
        gene_max = val_graph['genes'].node_id.max().item()
        for edge_type, edge_index in val_graph.edge_label_index_dict.items():
            # Run additional validation for chemicals in chem_indices
            full_edge_index = val_graph[edge_type].edge_index
            for chem, (pos_examples, neg_examples) in zip(chem_indices.keys(), get_chemical_validation(edge_index, full_edge_index, list(chem_indices.values()), gene_max)):
                # Compute positive and negative predictions
                y_pred_pos = model(val_graph, edges_to_predict={edge_type: pos_examples}, return_embs=False).squeeze()  # (C,)
                y_pred_neg = model(val_graph, edges_to_predict={edge_type: neg_examples}, return_embs=False).squeeze()  # (N,) or (N,)

                # Repeat y_pred_neg for each positive example (C, N)
                C = y_pred_pos.shape[0]
                N = y_pred_neg.shape[0] if y_pred_neg.dim() == 1 else y_pred_neg.shape[-1]
                y_pred_neg_matrix = y_pred_neg.unsqueeze(0).expand(C, N)  # (C, N)
                # y_pred_pos_matrix = y_pred_pos.unsqueeze(1).expand(C, N)  # (C, N) -- repeats each pos value N times

                chem_val_data = {
                    "y_pred_pos": y_pred_pos,
                    "y_pred_neg": y_pred_neg_matrix
                }
                chem_val_result = ev.eval(chem_val_data)
                # Rank of each positive (1 = best)
                ranks = (y_pred_neg_matrix >= y_pred_pos.unsqueeze(1)).sum(dim=1) + 1
                print(f"Chemical validation for {rev_chemicals[chem]}:")
                print(f"\tChemical pos examples shape: {pos_examples.shape}")
                print(f"\tMRR: {chem_val_result['mrr_list'].mean()}")
                print(f"\thits@1: {chem_val_result['hits@1_list'].mean()}")
                print(f"\thits@3: {chem_val_result['hits@3_list'].mean()}")
                print(f"\thits@10: {chem_val_result['hits@10_list'].mean()}", flush=True)
                # ------------------------------------------------------------------
                # Construct one prediction score per gene
                # ------------------------------------------------------------------
                # Genes omitted from both sets are known training positives.
                # They are deliberately excluded from this filtered ranking,
                # so their score must not be left uninitialised (which could
                # otherwise put arbitrary values in the reported top-k).
                scores = torch.full(
                    (gene_max + 1,),
                    float('-inf'),
                    dtype=y_pred_pos.dtype,
                    device=y_pred_pos.device
                )
                labels = torch.zeros(
                    gene_max + 1,
                    dtype=torch.bool,
                    device=y_pred_pos.device
                )

                # Fill scores
                scores[pos_examples[0]] = y_pred_pos
                scores[neg_examples[0]] = y_pred_neg

                # Mark known positives
                labels[pos_examples[0]] = True

                # ------------------------------------------------------------------
                # Rank all genes
                # ------------------------------------------------------------------
                ranking = torch.argsort(scores, descending=True)
                ranked_labels = labels[ranking]
                n_pos = labels.sum().item()

                

                for k in [5, 10]:
                    hits = ranked_labels[:k].sum().item()
                    precision = hits / k
                    recall = hits / n_pos

                    precision_at[k].append(precision)
                    if k == 5:
                        chemical_precision_at_5[rev_chemicals[chem]] = precision
                    # if k == 5:
                    #     precisions_at_5.append(precision)

                    print(f"Top-{k} hits: {hits}/{k}")
                    print(f"Precision@{k}: {precision:.3f}")
                    print(f"Recall@{k}: {recall:.3f}", flush=True)
    precision_at = {k: np.mean(v) for k,v in precision_at.items()}
    return precision_at, chemical_precision_at_5
    # return float(np.mean(precisions_at_5)) if precisions_at_5 else float('nan')
# %%
# need to also ignore true edges from training data!!!!
def get_chemical_validation(edge_index, full_edge_index, chem_index, gene_max):
    """
    For each chemical in chem_index, extract links in edge_index where the tail is the chemical
    (i.e., edge_index[1, :] == chemical).

    For negative examples, ignores any gene with a true interaction with the chemical in either
    edge_index (validation) or full_edge_index (test/train).

    Yields:
        pos_examples: Tensor of shape (2, C): each column is (gene, chemical) for positive links
        neg_examples: Tensor of shape (2, N): each column is (neg_gene, chemical) for negative links
    """
    edge_tails = edge_index[1]
    edge_heads = edge_index[0]
    full_edge_tails = full_edge_index[1]
    full_edge_heads = full_edge_index[0]
    for chemical in chem_index:
        # Identify positive gene indices for this chemical (in validation)
        pos_mask = (edge_tails == chemical)
        pos_genes = edge_heads[pos_mask]
        pos_genes_unique = torch.unique(pos_genes)
        C = pos_genes_unique.shape[0]

        # If no positives, skip
        if C == 0:
            continue

        # Construct positive examples: (2, C) where row 0 is gene, row 1 is chemical (broadcast)
        pos_examples = torch.stack(
            [pos_genes_unique, torch.full_like(pos_genes_unique, chemical)],
            dim=0
        )  # (2, C)

        # Compute all true gene indices for this chemical from *both* sets
        # (i.e., ignore any gene with a true link to this chem in *either* validation or full set)
        # Get all genes with edge to this chemical in validation (pos_genes_unique already)
        # Get also from full set:
        true_mask_full = (full_edge_tails == chemical)
        true_genes_full = full_edge_heads[true_mask_full]
        # Union of all true genes for this chemical
        true_genes = torch.unique(torch.cat([pos_genes_unique, true_genes_full]))

        # Set of all possible gene indices (0 .. gene_max-1)
        all_genes = torch.arange(gene_max+1, dtype=true_genes.dtype, device=edge_index.device)
        # Remove ALL true genes to form negative candidates
        is_neg = torch.ones(gene_max+1, dtype=torch.bool, device=edge_index.device)
        is_neg[true_genes] = False
        neg_genes = all_genes[is_neg]  # (N,)
        N = neg_genes.shape[0]

        # Construct negative examples: (2, N) where row 0 is neg_gene, row 1 is chemical
        neg_examples = torch.stack(
            [neg_genes, torch.full_like(neg_genes, chemical)],
            dim=0
        )  # (2, N)

        yield pos_examples, neg_examples


def get_target_index_splits(edge_index, target_chemicals, message_ratio):
    """Yield disjoint target-chemical label edges and their message-passing edges.

    The held-out label edges are removed from message passing exactly as in the
    general-training phase.  Non-target edges remain available as biological
    context, but never contribute a supervised loss in this phase.
    """
    target_chemicals = torch.as_tensor(
        list(target_chemicals), device=edge_index.device, dtype=edge_index.dtype
    )
    target_idx = torch.where(torch.isin(edge_index[1], target_chemicals))[0]
    if target_idx.numel() == 0:
        return

    target_idx = target_idx[torch.randperm(target_idx.numel(), device=target_idx.device)]
    num_splits = max(1, int(np.ceil(1 / (1 - message_ratio))))
    for label_idx in torch.tensor_split(target_idx, num_splits):
        if label_idx.numel() == 0:
            continue
        message_mask = torch.ones(edge_index.shape[1], dtype=torch.bool,
                                  device=edge_index.device)
        message_mask[label_idx] = False
        yield torch.where(message_mask)[0], label_idx


def freeze_encoder(model, freeze=True):
    """Freeze/unfreeze embeddings and GraphSAGE layers, leaving the decoder trainable."""
    module = getattr(model, '_orig_mod', model)  # torch.compile compatibility
    for parameter in chain(module.node_embeddings.parameters(), module.gnn.parameters()):
        parameter.requires_grad = not freeze


def sample_target_negatives(pos_edges, training_positive_edges, max_h, max_t, neg_ratio):
    """Sample same-chemical negatives while rejecting training positives only.

    Validation links must remain unknown during optimisation; excluding them
    here would leak held-out labels into training.
    """
    num_neg = pos_edges.shape[1] * neg_ratio
    tails = pos_edges[1].repeat_interleave(neg_ratio)
    heads = torch.randint(0, max_h, (num_neg,), device=pos_edges.device)
    positive_keys = training_positive_edges[0] * max_t + training_positive_edges[1]

    # Rejection sampling is cheap here because chemical positives are sparse.
    invalid = torch.isin(heads * max_t + tails, positive_keys)
    while invalid.any():
        heads[invalid] = torch.randint(
            0, max_h, (invalid.sum().item(),), device=heads.device
        )
        invalid = torch.isin(heads * max_t + tails, positive_keys)
    return torch.stack((heads, tails))


def target_retrieval_metrics(model, graph, edge_type, validation_edges,
                             target_chemicals, score_batch_size=500_000):
    """Mean Precision/Recall@5 and @10 over target chemicals.

    Training positives are excluded from the candidate set.  This mirrors the
    hypothesis-generation setting: rank every *unknown* gene for one chemical.
    """
    gene_ids = graph['genes'].node_id
    train_edges = graph[edge_type].edge_index
    precisions_at_5, precisions_at_10 = [], []
    recalls_at_5, recalls_at_10 = [], []
    with torch.no_grad():
        for chemical in target_chemicals:
            positives = validation_edges[0, validation_edges[1] == chemical].unique()
            if positives.numel() == 0:
                continue
            known = train_edges[0, train_edges[1] == chemical].unique()
            # Validation positives remain scoreable positive candidates; they
            # are never placed in the negative candidate set.  Together with
            # the filtered training positives, this is the standard filtered
            # link-prediction evaluation protocol.
            candidates = gene_ids[~torch.isin(gene_ids, known)]
            edge_batch = torch.stack([
                candidates,
                torch.full_like(candidates, chemical),
            ])
            scores = []
            for batch in edge_batch.split(score_batch_size, dim=1):
                scores.append(model(graph, edges_to_predict={edge_type: batch},
                                    return_embs=False).flatten())
            top_genes = candidates[torch.cat(scores).topk(min(10, candidates.numel())).indices]
            top_5_size = min(5, top_genes.numel())
            hits_at_5 = torch.isin(top_genes[:top_5_size], positives).float().sum()
            hits_at_10 = torch.isin(top_genes, positives).float().sum()
            precisions_at_5.append(hits_at_5 / top_5_size)
            precisions_at_10.append(hits_at_10 / top_genes.numel())
            recalls_at_5.append(hits_at_5 / positives.numel())
            recalls_at_10.append(hits_at_10 / positives.numel())
    if not recalls_at_5:
        return {metric: float('nan') for metric in (
            'precision@5', 'recall@5', 'precision@10', 'recall@10'
        )}
    return {
        'precision@5': torch.stack(precisions_at_5).mean().item(),
        'recall@5': torch.stack(recalls_at_5).mean().item(),
        'precision@10': torch.stack(precisions_at_10).mean().item(),
        'recall@10': torch.stack(recalls_at_10).mean().item(),
    }
# %%
MODEL = 'complex' #'classifier', 'rotate'
#MODEL = 'rotate' #'classifier', 'rotate'
MODEL = 'classifier'
EPOCHS = 500
LR = 1e-2
REGULARIZATION = 1e-3
SEM_WEIGHT = 1e-2
NEG_WEIGHT = 0#1e-1
VAL_BATCH_SIZE = 500000
NEG_VAL_RATIO = 1000
FOLDS = 10
MESSAGE_RATIO = 0.9
neg_ratio = 60
num_batches = 10 
pos_weight = torch.tensor([400.], device=device)


print(f"SEM_WEIGHT: {SEM_WEIGHT}")
print(f"NEG_WEIGHT: {NEG_WEIGHT}")

print(f"neg val ratio: {NEG_VAL_RATIO}")
print(f"MESSAGE RATIO: {MESSAGE_RATIO}")
print(f"neg ratio: {neg_ratio}")
print(f"pos weight: {pos_weight}")

if device == 'cpu':
    gnn_channels = [4, 4]
    nn_channels = [4]
    node_dim = 4
    HIDDEN_DIM = 4
    EPOCHS = 7
    neg_ratio = 2
    num_batches = 2
    MESSAGE_RATIO = 0.5
    FINETUNE_EPOCHS = 2

else:
    HIDDEN_DIM = 128
    HIDDEN_DIM = 64
    #gnn_channels = [64, 128, 128]
    gnn_channels = [128, 128]
    #gnn_channels = [32, 64, 128]
    nn_channels = [32, 16, 4]
    #nn_channels = []
    nn_channels = [32, 8, 4]
    node_dim = 32
    node_dim = 64
    

domain_weights = {'mat_ent': 5}
print(f"domain weights: {domain_weights}")
print(f"gnn channels: {gnn_channels}")
print(f"nn channels: {nn_channels}")
print(f"node dim: {node_dim}")
print(f"hidden dim: {HIDDEN_DIM}")
print(f"val batch size: {VAL_BATCH_SIZE}")
print(f"scoring head: {MODEL}")

# NEED TO REMOVE THIS INFORMATION FROM (GENES, RO_0002200, QUALITY)!!!!!!!!!
# predicted_edge_types = [('genes', 'hasChemStressResistance', 'mat_ent')]
predicted_edge_types = [('genes', 'hasChemStressResistance_Decreased', 'mat_ent')]
suffixes = ['_Increased', '_Decreased']

if '_Increased' in predicted_edge_types[0][1] or '_Decreased' in predicted_edge_types[0][1]:
    assert len(predicted_edge_types) == 1
    pe = predicted_edge_types[0]
    rel = pe[1]
    base_rel = '_'.join(rel.split('_')[:-1])
    del graph[(pe[0], base_rel, pe[2])]
    del graph[(pe[2], 'rev_' + base_rel, pe[0])]

else:
    for pe in predicted_edge_types:
        for suf in suffixes:
            del graph[(pe[0], pe[1] + suf, pe[2])]
            del graph[(pe[2], 'rev_' + pe[1] + suf, pe[1])]
graph = graph.to(device)

message_edge_types=[e for e, v in graph.edge_index_dict.items() if v.shape[1] > 100]

print(f"pos weight: {pos_weight}")
print(f"neg ratio: {neg_ratio}")
print(f"num batches: {num_batches}")


ev = Evaluator(name='ogbl-biokg')
link_split = RandomLinkSplit(
    num_val=0.0,
    num_test=0.0,
    disjoint_train_ratio=0.2,
    edge_types=predicted_edge_types,
    rev_edge_types=[(e[2], f'rev_{e[1]}', e[0]) for e in predicted_edge_types],
    add_negative_train_samples=True,
    neg_sampling_ratio=neg_ratio
)

node_max_ids = {k: len(v) for k,v in graph.node_id_dict.items()}
# %%
if MODEL == 'classifier':
    model = Classifier(
        gnn_channels=gnn_channels,
        nn_channels=nn_channels,
        embedding_dims={k: node_dim for k in graph.node_types},
        number_classes={k: len(v) for k,v in graph.node_id_dict.items()},
        message_edge_types=message_edge_types,
        predicted_edge_types=predicted_edge_types,
        custom=False,
        target_chemical_indices=(TARGET_CHEMICALS if USE_TARGET_CHEMICAL_HEADS else None),
        target_head_channels=TARGET_HEAD_CHANNELS,
        device=device
    )
    loss_func = BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)
elif MODEL == 'complex':
    model = MultiDomainComplEx(
        gnn_channels=gnn_channels,
        embedding_dims={k: node_dim for k in graph.node_types},
        number_classes={k: len(v) for k,v in graph.node_id_dict.items()},
        message_edge_types=message_edge_types,
        predicted_edge_types=predicted_edge_types,
        hidden_dim=HIDDEN_DIM,
        use_layernorm=True,
        device=device
    )
    loss_func = BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)
elif MODEL == 'rotate':
    model = MultiDomainRotatE(
        gnn_channels=gnn_channels,
        embedding_dims={k: node_dim for k in graph.node_types},
        number_classes={k: len(v) for k,v in graph.node_id_dict.items()},
        message_edge_types=message_edge_types,
        predicted_edge_types=predicted_edge_types,
        hidden_dim=HIDDEN_DIM,
        gamma=12.0,
        use_layernorm=True,
        device=device
    )
    loss_func = F.logsigmoid
else:
    raise NotImplementedError(f"not implemented model {MODEL}")

print(f"predicting: {predicted_edge_types}")
# assert False
model.to(device)
if device != 'cpu':
    model = torch.compile(model)
# edge_idx_lens = {k: train_graph.edge_index_dict[k].shape[1] for k in predicted_edge_types}
best_metric = []
best_models = []
best_chemical_precision_5 = {k: [] for k in chemicals.keys()}
final_mrr = []
# Create a list to store information about all folds
# saved_folds = []

for k, v in graph.node_id_dict.items():
    graph[k].node_id = torch.arange(len(v))
for fold_num, fold_info in enumerate(generate_folds(graph, predicted_edge_types, num_folds=FOLDS, file_path='saved_folds.json')):
    best_mrr = -np.inf
    best_target_precision_at_5 = -np.inf
    best_target_precision_at = {5: -np.inf, 10: -np.inf}
    print('fold num:', fold_num)
    
    model.reset_parameters()
    best_model = copy.deepcopy(model)
    optimizer = torch.optim.AdamW([{'params': model.node_embeddings.parameters(), 'weight_decay': 0},{'params': model.gnn.parameters(), 'weight_decay': REGULARIZATION/10}, {'params': chain(model.lin_layers.parameters(), model.last_layer.parameters()), 'weight_decay': REGULARIZATION}], lr=LR)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=10, min_lr=1e-6)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,   # start at 10% of lr
        end_factor=1.0,
        total_iters=5        # warmup for 5 epochs
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=495,           # remaining epochs
        eta_min=1e-5
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[5]
    )

    train_graph = copy.copy(graph).to(device)
    
    for e, d in fold_info.items():
        train_graph[e].edge_index = d['train'].T
        train_graph[(e[2], f'rev_{e[1]}', e[0])].edge_index = d['rev'].T

    val_graph = copy.copy(train_graph).to(device)
    # fi = {}
    for e, d in fold_info.items():
        val_graph[e].edge_label_index = d['val'].T
        if 'neg' in d:
            val_graph[e].neg_edge_label_index = d['neg']
        else:
            neg_index = sample_neg_ex(
                    d['val'].shape[0],
                    len(val_graph[e[0]].node_id),
                    len(val_graph[e[2]].node_id),
                    neg_ratio=int(NEG_VAL_RATIO / 2),
                    device=device,
                    head=d['val'][:,0]
                )
            # print(neg_index.shape)
            neg_index = torch.cat((neg_index, sample_neg_ex(
                    d['val'].shape[0],
                    len(val_graph[e[0]].node_id),
                    len(val_graph[e[2]].node_id),
                    neg_ratio=int(NEG_VAL_RATIO / 2),
                    device=device,
                    tail=d['val'][:,1]
                )), dim=-1)
            # print(neg_index.shape)
            val_graph[e].neg_edge_label_index = neg_index
    # break
    # saved_folds.append(fi)
    split_train_graph = copy.copy(train_graph).to(device)
    edge_idx_lens = {k: train_graph.edge_index_dict[k].shape[1] for k in predicted_edge_types}
    since_improved = 0
    for epoch in tqdm.tqdm(range(EPOCHS)):
        # break
        model.train()
        epoch_loss = 0
        epoch_sem_loss = 0

        for edge_indices in get_balanced_index_splits(
                            edge_idx_lens, MESSAGE_RATIO, train_graph,
                            predicted_edge_types, TARGET_CHEMICALS,
                            target_ratio=0.4,      # tune this
                        ):
            for k, (message_edges, label_edges) in edge_indices.items():
                pos_edges = train_graph[k].edge_index[:, label_edges].to(device)

                head_max_id = node_max_ids[k[0]]
                tail_max_id = node_max_ids[k[2]]

                message_edge_index = train_graph[k].edge_index[:, message_edges]
                rev = (k[2], 'rev_' + k[1], k[0])
                rev_message_edge_index = train_graph[rev].edge_index[:, message_edges]

                num_pos = pos_edges.shape[1]

                neg_edges = sample_neg_ex(
                        len(label_edges),
                        head_max_id,
                        tail_max_id,
                        neg_ratio=neg_ratio,
                        true_edges=train_graph[k].edge_index
                    ).to(device)
                neg_edges = torch.cat((neg_edges, sample_neg_ex(
                        len(label_edges),
                        head_max_id,
                        tail_max_id,
                        head=pos_edges[0,:],
                        neg_ratio=neg_ratio,
                        true_edges=train_graph[k].edge_index
                        #device=device
                    ).to(device)), dim=1)
                neg_edges = torch.cat((neg_edges, sample_neg_ex(
                        len(label_edges),
                        head_max_id,
                        tail_max_id,
                        tail=pos_edges[1,:],
                        neg_ratio=8*neg_ratio,
                        true_edges=train_graph[k].edge_index
                    ).to(device)), dim=1)

                split_train_graph[k].edge_index = message_edge_index.to(device)

                split_train_graph[rev].edge_index = rev_message_edge_index.to(device)
                edge_label_indices = torch.cat((pos_edges, neg_edges), dim=1)
                edge_labels = torch.cat((torch.ones_like(pos_edges[0,:], dtype=torch.float), torch.zeros_like(neg_edges[0,:], dtype=torch.float)))


                split_train_graph[k].edge_label_index = edge_label_indices.to(device)
                split_train_graph[k].edge_label = edge_labels.to(device)
            split_train_graph = split_train_graph.to(device)
        
            optimizer.zero_grad()
                
            if SEM_WEIGHT > 0:
                y, x_dicts = model(split_train_graph, edges_to_predict=split_train_graph.edge_label_index_dict, return_embs=True)
                sem_loss, neg_sem_loss, layer_losses = box_loss(x_dicts,
                                    gci0, loss_type='distance',
                                    neg=(NEG_WEIGHT > 0),
                                    return_layer_loss=True,
                                    domain_weights=domain_weights)
            else:    
                y = model(split_train_graph, edges_to_predict=split_train_graph.edge_label_index_dict, return_embs=False)

            all_labels = torch.tensor([], device=device)
            for k, v in split_train_graph.edge_label_dict.items():
                all_labels = torch.cat((all_labels, v.to(device)), dim=0)
            if MODEL == 'rotate':
                neg_mask = all_labels == 0
                pos_mask = all_labels == 1
                scores = y.squeeze()
                loss = -(pos_weight * loss_func(scores[pos_mask])).sum() - loss_func(-scores[neg_mask]).sum()
            else:
                loss = loss_func(y.squeeze(), all_labels)
            epoch_loss += loss.detach().item()
            if SEM_WEIGHT > 0:
                sl = SEM_WEIGHT * (sem_loss + NEG_WEIGHT * neg_sem_loss)
                loss += sl
                epoch_sem_loss += sl.detach().item()
            
            print(f"train loss: {loss}")
            # print(f"train acc: {acc}")
            loss.backward()
            optimizer.step()
            # print('label shape:', all_labels.shape)

            if device != 'cpu':
                if SEM_WEIGHT > 0:
                    del x_dicts, all_labels, y
                else:
                    del all_labels, y
                torch.cuda.empty_cache()
        print(f"epoch train loss: {epoch_loss}")
        print(f"epoch sem loss: {epoch_sem_loss}", flush=True)
        # continue
        model.eval()
        val = eval_full(model, val_graph)

        target_precision_at, _ = eval_chemicals(model, val_graph, chem_indices)
        print(f"Mean target Precision@5: {target_precision_at[5]:.4f}")
        print(f"Mean target Precision@10: {target_precision_at[10]:.4f}", flush=True)
            
        if epoch > 4:
            print("current:", target_precision_at, flush=True)
            print("best:", best_target_precision_at, flush=True)
            if target_precision_at[5] > best_target_precision_at[5]:
                print('BEST TARGET MODEL!!', flush=True)
                best_target_precision_at = target_precision_at
                best_mrr = val['mrr_list'].mean().cpu().item()
                since_improved = 0
                best_model = copy.deepcopy(model)
            elif (target_precision_at[5] == best_target_precision_at[5]
                    and target_precision_at[10] > 
                            best_target_precision_at[10]):
                print('BEST TARGET MODEL!!', flush=True)
                best_target_precision_at = target_precision_at
                best_mrr = val['mrr_list'].mean().cpu().item()
                since_improved = 0
                best_model = copy.deepcopy(model)
            elif (target_precision_at[5] == best_target_precision_at[5]
                    and target_precision_at[10] == 
                            best_target_precision_at[10]
                        and val['mrr_list'].mean() > best_mrr):
                print('BEST TARGET MODEL!!', flush=True)
                best_mrr = val['mrr_list'].mean().cpu().item()
                best_target_precision_at = target_precision_at
                since_improved = 0
                best_model = copy.deepcopy(model)
            else:
                since_improved += 1
        scheduler.step()

        if epoch == 0:
            nbr_params = count_trainable_params(model)
            print("Trainable params:", nbr_params, flush=True)

        if since_improved > 70:
            print(target_precision_at)
            print('target Precision@5 has not improved in 70 epochs, early stopping...', flush=True)
            break


    # ------------------------------------------------------------------
    # Stage 2: target-chemical retrieval fine-tuning.
    # Start from the best general model, freeze the encoder, and train only
    # the shared decoder on within-chemical (fixed-tail) negative examples.
    # Select the checkpoint by Precision@5, the metric closest to the wet-lab
    # candidate-selection decision, not by global MRR.
    # ------------------------------------------------------------------
    model = copy.deepcopy(best_model).to(device)
    freeze_encoder(model, freeze=False)
    finetune_optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=FINETUNE_LR,
        weight_decay=REGULARIZATION,
    )
    finetune_loss = BCEWithLogitsLoss(
        reduction='mean',
        pos_weight=torch.tensor([FINETUNE_NEG_RATIO], device=device),
    )
    best_target_score = -np.inf
    best_target_model = copy.deepcopy(model)
    edge_type = predicted_edge_types[0]
    reverse_edge_type = (edge_type[2], f'rev_{edge_type[1]}', edge_type[0])
    training_target_edges = train_graph[edge_type].edge_index
    best_precision_before = copy.copy(best_target_precision_at)
    for finetune_epoch in range(FINETUNE_EPOCHS):
        if finetune_epoch == round(FINETUNE_EPOCHS/2) and False:
            print('freezing encoder after half finetuning', flush=True)
            freeze_encoder(model, freeze=True)
        model.train()
        target_loss = 0.0
        for message_edges, label_edges in get_target_index_splits(
                train_graph[edge_type].edge_index, TARGET_CHEMICALS,
                FINETUNE_MESSAGE_RATIO):
            pos_edges = train_graph[edge_type].edge_index[:, label_edges]
            neg_edges = sample_target_negatives(
                pos_edges,
                training_target_edges,
                node_max_ids[edge_type[0]],
                node_max_ids[edge_type[2]],
                FINETUNE_NEG_RATIO,
            )
            split_train_graph[edge_type].edge_index = train_graph[edge_type].edge_index[:, message_edges]
            split_train_graph[reverse_edge_type].edge_index = train_graph[reverse_edge_type].edge_index[:, message_edges]
            split_train_graph[edge_type].edge_label_index = torch.cat((pos_edges, neg_edges), dim=1)
            split_train_graph[edge_type].edge_label = torch.cat((
                torch.ones(pos_edges.shape[1], device=device),
                torch.zeros(neg_edges.shape[1], device=device),
            ))

            finetune_optimizer.zero_grad()
            predictions = model(
                split_train_graph,
                edges_to_predict={edge_type: split_train_graph[edge_type].edge_label_index},
                return_embs=False,
            ).flatten()
            loss = finetune_loss(predictions, split_train_graph[edge_type].edge_label)
            loss.backward()
            finetune_optimizer.step()
            target_loss += loss.detach().item()

        model.eval()
        target_metrics = target_retrieval_metrics(
            model, val_graph, edge_type, val_graph[edge_type].edge_label_index,
            TARGET_CHEMICALS, VAL_BATCH_SIZE,
        )
        target_score = target_metrics['precision@5']
        target_precision_at = {k: target_metrics[f'precision@{k}']
                                    for k in [5, 10]}
        print(f"fine-tune epoch {finetune_epoch + 1}/{FINETUNE_EPOCHS}: "
              f"loss={target_loss:.4f}, target Precision@5={target_score:.4f}, "
              f"Recall@5={target_metrics['recall@5']:.4f}, "
              f"Precision@10={target_metrics['precision@10']:.4f}, "
              f"Recall@10={target_metrics['recall@10']:.4f}", flush=True)
        # if not np.isnan(target_score) and target_score > best_target_score:
        if target_precision_at[5] > best_target_precision_at[5]:
            best_target_precision_at = target_precision_at
            # best_target_score = target_score
            best_target_model = copy.deepcopy(model)
        elif (target_precision_at[5] == best_target_precision_at[5] and
                target_precision_at[10] > best_target_precision_at[10]):
            best_target_precision_at = target_precision_at
            # best_target_score = target_score
            best_target_model = copy.deepcopy(model)

    

    _, chem_precision = eval_chemicals(best_model, val_graph, chem_indices)
    final_val = eval_full(best_model, val_graph)
    final_mrr.append(final_val['mrr_list'].mean().cpu().item())
    for k,v in chem_precision.items():
        best_chemical_precision_5[k].append(v)
    best_model = best_target_model
    # print(f"Best MRR: {best_mrr}")
    for k in [5, 10]:
        print(f"Best target Precision@{k} before fine-tuning: {best_precision_before[k]:.4f}")
        print(f"Best target Precision@{k} after fine-tuning: {best_target_precision_at[k]:.4f}", flush=True)
    best_metric.append(best_target_precision_at[5])
    

    best_model.to('cpu')
    best_models.append(best_model._orig_mod)

print(f"Best models: {np.mean(best_metric)} +- {np.std(best_metric)}", flush=True)
print(f"Best models MRR: {np.mean(final_mrr)} +- {np.std(final_mrr)}", flush=True)
print("Chemical specific precision@5:", flush=True)
for k,v in best_chemical_precision_5.items():
    print(f"{k}: {np.mean(v)} +- {np.std(v)}", flush=True)
# %%
# with open(os.path.join(BASE, 'saved_folds.json'), 'w') as fi:
#     json.dump(saved_folds, fi)
fn = os.path.join(BASE, f'trained_yeast_models/cv_model_{time.strftime("%Y%m%d-%H%M%S")}.pkl')

print(f"saving best model to {fn}...")
with open(fn, 'wb') as fo:
    pickle.dump(best_models, fo)
# %%
