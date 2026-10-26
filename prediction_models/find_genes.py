# %%
import torch
import os, pickle
import pandas as pd
import torch.nn.functional as F
# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = 'cv_model_20260720-174936.pkl'
EDGE = ('genes', 'hasChemStressResistance_Decreased', 'mat_ent')
print(BASE)

GENE_SET = 'restricted'
GENE_SET = 'full'
# %%
chemicals = {'Hydrogen peroxide': 'CHEBI_16240',
'Tunicamycin': 'CHEBI_29699',
'Caffeine': 'CHEBI_27732',
'Table salt': 'CHEBI_26710',
'Ethanol': 'CHEBI_16236',
'Acetic acid': 'CHEBI_15366'}
# %%
with open(os.path.join(BASE, 'kg.pkl'), 'rb') as f:
    g = pickle.load(f)
    graph = g['graph']
    index = g['index']
rev_gene_index = {v: k.split('#')[-1] for k,v in index['genes'].items()}

# %%
chemical_idxs = {k: index['mat_ent'][f'http://purl.obolibrary.org/obo/{chebi}']
             for k, chebi in chemicals.items()}
edges_oi = graph[EDGE].edge_index

for k, idx in chemical_idxs.items():
    linked_genes = edges_oi[0,edges_oi[1,:] == idx]
    print(f"{k}: {len(linked_genes)} genes")
# %%
if GENE_SET == 'restricted':
    matav_df = pd.read_excel(os.path.join(BASE, 'gene_sets/Matav50.xlsx'), sheet_name='DATA')
    all_genes = matav_df['ORF name'].unique().tolist()
    all_gene_indices = {k: index['genes'][f'http://sgd-kg.project-genesis.io#{k}']
                        for k in all_genes if f'http://sgd-kg.project-genesis.io#{k}' in index['genes']}
    print(matav_df.head())
    print(len(matav_df))
    print(len(all_genes))
    print(len(all_gene_indices))
elif GENE_SET == 'full':
    all_gene_indices = {k: v for k, v in index['genes'].items()}
    print(len(all_gene_indices))
else:
    raise ValueError(f"Unknown GENE_SET: {GENE_SET}")

# %%
unknown_genes_df = pd.read_csv(os.path.join(BASE, 'gene_sets/unknown_genes.csv'))
print(unknown_genes_df.head())
unknown_genes = unknown_genes_df['Gene_stable_ID'].unique().tolist()
unknown_gene_indices = {k: index['genes'][f'http://sgd-kg.project-genesis.io#{k}']
                    for k in unknown_genes if f'http://sgd-kg.project-genesis.io#{k}' in index['genes']}
print(len(unknown_genes_df))
print(len(unknown_genes))
print(len(unknown_gene_indices))

# %%
chemical_indices = {index['mat_ent'][f'http://purl.obolibrary.org/obo/{chebi}']:
                    chem for chem, chebi in chemicals.items()}
# %%
def get_gene_candidates(gene_pool, chemical_idx, edge_index):
    candidates = []
    linked_genes = set(edge_index[0,edge_index[1,:] == chemical_idx].tolist())
    linked_genes = set()
    links = set()
    for c in chemical_idx:
        for g in edge_index[0,edge_index[1,:] == c].tolist():
            links.add((g, c))

    for g in gene_pool:
        for c in chemical_idx:
            if (g, c) not in links:
                candidates.append((g, c))

    return torch.tensor(candidates).T
# %%
with open(os.path.join(BASE, 'prediction_models', MODEL_FILE), 'rb') as fi:
    models = pickle.load(fi)#[:10]
# %%
hypo_dfs = {}
for chem, chebi in chemicals.items():
    chem_idx = [index['mat_ent'][f'http://purl.obolibrary.org/obo/{chebi}']]
    
    print(chem)
    candidates = get_gene_candidates(list(all_gene_indices.values()), chem_idx,
                                     graph[EDGE].edge_index)
    predictions = torch.tensor([])
    for m in models:
        m.to('cpu')
        m.device = 'cpu'
        m.gnn.device = 'cpu'
        m.eval()
        with torch.no_grad():
            pred = m(graph, edges_to_predict={EDGE: candidates},
                     return_embs=False)
            predictions = torch.cat((predictions, pred), dim=-1)

    predictions = predictions.mean(dim=-1).sort(descending=True)

    confidences = predictions.values
    genes = [rev_gene_index[g] for g in
             candidates[0, predictions.indices].tolist()]
    predicted_chemicals = [chemical_indices[c] for c in candidates[1, predictions.indices].tolist()]
    chem_data = {'genes': genes, 'chemicals': predicted_chemicals, 'conf': confidences.tolist(),
                 'prob': F.sigmoid(confidences).tolist()}
    hypo_dfs[chem] = pd.DataFrame(chem_data)
    if GENE_SET == 'restricted':
        fp = os.path.join(BASE, f"ranked_genes/{chem.replace(' ', '_')}.csv")
    elif GENE_SET == 'full':
        fp = os.path.join(BASE, f"ranked_genes/unrestricted_gene_sets/{chem.replace(' ', '_')}.csv")
      
    hypo_dfs[chem].to_csv(fp)

# %%
chem_idxs = [index['mat_ent'][f'http://purl.obolibrary.org/obo/{chebi}']
             for chebi in chemicals.values()]
candidates = get_gene_candidates(list(unknown_gene_indices.values()), chem_idxs,
                                 graph[EDGE].edge_index)

predictions = torch.tensor([])
for m in models:
    m.to('cpu')
    m.device = 'cpu'
    m.gnn.device = 'cpu'
    m.eval()
    with torch.no_grad():
        pred = m(graph, edges_to_predict={EDGE: candidates},
                    return_embs=False)
        predictions = torch.cat((predictions, pred), dim=-1)
    
predictions = predictions.mean(dim=-1).sort(descending=True)

confidences = predictions.values
genes = [rev_gene_index[g] for g in candidates[0, predictions.indices].tolist()]
predicted_chemicals = [chemical_indices[c] for c in candidates[1, predictions.indices].tolist()]
unknown_hypos_df = pd.DataFrame({'genes': genes, 'chemicals': predicted_chemicals, 'conf': confidences.tolist(),
                                 'prob': F.sigmoid(confidences).tolist()})

fp = os.path.join(BASE, f"ranked_genes/unknown_genes.csv")
unknown_hypos_df.to_csv(fp)
# %%
