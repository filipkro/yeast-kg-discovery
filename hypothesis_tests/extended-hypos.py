# %%
import pandas as pd
import os, pickle

# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE)
# %%
chemicals = {'Hydrogen peroxide': 'CHEBI_16240',
'Tunicamycin': 'CHEBI_29699',
'Caffeine': 'CHEBI_27732',
'NaCl': 'CHEBI_26710',
'Ethanol': 'CHEBI_16236',
'Acetic acid': 'CHEBI_15366'}
# %%
with open(os.path.join(BASE, 'kg.pkl'), 'rb') as f:
    g = pickle.load(f)
    graph = g['graph']
    index = g['index']
rev_gene_index = {v: k.split('#')[-1] for k,v in index['genes'].items()}
# %%
HYPO_FILE = "hypothesis_tests.csv"
tested_df = pd.read_csv(os.path.join(BASE, "hypo_tests/", HYPO_FILE))
tested_df.head()
# %%
gene_chem_exp = {}
for filename in os.listdir(os.path.join(BASE, "ranked_genes/used_genes")):
    if filename.endswith(".csv"):
        group = filename.replace(".csv", "")#.replace("_", " ")
        
        print(group)
        # used_genes[group] = []
        df = pd.read_csv(os.path.join(BASE, f"ranked_genes/used_genes/{filename}"))
        for _, row in df.iterrows():
            gene = row['genes']
            chem = row['chemicals']
            hypo_info = {'group': group, 'prob': row.prob, 'rank': row.name+1}
            gene_chem_exp[(gene, chem)] = hypo_info

            # print(row.name)
            # break
# %%
gene_chem_exp_unrestricted = {}
num_unrestricted_predictions = {}
for filename in os.listdir(os.path.join(BASE, "ranked_genes/unrestricted_gene_sets")):
    if filename.endswith(".csv"):
        group = filename.replace(".csv", "")#.replace("_", " ")
        
        print(group)
        # used_genes[group] = []
        df = pd.read_csv(os.path.join(BASE, f"ranked_genes/unrestricted_gene_sets/{filename}"))
        # i = 0
        num = None
        for _, row in df.iterrows():
            gene = row['genes']
            chem = row['chemicals']
            if num is None:
                num = len(df)
                num_unrestricted_predictions[chem] = num
            hypo_info = {'group': group, 'prob': row.prob, 'rank': row.name+1}
            gene_chem_exp_unrestricted[(gene, chem)] = hypo_info
# %%
for k in gene_chem_exp[('YJL204C', 'NaCl')].keys():
    tested_df[k] = tested_df.apply(lambda x: gene_chem_exp[(x.orf, x.chemical)][k], axis=1)

tested_df['unrestricted_rank'] = tested_df.apply(lambda x: gene_chem_exp_unrestricted[(x.orf, x.chemical)]['rank'], axis=1)


# %%
# num_unrestricted_predictions
tested_df['rank_percentile'] = tested_df.apply(lambda x: 1 - (x.unrestricted_rank - 1)/(num_unrestricted_predictions[x.chemical] - 1), axis=1)


# %%
def get_neighbors(node_name, node_domain, domain_prefix):
    node_idx = index[node_domain][f'{domain_prefix}{node_name}']
    num_neighbors = 0
    for e, edges in graph.edge_index_dict.items():
        if e[2] == node_domain:
            num_neighbors += (edges[1,:] == node_idx).sum().item()
    return num_neighbors

def gene_neighbors(gene):
    return get_neighbors(gene, 'genes', 'http://sgd-kg.project-genesis.io#')

def chemical_neighbors(chemical):
    return get_neighbors(chemicals[chemical], 'mat_ent', 'http://purl.obolibrary.org/obo/')

# %%
tested_df['gene_neighbors'] = tested_df.apply(lambda x: gene_neighbors(x.orf), axis=1)
tested_df['chemical_neighbors'] = tested_df.apply(lambda x: chemical_neighbors(x.chemical), axis=1)
tested_df['summed_neighbors'] = tested_df['gene_neighbors'] + tested_df['chemical_neighbors']
# %%
tested_df.to_csv(os.path.join(BASE, "hypo_tests/hypothesis_tests_with_predictions_info.csv"), index=False)
# %%
