# %%
import pandas as pd
import os

# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE)
# %%
tested_df = pd.read_csv(os.path.join(BASE, "od_analysis/plate_layout.csv"))
print(tested_df.head())
# %%
tested_set = set()
for _, row in tested_df[tested_df['removed'].isnull()].iterrows():
    
    tested_set.add((row.orf, row.chemical))
print(tested_set)

# %%
used_genes = {}
for filename in os.listdir(os.path.join(BASE, "ranked_genes")):
    if filename.endswith(".csv"):
        group = filename.replace(".csv", "")#.replace("_", " ")
        
        print(group)
        used_genes[group] = []
        df = pd.read_csv(os.path.join(BASE, f"ranked_genes/{filename}"))
        for _, row in df.iterrows():
            if len(used_genes[group]) >= 5:
                break
            gene = row['genes']
            chem = row['chemicals']
            if chem == 'Table salt':
                chem = 'NaCl'
                row['chemicals'] = 'NaCl'
            
            if (gene, chem) in tested_set:
                used_genes[group].append(row)
# %%
for k,v in used_genes.items():
    df = pd.DataFrame(v).drop('Unnamed: 0', axis=1)
    df.to_csv(os.path.join(BASE, f"ranked_genes/used_genes/{k}.csv"), index=False)
