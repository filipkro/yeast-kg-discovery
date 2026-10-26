# %%
import os
import pandas as pd
import json
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE)
# %%
INPUT_DIR = os.path.join(BASE, "ranked_genes")
experiments = {}
for chem_file in os.listdir(INPUT_DIR):
    if chem_file.endswith(".csv"):
        df = pd.read_csv(os.path.join(INPUT_DIR, chem_file))
        # print(chem_file)
        for i, row in df.iterrows():
            g = row['genes']
            c = row['chemicals']
            if g not in experiments:
                experiments[g] = [c]
            else:
                experiments[g].append(c)
            if i == 4:
                break
print(len(experiments))
# %%
with open(os.path.join(BASE, "ranked_genes/experiments.json"), "w") as f:
    json.dump(experiments, f, indent=4)
# %%
