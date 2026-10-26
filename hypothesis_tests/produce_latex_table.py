# %%
# from pathlib import Path
import os

import pandas as pd
from scipy.stats import t
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(BASE)
# %%
INPUT_FILE = os.path.join(BASE, "hypo_tests/hypothesis_tests_with_predictions_info.csv")
OUTPUT_FILE = os.path.join(BASE, "hypo_tests/true_hypos_table.txt")
OUTPUT_FILE_FULL = os.path.join(BASE, "hypo_tests/hypos_table_full.txt")


# # Display names for the chemicals
# CHEMICAL_NAMES = {
#     "Hydrogen_peroxide": r"\makecell[l]{Hydrogen\\peroxide}",
#     "Tunicamycin": "Tunicamycin",
#     "Caffeine": "Caffeine",
#     "Acetic_acid": r"\makecell[l]{Acetic acid}",
#     "Ethanol": r"\makecell[l]{Ethanol}",
#     "NaCl": r"\makecell[l]{NaCl}",
#     "unknown_genes": r"\makecell[l]{Poorly\\characterized}",
# }

# # Desired order
# ORDER = [
#     "Hydrogen_peroxide",
#     "Tunicamycin",
#     "Caffeine",
#     "Acetic_acid",
#     "Ethanol",
#     "NaCl",
#     "unknown_genes",
# ]
df = pd.read_csv(INPUT_FILE)
# %%
lines = []

for row in df.itertuples():
    if not row.significant:
        break
    chem = row.chemical
    col_string = "\\rowcolor{gray!15}" if row.group == "unknown_genes" else ""
    p = f"{row.p:.3f}" if row.p > 0.001 else f"$<$0.001"
    fdr = f"{row.FDR:.3f}" if row.FDR > 0.001 else f"$<$0.001"
    lines.append(rf"{col_string}{chem} & {row.orf} & {row.rank}  & {row.interaction:.3f} & {100*row.fitness_reduction:.1f}\% & {p} & {fdr} \\")

with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(lines))

print(f"Wrote {OUTPUT_FILE}")
# %%

lines = []

for row in df.itertuples():

    chem = row.chemical
    col_string = "\\rowcolor{gray!15}" if row.group == "unknown_genes" else ""
    support = "$\checkmark$" if row.significant else ""
    p = f"{row.p:.3f}" if row.p > 0.001 else f"$<$0.001"
    fdr = f"{row.FDR:.3f}" if row.FDR > 0.001 else f"$<$0.001"
    lines.append(rf"{col_string}{chem} & {row.orf} {support}& {row.rank} / {row.unrestricted_rank} & {row.interaction:.3f} & {row.upper_bound:.3f} & {100*row.fitness_reduction:.1f}\% & {p} & {fdr} \\")

with open(OUTPUT_FILE_FULL, "w") as f:
    f.write("\n".join(lines))

print(f"Wrote {OUTPUT_FILE_FULL}")
# %%
