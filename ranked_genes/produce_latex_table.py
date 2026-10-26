# %%
# from pathlib import Path
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(BASE)
# %%
INPUT_DIR = os.path.join(BASE, "ranked_genes/used_genes")
OUTPUT_FILE = "predicted_genes_table.txt"

# Display names for the chemicals
CHEMICAL_NAMES = {
    "Hydrogen_peroxide": r"\makecell[l]{Hydrogen\\peroxide}",
    "Tunicamycin": "Tunicamycin",
    "Caffeine": "Caffeine",
    "Acetic_acid": r"\makecell[l]{Acetic acid}",
    "Ethanol": r"\makecell[l]{Ethanol}",
    "NaCl": r"\makecell[l]{NaCl}",
    "unknown_genes": r"\makecell[l]{Poorly\\characterized}",
}

# Desired order
ORDER = [
    "Hydrogen_peroxide",
    "Tunicamycin",
    "Caffeine",
    "Acetic_acid",
    "Ethanol",
    "NaCl",
]
#     "unknown_genes",
# ]

GENE_NAMES = {'YML097C' : 'VPS9',
              'YOR070C' : 'GYP1',
              'YOR322C' : 'LBD19',
              'YMR015C' : 'ERG5',
              'YNL064C' : 'YDJ1',
              'YPL065W' : 'VPS28',
              'YOL018C' : 'TLG2',
              'YGR092W' : 'DBF2',
              'YML007W' : 'YAP1',
              'YJL183W' : 'MNN11',
              'YGR036C' : 'CAX4',
              'YIR023W' : 'DAL81',
              'YLR087C' : 'CSF1',
              'YER083C' : 'GET2',
              'YLR386W' : 'VAC14',
              'YNR006W' : 'VPS27',
              'YDR069C' : 'DOA4',
              'YKL212W' : 'SAC1',
              'YDR495C' : 'VPS3',
              'YAL021C' : 'CCR4',
              'YDR137W' : 'RGP1',
              'YNL294C' : 'MSB3',
              'YJL204C' : 'RCY1',
              'YOL081W' : 'IRA2',
              'YOR332W' : 'VMA4',
              'YLR182W' : 'SWI6',
              'YHR060W' : 'VMA22',
              'YPL055C' : 'LGE1',
              'YLR426W' : 'TDA5',
              'YKL037W' : 'AIM26',
              'YHR142W' : 'SPO12',
              'YNL133C' : 'FYV6'}

GENE_DESC = {'YML097C' : 'Involved in ergosterol biosynthesis',
              'YER083C' : 'Involved in insertion of proteins in to ER membrane',
              'YOR070C' : 'Cis-golgi GAP for yeast Rabs',
              'YOR322C' : 'Involved in endocytosis',
              'YMR015C' : 'Involved in vacuolar transport',
              'YNL064C' : 'Regulates protein folding',
              'YPL065W' : 'Involved in endosomal protein sorting',
              'YOL018C' : 'Subunit of the SNARE complex',
              'YGR092W' : 'Kinase involved in transcription and stress response',
              'YML007W' : 'Transcription factor involved in oxidative stress response',
              'YJL183W' : 'Subunit of Golgi mannosyltransferase complex',
              'YGR036C' : 'Involved in protein N-glycosylation in ER',
              'YIR023W' : 'Positive regulator of nitrogen degradation pathways',
              'YLR087C' : 'Required for fermentation at low temperatures',
              'YLR386W' : 'Involved in vacuolar transport',
              'YNR006W' : 'Required for recycling Golgi proteins',
              'YDR069C' : 'Ubiquitin hydrolase',
              'YKL212W' : 'Regulates sphingolipid biosynthesis',
              'YDR495C' : 'Required for vacuolar protein sorting and processing',
              'YAL021C' : 'Involved in regulation of gene expression',
              'YDR137W' : 'Subunit of the Golgi membrane exchange factor',
              'YNL294C' : 'Regulates endocytosis',
              'YJL204C' : 'Involved in endocytic recycling',
              'YOL081W' : 'Regulate cAMP levels under nutrient limitation',
              'YOR332W' : 'Required for vacuolar H+-ATPase function',
              'YLR182W' : 'Transcription cofactor involved in meiotic gene expression',
              'YHR060W' : 'Required for vacuolar H+-ATPase function',
              'YPL055C' : 'Null mutants have abnormally large cells',
              'YLR426W' : 'Detected in highly purified mitochondria',
              'YKL037W' : 'Elevated mitochondrial genome loss for null mutants',
              'YHR142W' : 'Positive regulator of mitotic shift',
              'YNL133C' : 'Proposed to be involved in pre-mRNA splicing'}

lines = []

for chem in ORDER:
    df = pd.read_csv(f"{INPUT_DIR}/{chem}.csv")

    # Keep highest probability occurrence of each gene
    df = (
        df.sort_values("prob", ascending=False)
          .drop_duplicates(subset="genes", keep="first")
          .head(5)
    )

    chem_name = CHEMICAL_NAMES[chem]

    first = True
    for _, row in df.iterrows():
        gene = row["genes"]
        prob = row["prob"]
        st_name = GENE_NAMES[gene]
        desc = GENE_DESC[gene]

        if first:
            lines.append(
                rf"\multirow{{5}}{{*}}{{{chem_name}}}"
                rf" & {gene} / {st_name} & {prob:.3f} & {desc} \\"
            )
            first = False
        else:
            lines.append(
                rf"& {gene} / {st_name} & {prob:.3f} & {desc} \\"
            )

    # if chem != ORDER[-1]:
    lines.append(r"\addlinespace")
    lines.append("")

lines.append(r"\multicolumn{4}{l}{\textbf{Uncharacterised gene group}} \\")
lines.append("")
# for _ in range(10):
#     lines.append("\n")

df = pd.read_csv(f"{INPUT_DIR}/unknown_genes.csv")
# Keep highest probability occurrence of each gene
df = (
    df.sort_values("prob", ascending=False)
        .drop_duplicates(subset="genes", keep="first")
        .head(5)
)

chem_name = CHEMICAL_NAMES[chem]

# first = True
for _, row in df.iterrows():
    gene = row["genes"]
    prob = row["prob"]
    st_name = GENE_NAMES[gene]
    desc = GENE_DESC[gene]
    chem = row["chemicals"]

    lines.append(rf"{chem} & {gene} / {st_name} & {prob:.3f} & {desc} \\")


with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(lines))

print(f"Wrote {OUTPUT_FILE}")
# %%
