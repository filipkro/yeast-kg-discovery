# %%
from pathlib import Path

from rdflib import Graph, URIRef #, Namespace, URIRef, Literal
# from rdflib.namespace import RDF, XSD
from rimbo_functions import KG
# import demo_functions


from pathlib import Path

import pandas as pd
from rdflib import Graph
import os
from rimbo_functions import (create_knowledge_graph, create_revision,
                             print_revision)
# %%

# =============================================================================
# Configuration
# =============================================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE)
# %%
HYPOTHESES_CSV = "hypothesis_tests.csv"
HYPOTHESES_CSV = os.path.join(BASE, "hypo_tests", HYPOTHESES_CSV)
ORIGINAL_KG_PATH = "<path/to/yeast_kg.ttl>"

OUTPUT_RIMBO_PATH = "yeast_kg_revisions.ttl"

FDR_THRESHOLD = 0.05

# =============================================================================
# Chemical mapping
# =============================================================================

# Chemical name -> ChEBI ID
#
# Replace/add entries with your actual dictionary.

# CHEMICAL_TO_CHEBI = {
#     # "Caffeine": "CHEBI:27732",
#     # "Hydrogen peroxide": "CHEBI:16240",
#     # ...
# }
CHEMICAL_TO_CHEBI = {'Hydrogen peroxide': 'CHEBI_16240',
'Tunicamycin': 'CHEBI_29699',
'Caffeine': 'CHEBI_27732',
'NaCl': 'CHEBI_26710',
'Ethanol': 'CHEBI_16236',
'Acetic acid': 'CHEBI_15366'}


# =============================================================================
# Main
# =============================================================================

# def main():
# -------------------------------------------------------------------------
# Read hypotheses
# -------------------------------------------------------------------------

df = pd.read_csv(HYPOTHESES_CSV)

required_columns = {
    "orf",
    "chemical",
    "p",
    "FDR",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        "CSV is missing required columns: "
        f"{sorted(missing_columns)}"
    )

# -------------------------------------------------------------------------
# Create RIMBO provenance graph
# -------------------------------------------------------------------------

rimbo = Graph()

original_kg, original_kg_file = create_knowledge_graph(
    graph=rimbo,
    kg_id="yeast_kg_v0",
    file_path=ORIGINAL_KG_PATH,
)

# -------------------------------------------------------------------------
# Create one revision for every hypothesis
# -------------------------------------------------------------------------

for i, row in df.iterrows():

    orf = str(row["orf"]).strip()
    chemical = str(row["chemical"]).strip()

    # ---------------------------------------------------------------------
    # Resolve chemical
    # ---------------------------------------------------------------------

    if chemical not in CHEMICAL_TO_CHEBI:
        raise ValueError(
            f"No ChEBI ID found for chemical '{chemical}' "
            f"(CSV row {i})."
        )

    chebi_id = CHEMICAL_TO_CHEBI[chemical]

    # ---------------------------------------------------------------------
    # Statistical evidence
    # ---------------------------------------------------------------------

    p_value = float(row["p"])
    fdr = float(row["FDR"])

    # FDR determines whether the hypothesis is experimentally supported.
    supports_hypothesis = fdr < FDR_THRESHOLD

    # ---------------------------------------------------------------------
    # Deterministic revision ID
    # ---------------------------------------------------------------------

    revision_id = (f"revision_{orf}_{chebi_id}")

    # ---------------------------------------------------------------------
    # Create RIMBO revision
    # ---------------------------------------------------------------------

    create_revision(
        graph=rimbo,
        revision_id=revision_id,
        original_kg=original_kg,
        original_kg_representation=original_kg_file,
        orf_name=orf,
        chemical_id=chebi_id,
        p_value=p_value,
        fdr=fdr,
        supports_hypothesis=supports_hypothesis,
    )

# -------------------------------------------------------------------------
# Save provenance graph
# -------------------------------------------------------------------------

rimbo.serialize(destination=OUTPUT_RIMBO_PATH, format="turtle")

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------

n_supported = sum(float(fdr) < FDR_THRESHOLD for fdr in df["FDR"])

n_unsupported = len(df) - n_supported

print(f"Processed {len(df)} hypotheses.")

print(f"Supported:   {n_supported}")

print(f"Unsupported: {n_unsupported}")

print(f"FDR threshold: {FDR_THRESHOLD}")

print(f"RIMBO graph contains {len(rimbo)} triples.")

print(f"Saved RIMBO graph to: {OUTPUT_RIMBO_PATH}")


# if __name__ == "__main__":
#     main()


# %%
# supported
print_revision(graph=rimbo, revision=KG["revision_YKL037W_CHEBI_27732"])
# %%
# unsupported
print_revision(graph=rimbo, revision=KG["revision_YIR023W_CHEBI_27732"])
# %%
