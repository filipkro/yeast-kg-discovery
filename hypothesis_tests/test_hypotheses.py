# %%
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import os
import numpy as np
from scipy.stats import t
# ==========================================================
# Configuration
# ==========================================================
# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE)
# %%

CSV_FILE = "20261021_log_aucs.csv"
RESPONSE = "auc"
RESPONSE = "log_auc"

WT_NAME = "Ref"
CONTROL = "Control"      # e.g. "Control", "Untreated", "NoChemical" - needs to match the CSV file

ALPHA = 0.05

# ==========================================================
# Read data
# ==========================================================
file_path = os.path.join(BASE, "od_analysis/out", CSV_FILE)
df = pd.read_csv(file_path, keep_default_na=False)

df["orf"] = df["orf"].astype(str)
df["chemical"] = df["chemical"].astype(str)

results = []
resids = []

orfs = sorted(g for g in df["orf"].unique() if g != WT_NAME)
chemicals = sorted(c for c in df["chemical"].unique() if c != CONTROL)

# ==========================================================
# Analyse every orf × chemical pair
# ==========================================================

for orf in orfs:
    for chem in chemicals:

        subset = df[
            df["orf"].isin([WT_NAME, orf]) &
            df["chemical"].isin([CONTROL, chem])
        ].copy()

        # Require all four experimental groups
        groups = subset.groupby(["orf", "chemical"]).size()

        if len(groups) != 4:
            print(f"Skipping {orf} / {chem} (missing group)")
            continue

        # Make WT and Control the reference categories
        subset["orf"] = pd.Categorical(
            subset["orf"],
            categories=[WT_NAME, orf]
        )

        subset["chemical"] = pd.Categorical(
            subset["chemical"],
            categories=[CONTROL, chem]
        )


        # --------------------------------------------------
        # Linear model
        # --------------------------------------------------

        model = smf.ols(
            f"{RESPONSE} ~ orf * chemical",
            data=subset
        ).fit(cov_type="HC3", use_t=True)

        comparison = pd.DataFrame({
            # "OLS_SE": naive_model.bse,
            "HC3_SE": model.bse,
            # "OLS_p": naive_model.pvalues,
            "HC3_p": model.pvalues
        })
        print(comparison)


        interaction = f"orf[T.{orf}]:chemical[T.{chem}]"

        tmp = pd.DataFrame({
            "orf": subset["orf"],
            "chem": subset["chemical"],
            "experiment": f"{orf} : {chem}",
            "residuals": model.resid,
            "fitted_vals": model.fittedvalues
        })
        resids.append(tmp)

        print(model.params)
        beta0 = model.params["Intercept"]
        beta1 = model.params[f"orf[T.{orf}]"]
        beta2 = model.params[f"chemical[T.{chem}]"]
        beta = model.params[interaction]
        tstat = model.tvalues[interaction]
        se = model.bse[interaction]
        upper_bound = beta + t.ppf(0.95, model.df_resid) * se
        # --------------------------------------------------
        # One-sided p-value
        #
        # H0 : beta = 0
        # HA : beta < 0
        # --------------------------------------------------

        p_two = model.pvalues[interaction]

        if beta < 0:
            p_one = p_two / 2
        else:
            p_one = 1 - p_two / 2

        # --------------------------------------------------
        # Descriptive statistics
        # --------------------------------------------------

        wt_control = subset[
            (subset.orf == WT_NAME) &
            (subset.chemical == CONTROL)
        ][f"{RESPONSE}"].mean()

        wt_stress = subset[
            (subset.orf == WT_NAME) &
            (subset.chemical == chem)
        ][f"{RESPONSE}"].mean()

        mut_control = subset[
            (subset.orf == orf) &
            (subset.chemical == CONTROL)
        ][f"{RESPONSE}"].mean()

        mut_stress = subset[
            (subset.orf == orf) &
            (subset.chemical == chem)
        ][f"{RESPONSE}"].mean()

        wt_reduction = wt_stress - wt_control
        mut_reduction = mut_stress - mut_control
        extra_reduction = mut_reduction - wt_reduction

        results.append({

            "orf": orf,
            "chemical": chem,

            "WT_control": wt_control,
            "WT_stress": wt_stress,

            "Mutant_control": mut_control,
            "Mutant_stress": mut_stress,

            "WT_reduction": wt_reduction,
            "Mutant_reduction": mut_reduction,
            "Extra_reduction": extra_reduction,

            "beta0_intercept": beta0,
            "beta1_mutant": beta1,
            "beta2_chemical": beta2,

            "interaction": beta,
            "std_error": se,
            "t": tstat,
            "p": p_one,
            "upper_bound": upper_bound

        })

# ==========================================================
# Results table
# ==========================================================

results = pd.DataFrame(results)

if len(results) == 0:
    raise RuntimeError("No valid hypothesis tests could be performed.")

reject, fdr, _, _ = multipletests(
    results["p"],
    alpha=ALPHA,
    method="fdr_bh"
)

results["FDR"] = fdr
results["significant"] = reject


# Relative stress response:
# How much of the WT's normalised growth under stress does the mutant retain?
# (Mutant_stress / Mutant_control) / (WT_stress / WT_control)
#
# Relative reduction:
# How large is the mutant's stress-induced reduction compared with the WT's reduction?
# abs(Mutant_reduction) / abs(WT_reduction)
results["relative_reduction"] = (results["Mutant_reduction"] /
                                   results["WT_reduction"].abs())

results["fitness_reduction"] = 1 - np.exp(results['interaction']) 

results["relative_stress_response"] = (
    (results["Mutant_stress"] / results["Mutant_control"]) /
    (results["WT_stress"] / results["WT_control"])
)
results["log2_relative_stress_response"] = np.log2(
    results["relative_stress_response"]
)

results = results.sort_values("p")

results = results[
    [
        "orf",
        "chemical",

        "WT_control",
        "WT_stress",
        "Mutant_control",
        "Mutant_stress",
        "relative_stress_response",
        "log2_relative_stress_response",
        "relative_reduction",

        "WT_reduction",
        "Mutant_reduction",
        "Extra_reduction",

        "beta0_intercept",
        "beta1_mutant",
        "beta2_chemical",

        "interaction",
        "fitness_reduction",
        "std_error",
        "t",
        "p",
        "FDR",
        "upper_bound",
        "significant"
    ]
]

results.to_csv("hypothesis_tests.csv", index=False)

resids = pd.concat(resids, ignore_index=True)
resids.to_csv("hypotests_resids.csv", index = False)

# ==========================================================
# Display
# ==========================================================

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

print(results.round(4))

print("\nSaved results to hypothesis_tests.csv")
# %%
