# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import datetime
import string
import statsmodels.api as sm
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation
from sklearn.metrics import r2_score
from skmisc.loess import loess

# %%
FONT_SIZE = 16
LABEL_SIZE = 14

# %%
class PhenomicsProcessing:
    def __init__(self):
        self.raw_data = pd.DataFrame()
        self.layout = pd.DataFrame()
        self.blanked_data = pd.DataFrame()

    def load_growth_data(self, dir, batch):
        """
        Reads growth data from .DAT files in the specified directory and returns a DataFrame with the extracted information.
        """
        
        # Reads files in target directory and sorts them accordingly
        files = [f for f in os.listdir(dir) if f.endswith(".DAT")]
        files.sort()
        results = []
    
        t0 = None

        # Iterates over files
        for file in files:
            file_path = os.path.join(dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    data_section = False

                    for line in f:
                        # Extracts time
                        if line.startswith("Date"):
                            tnow = line.split(" ")[1] + " " + line.split(" ")[4].strip()
                            tnow = datetime.datetime.strptime(tnow, "%d/%m/%Y %H:%M:%S")

                            # Calculates time since start of experiment
                            if t0 is None:
                                t0 = tnow
                            delta_t = (tnow - t0).total_seconds() / 3600

                        # Extracts loop and plate number. Plate number is Eve's plate number + 4 * batch number
                        if line.startswith("ID1"):
                            ID1 = line.split(" ")[1].split("_")
                            lnow = ID1[0]
                            pnow = int(ID1[1][-1]) + (batch-1) * 4

                        # Starts reading for OD measurements. The line that starts with Cycle: is always above the measurements
                        if line.startswith("Cycle:"):
                            data_section = True
                            continue

                        # Extracts OD measurements
                        if data_section and line != "\n":
                            measurement = line.split(":")
                            well = measurement[0]
                            od_value = measurement[1].strip()
                            try:
                                od_value = float(od_value)
                            except ValueError:
                                od_value = np.nan
    
                            results.append([tnow, delta_t, lnow, pnow, well, od_value])
                    
            except Exception as e:
                print(f"Error reading the file {file}: {e}")
    
        # Compiles all data into a data frame
        df = pd.DataFrame(results, columns=["Date and Time", "deltaT", "Loop", "Plate", "Well", "OD"])
        df.sort_values(by = ["Date and Time", "Well"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Concatenates each batch file
        self.raw_data = pd.concat([
            self.raw_data if not self.raw_data.empty else None, 
            df], ignore_index=True)

        print(f"Loaded file {dir} with batch {batch}")

    def load_layout(self, path):
        """ 
        Loads design layouts from a .csv file. 
        """
        self.layout = pd.read_table(path, sep=",")
        self.layout["experiment"] = self.layout["orf"] + " : " + self.layout["chemical"]
        print(f"Loaded layout at {path}")

    def grouper(self, platenum, input, group):
        """
        This function takes a DataFrame and a group identifier (A-H) as input and returns a filtered DataFrame containing only the wells corresponding to the specified group.
        """
        rows = ["A", "B", "C", "D", "E", "F", "G", "H"]
        cols = ["03", "04", "05", "06", "07", "08", "09", "10"]

        wells = {}

        for i in string.ascii_uppercase[0:8]:
            wells[i] = ["".join(x) for x in zip(rows, cols)]
            cols = np.roll(cols, 1)

        subset = input[input["Plate"] == platenum]
        out = subset[subset["Well"].isin(wells[group])]

        return out

    def blank_outliers(self, inpt, cv_threshold = 0.2):
        """
        Removes blank outliers based on mad threshold.
        """

        out = inpt.copy()

        print(f"Removing blank wells with instrument errors (CV > {cv_threshold})...")
        for plate in inpt["Plate"].unique():
            subset = inpt[inpt["Plate"] == plate]
            blanks = subset[subset["Well"].str.contains(r".02|.11")]

            for well in blanks["Well"].unique():
                well_m = blanks.loc[blanks["Well"] == well, "OD"].mean()
                well_s = blanks.loc[blanks["Well"] == well, "OD"].std()
                well_cv = well_s/well_m

                if well_cv > cv_threshold:
                    out = out.drop(out[(out.Plate == plate) & (out.Well == well)].index)
                    print(f"    Removed Plate {plate} - Well {well}. CV = {well_cv:.2f} > {cv_threshold}")
        return out

                


    def bg_blank(self, save = False):
        """
        Blanks the data using the empty wells (modified Verduyn media, not inoc.) in the plate.
        """
    
        self.blanked_data = self.raw_data.copy()
        
        plates = self.blanked_data["Plate"].unique()
        for plate in plates:
            subset = self.raw_data.loc[self.raw_data["Plate"] == plate]
            blanks = subset[subset["Well"].str.contains(r".02|.11")]
            avg_blank = np.median(blanks["OD"])
            self.blanked_data.loc[self.blanked_data["Plate"] == plate, "OD"] -= avg_blank
    
            if save:
                fig, ax = plt.subplots()
                for well in blanks.loc[blanks["Plate"]==plate, "Well"].unique():
                        ax.scatter(blanks.loc[(blanks["Plate"]==plate) & (blanks["Well"] == well), "deltaT"], 
                                   blanks.loc[(blanks["Plate"]==plate) & (blanks["Well"] == well), "OD"], label=well)
                        ax.legend(loc="upper right")
                        ax.set_title(f"Plate {plate}")
                fig.savefig(f"{save}/Plate_{plate}_blanks.svg")

        print("Finished blanking using background blanks.")

    def min_blank(self, save = False):
        """
        Blanks the data using the minimum value in the growth curve.
        """

        self.blanked_data = self.raw_data.copy()

        plates = self.blanked_data["Plate"].unique()
        for plate in plates:
            for well in self.blanked_data.loc[self.blanked_data["Plate"] == plate, "Well"].unique():
                blank = self.blanked_data.loc[(self.blanked_data["Plate"] == plate) & (self.blanked_data["Well"] == well), "OD"].min()

                self.blanked_data.loc[(self.blanked_data["Plate"] == plate) & (self.blanked_data["Well"] == well), "OD"] -= blank

        print("Finished blanking using minimum setting.")

    def add_experiment_label(self, input):

        for plate in input["Plate"].unique():
            for group in string.ascii_uppercase[0:8]:
                tmp = self.grouper(plate, input, group)
                experiment = self.layout.loc[(self.layout["plate"]==plate) & (self.layout["group"] == group), "experiment"].item()

                input.loc[(input["Plate"] == plate) & (input["Well"].isin(tmp["Well"].unique())), "experiment"] = experiment

        print(f"Added experiment labels to dataframe")

    def remove_tagged_experiments(self, inpt):
        to_be_removed = self.layout.loc[self.layout["removed"].notna(), ["plate", "experiment"]].drop_duplicates()
        to_be_kept = self.layout.loc[self.layout["removed"].isna(), ["plate", "experiment"]].drop_duplicates()

        omitted = inpt.merge(
            to_be_removed,
            left_on = ["Plate", "experiment"],
            right_on=["plate", "experiment"],
            how="inner"
        )

        kept = inpt.merge(
            to_be_kept,
            left_on = ["Plate", "experiment"],
            right_on=["plate", "experiment"],
            how="inner"
        )

        print(f"Removed tagged wells from experiment")

        return kept, omitted

    def get_auc(self, input, save=False):
        auc_results = []

        for plate in input["Plate"].unique():
            for group in string.ascii_uppercase[0:8]:
                tmp = self.grouper(plate, input, group)
                repl = 1
                for j in tmp["Well"].unique():
                    tmp2 = tmp[tmp["Well"] == j]
                    # auc = np.trapezoid(tmp2["OD"], tmp2["deltaT"])
                    auc = np.trapz(tmp2["OD"], tmp2["deltaT"])
                    
                    lbl = self.layout[(self.layout["plate"] == plate) & (self.layout["group"] == group)]
                    auc_results.append([f"Plate{plate}_{j}", plate, j, lbl["orf"].item(), lbl["chemical"].item(), repl, auc])

                    repl += 1


        self.aucs = pd.DataFrame(auc_results, columns=["id", "plate", "Well", "orf", "chemical", "replicate", "auc"])
        self.aucs = self.aucs.drop_duplicates(subset=["orf", "chemical", "replicate"])

        print("Calculated AUC")

    def log_transform_aucs(self):
        self.log_aucs = self.aucs.copy()
        self.log_aucs["log_auc"] = np.log(self.log_aucs["auc"])
        print("Log transformed AUCs")

    def get_growth_params_fitting(self, input, r2_cutoff = 0.95):
        """ 
        Gets the growth parameters from fitting to a logistic function.
        """
        params = []

        for plate in input["Plate"].unique():
            for group in string.ascii_uppercase[0:8]:
                tmp = self.grouper(plate, input, group)
                repl = 1
                for j in tmp["Well"].unique():
                    tmp2 = tmp[tmp["Well"] == j]
                    lbl = self.layout[(self.layout["plate"] == plate) & (self.layout["group"] == group)]

                    try:
                        popt, pcov = curve_fit(self.logistic_growth_model, tmp2["deltaT"], tmp2["OD"], p0 = [1, 0.4, 0.025], bounds= ((0, 0, 0), (3, 1, 1)))  

                        y_pred = self.logistic_growth_model(tmp["deltaT"], *popt)
                        r2 = r2_score(tmp["OD"], y_pred)

                        if r2 < r2_cutoff:
                            print(f"{lbl['orf'].item()} - {lbl['chemical'].item()} {repl} - R2 score too low. R2 = {r2}")
                        elif popt[0] >= 10:
                            print(f"{lbl['orf'].item()} - {lbl['chemical'].item()} {repl} - Carrying capacity too high = {popt[0]}")
                        else:
                            params.append([f"Plate{plate}_{j}", lbl["orf"].item(), lbl["chemical"].item(), repl, *popt])
                            

                    except RuntimeError:
                        print(f"{lbl['orf'].item()} - {lbl['chemical'].item()} {repl} - Skipping due to optimal solution not found")

                    repl +=1

        self.growth_params = pd.DataFrame(params, columns=["id", "orf", "chemical", "replicate", "K", "mu", "x0"])

        print("Obtained growth parameters by fitting to a logistic function")

    def logistic_growth_model(self, t, K, mu, x0):
        return (K/(1 + ((K-x0)/x0) * np.e**(-mu*t)))


    def smooth_ln_curves(self, input, frac=0.3):
        """
        
        """
        groups = []
        for plate in input["Plate"].unique():

            for group in string.ascii_uppercase[0:8]:
                tmp = self.grouper(plate, input, group)
                repl = 1
                for j in tmp["Well"].unique():
                    tmp2 = tmp[tmp["Well"] == j]
                    lbl = self.layout[(self.layout["plate"] == plate) & (self.layout["group"] == group)]

                    tmp2["OD"] = np.log(tmp2["OD"])
                    loesstmp = sm.nonparametric.lowess(tmp2["OD"], tmp2["deltaT"], frac=frac)
                    experiment = self.layout.loc[(self.layout["plate"]==plate) & (self.layout["group"] == group), "experiment"].item()
                    tandOD = np.unique(loesstmp, axis=0)
                    groups.append([f"Plate{plate}_{j}", lbl["orf"].item(), lbl["chemical"].item(), repl, experiment, tandOD[:,0], tandOD[:,1]])
                    repl += 1

        self.smoothed_curves = pd.DataFrame(groups, columns=["id","orf","chemical", "replicate", "experiment","time", "ln_od"])
        print(f"Fitted ln(OD) using LOESS regression, frac = {frac}")
        

    def growth_params_empirical(self, inpt):
        """
        Obtains growth parameters K (carrying capacity/max OD) and max_mu (max biomass specific growth rate/max slope of ln(OD))
        """
        params = []

        for exp in inpt["experiment"].unique():
            experiment_repl = inpt.loc[inpt["experiment"] == exp, "replicate"]
            for rep in experiment_repl:

                mask = (inpt["experiment"] == exp) & (inpt["replicate"] == rep)

                orf = inpt.loc[mask, "orf"].iloc[0]
                chem = inpt.loc[mask, "chemical"].iloc[0]

                tim = inpt.loc[mask, "time"].iloc[0]
                ods = inpt.loc[mask, "ln_od"].iloc[0]

                # Make sure they're NumPy arrays
                tim = np.asarray(tim)
                ods = np.asarray(ods)

                dod_dt = np.gradient(ods, tim)

                params.append([exp, orf, chem, rep, max(np.exp(ods)), max(dod_dt)])

        self.growth_params = pd.DataFrame(params, columns = ["experiment", "orf", "chemical", "replicate", "K", "mu"])
        print("Obtained growth parameters empirically")

    def remove_outliers(self, inpt, param, threshold = 3):
        """
        Removes experiments based on the deviation from median AUC and maxOD
        """

        out = inpt.copy()
        out["experiment"] = out["orf"] + " : " + out["chemical"]

        print(f"Removing outliers that are outside {threshold} MADs from median.")
        for exp in out["experiment"].unique():
            subset = out[out["experiment"] == exp]
            med = np.median(subset[param])
            mad = np.median(np.abs(subset[param] - med))

            for replicate in subset["replicate"]:
                x = subset.loc[subset["replicate"] == replicate, param]

                if np.abs((x.item() - med)/mad) > threshold:
                    out = out.drop(out[(out.experiment == exp) & (out.replicate == replicate)].index)
                    print(f"    Removed {exp} - {replicate}")

        return out

        

    def duoplot(self, inpt1, inpt2, ids, savefig = False):
        """
        Plots boxplots of growth parameters to the left and a growth curve to the right. Always compares the given orf and chemcial against reference strain and control media.
        """

        # Initializes the plot and replaces the right-side subplots with a big subplot for the growth curves
        plt.close()
        fig = plt.figure(figsize=(9, 12))


        # Create a 2x2 grid with custom column and row ratios
        gs = gridspec.GridSpec(
            2, 2, width_ratios=[1, 1], height_ratios=[2, 1]  # Column 2 is twice as wide
        )  # Row 1 is 4x taller than Row 2

        ax_curve = fig.add_subplot(gs[0, :])
        ax_OD = fig.add_subplot(gs[1, 0])
        ax_mu = fig.add_subplot(gs[1, 1])

        # Sets the colors for the plots
        colors = ["#54B447", "#15B5EA", "#EBAE24",  "#F06E67"]
        mycolors = sns.color_palette(colors)
        mystyles = ["dashed", "dashdot", "solid", "dotted"]

        # Sets orf (deletant) and chemical (stressor) the user wants to plot
        orf = ids[0]
        chemical = ids[1]

        # Filters for rows that has reference : control, reference : chemical, orf : control, and orf : chemical.
        subset1 = inpt1.loc[((inpt1["orf"] == "Ref") | (inpt1["orf"] == orf)) & ((inpt1["chemical"] == "Control") | (inpt1["chemical"] == chemical))]
        subset1["experiment"] = subset1["orf"] + " : " + subset1["chemical"]
        order = [f"Ref : Control", f"Ref : {chemical}", f"{orf} : Control", f"{orf} : {chemical}"]

        shortened_chems = {
            "Hydrogen peroxide": r"$\text{H}_2\text{O}_2$",
            "Tunicamycin": r"Tunic.",
            "Caffeine": r"Caff.",
            "Ethanol": r"EtOH",
            "NaCl": r"NaCl",
            "Acetic acid": r"HAc"
        }

        # Orders the subsetted df so that it appears correctly in the box plots.
        subset1.experiment = pd.Categorical(subset1.experiment, categories = order)
        subset1 = subset1.sort_values("experiment")

        # Box plot of carrying capacity K
        sns.boxplot(x = subset1["experiment"], y = subset1["K"],  hue = subset1["experiment"], palette=mycolors,  ax = ax_OD)
        ax_OD.set_xlabel("")
        ax_OD.set_ylabel(r"Maximum $\text{OD}_{600}$", fontsize=FONT_SIZE+5)
        ax_OD.set_xticklabels([
            f"Ref. \n Control",
            f"Ref. \n" + shortened_chems[chemical],
            f"Mutant \n Control",
            f"Mutant \n" + shortened_chems[chemical]
            ])


        # Box plot of maximum logistic growth curve
        sns.boxplot(x = subset1["experiment"], y = subset1["mu"],  hue = subset1["experiment"], palette=mycolors,  ax = ax_mu)        
        ax_mu.set_xlabel("")
        ax_mu.set_ylabel(r"Maximum" + "\n" + r"growth rate $\mu_{max}$ ($h^{-1}$)", fontsize=FONT_SIZE+5)
        ax_mu.set_xticklabels([
            f"Ref. \n Control",
            f"Ref. \n" + shortened_chems[chemical],
            f"Mutant \n Control",
            f"Mutant \n" + shortened_chems[chemical]
            ])

        ax_OD.tick_params(axis="x", labelsize=LABEL_SIZE)
        ax_mu.tick_params(axis="x", labelsize=LABEL_SIZE)

        ax_OD.tick_params(axis="y", labelsize=LABEL_SIZE+2)
        ax_mu.tick_params(axis="y", labelsize=LABEL_SIZE+2)

        # Growth curves. 
        for i, exp in enumerate(order):

            data = inpt2[inpt2["experiment"] == exp]

            # Median and IQR
            summary = data.groupby("deltaT")["OD"].agg(median = "median", Q1=lambda x: x.quantile(0.25), Q3=lambda x: x.quantile(0.75)).reset_index()

            ax_curve.plot(summary["deltaT"], summary["median"], label=str(exp), color = mycolors[i], linestyle = mystyles[i])
            ax_curve.fill_between(summary["deltaT"], summary["Q1"] , summary["Q3"], alpha = 0.25, color = mycolors[i])
            ax_curve.legend(loc = 'upper left', frameon=False, prop={'size': 19})

            ax_curve.set_xlabel("Time (h)", fontsize=FONT_SIZE+5)
            ax_curve.set_ylabel(r"$\text{OD}_{600}$", fontsize=FONT_SIZE+5)
            ax_curve.set_xlim(0,24.5)
            # ax_curve.set_ylim(bottom = -0.05)
            ax_curve.tick_params(axis = "both", labelsize = LABEL_SIZE+2)

        plt.tight_layout()
        plt.show()

        if savefig:
            fig.savefig(f"../paper/figs/{savefig}", bbox_inches='tight')

    def plot_boxplots(self, inpt, chemical, color = "#15B5EA", savefig = False):

        fig, ax = plt.subplots(figsize = (10,3))
        
        # Sets the colors for the plots
        colors = ["#54B447", color]
        mycolors = sns.color_palette(colors)

        orfs = inpt.loc[inpt["chemical"] == chemical, "orf"].unique()
        orfs = np.delete(orfs, orfs == "Ref")

        order = [f"Ref : Control", f"Ref : {chemical}"]
        labels = ["Ref."]

        for orf in orfs:
            order.extend([f"{orf} : Control", f"{orf} : {chemical}"])
            lbl = orf.lower() + r"$\Delta$"
            labels.append(lbl)

            
        subset = inpt.loc[((inpt["orf"] == "Ref") | (inpt["orf"].isin(orfs))) & ((inpt["chemical"] == "Control") | (inpt["chemical"] == chemical))]
        subset["experiment"] = subset["orf"] + " : " + subset["chemical"]

        # Orders the subsetted df so that it appears correctly in the box plots.
        subset.experiment = pd.Categorical(subset.experiment, categories = order)
        subset = subset.sort_values("experiment")
        subset = subset.rename(columns={"chemical": "Chemical"})

        # Box plot
        sns.boxplot(x = subset["orf"], y = subset["auc"],  hue = subset["Chemical"], palette=mycolors)
        ax.set_xlabel("")
        ax.set_ylabel("Area Under Curve", fontsize=FONT_SIZE)

        # Tick labels
        ax.tick_params(axis="both", labelsize=LABEL_SIZE)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylim(bottom=0)

        ref = True
        for tick in ax.get_xticklabels():
            if ref:
                    ref = False
            else:
                tick.set_fontstyle("italic")

        # Legend
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(FONT_SIZE)

            if legend.get_title() is not None:
                legend.get_title().set_fontsize(FONT_SIZE)
        fig.show() 

        if savefig:
            fig.savefig(f"../paper/figs/{savefig}", bbox_inches='tight')

    def save_file(self, inpt, file, path = "./out/"):
        inpt.to_csv(f"{path}{file}", index=False)



# %%

def main():
    results = PhenomicsProcessing()
    results.load_growth_data("omega_polestar/20260909_DEL1", batch = 1)
    results.load_growth_data("omega_polestar/20260910_DEL2", batch = 2)
    results.load_growth_data("omega_polestar/20260917_DEL3", batch = 3)
    results.load_growth_data("omega_polestar/20260919_DEL4", batch = 4)
    results.load_growth_data("omega_polestar/20260926_DEL5", batch = 5)
    results.load_layout("./plate_layout.csv")

    results.add_experiment_label(results.raw_data)

    # Removes outlier wells from the raw data
    # results.raw_data = results.raw_data.drop(results.raw_data[(results.raw_data.Plate == 6) & (results.raw_data.Well == "H02")].index)

    results.raw_data = results.blank_outliers(results.raw_data)
    results.bg_blank()
    results.blanked_data, results.ommited_data = results.remove_tagged_experiments(results.blanked_data)

    # outliers Acetic acid and NaCl ref
    # results.blanked_data = results.blanked_data.drop(results.blanked_data[(results.blanked_data.Plate == 6) & (results.blanked_data.Well == "H04")].index)
    # results.blanked_data = results.blanked_data.drop(results.blanked_data[(results.blanked_data.Plate == 6) & (results.blanked_data.Well == "H03")].index)

    # # outlier YGR036C - caffeine
    # results.blanked_data = results.blanked_data.drop(results.blanked_data[(results.blanked_data.Plate == 2) & (results.blanked_data.Well == "H05")].index)

    # # outlier YKL037W - caffeine
    # results.blanked_data = results.blanked_data.drop(results.blanked_data[(results.blanked_data.Plate == 4) & (results.blanked_data.Well == "B09")].index)

    # # outlier YOR070C - acetic acid (C04, E06)
    # results.blanked_data = results.blanked_data.drop(results.blanked_data[(results.blanked_data.Plate == 7) & (results.blanked_data.Well == "C04")].index)
    # results.blanked_data = results.blanked_data.drop(results.blanked_data[(results.blanked_data.Plate == 7) & (results.blanked_data.Well == "E06")].index)

    # low growthparam 
    # results.blanked_data = results.blanked_data.drop(results.blanked_data[(results.blanked_data.Plate == 6) & (results.blanked_data.Well == "A08")].index)

    # Caluclate AUC and ln(AUC)
    results.get_auc(results.blanked_data)
    results.log_transform_aucs()

    # Smooths the ln(OD) curves and calculates growth parameters empirically
    results.smooth_ln_curves(results.blanked_data, frac=0.3)
    results.growth_params_empirical(results.smoothed_curves)
    results.log_aucs = results.remove_outliers(results.log_aucs, "log_auc", threshold = 10)

    # Save boxplots
    results.plot_boxplots(results.aucs, "Hydrogen peroxide", color = "#15B5EA", savefig = "hydrogen_peroxide.pdf")
    results.plot_boxplots(results.aucs, "Caffeine", color = "#D86CAA", savefig = "caffeine.pdf")
    results.plot_boxplots(results.aucs, "Tunicamycin", color = "#F06E67", savefig = "tunicamycin.pdf")
    results.plot_boxplots(results.aucs, "Ethanol", color = "#A880B9", savefig = "ethanol.pdf")
    results.plot_boxplots(results.aucs, "Acetic acid", color = "#EBAE24", savefig = "acetic_acid.pdf")
    results.plot_boxplots(results.aucs, "NaCl", color = "#2CB891", savefig = "nacl.pdf")

    # Save ln(AUC) and growth parameters to CSV files
    results.save_file(results.log_aucs, "20261021_log_aucs.csv")
    results.save_file(results.growth_params, "20261021_growth_params.csv")

    # Generate duoplots for each orf and chemical pair
    # strains = ["YER083C", "YER083C", "YOR070C", "YGR036C", "YJL204C", "YJL183W", "YOR070C", "YOR322C", "YMR015C", "YNL133C"]
    # chems = ["Hydrogen peroxide", "Acetic acid", "Hydrogen peroxide", "Caffeine", "NaCl", "Caffeine", "Acetic acid", "Hydrogen peroxide", "Hydrogen peroxide", "Tunicamycin"]

    # Generate duoplots for orfs and chems mentioned in the paper
    strains = ["YER083C", "YGR036C"]
    chems = ["Hydrogen peroxide", "Caffeine"]

    for i in zip(strains, chems):
        results.duoplot(results.growth_params, results.blanked_data, i, savefig = f"{i[0]}_{i[1]}_duoplot.pdf")


# %% 
if __name__ == "__main__":
    main()


# %%
