# Knowledge Graph-Based Hypothesis Generation and Experimental Validation in Yeast
This repository contains the code and data from the paper *Knowledge Graph-Based Hypothesis Generation and Experimental Validation in Yeast*. It is organised as folows:

```
/prediction_models/ - code to train models and predict hypotheses
/gene_sets/ - list and categories of genes available for experiments
/ranked_genes/ - the hypotheses predictions
/od_analysis/ - od data from experiments and code to analyse this data
/hypothesis_tests/ - code and outputs of hypothesis tests
/rimbo/ - code and ontology fro the extension of RIMBO
/kg.pkl - knowledge graph
```

To run predictions download trained models and cross validation fold information from [here](https://doi.org/10.5281/zenodo.23664660). Fold information is also needed to rerun training using the same folds. These files should be pu in the `prediction_models/` directory.

Python (tested with Python 3.10.15) requirements can be found in `requirements.txt` and can be installed by running:
```
pip install -r requirements.txt
```

# How to cite
...