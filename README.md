# GATE-CH: German Award Criteria Extraction from Swiss Tenders

## Installation

Create a virtual environment and run the installation script.

```
conda create -n gate_ch python=3.10
conda activate gate_ch
python install_packages.py
```

## Task 1: Award Criteria Presence Detection

To train and evaluate the binary classififier using the [SetFit](https://huggingface.co/docs/setfit/index) run the following code:

```
python train_set_fit_model.py
```

The results will be stored under a sub directory called `test_reports_setfit_hyperparameter_search_False__criteria_relevance`.

## Task 1: Award Criteria Extraction

To evaluate the LLM prompting using [VAGOsolutions/SauerkrautLM-Mixtral-8x7B-Instruct](https://huggingface.co/VAGOsolutions/SauerkrautLM-Mixtral-8x7B-Instruct), run the following code:

```
python evaluating_llm_promping.py -mnp VAGOsolutions/SauerkrautLM-Mixtral-8x7B-Instruct
```

The results will be saved under a new folder called `llm_prompting_evaluation_results`.


## Tables
After running the experiments, you can generate the tables with the metrics (Precision, Recall, F1) using the note book `Evaluation_Tables.ipynb`.