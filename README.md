# HD_Feature_Selection

Repository to reproduce experiments and figures from: “On the (In)Significance of Feature Selection in High-Dimensional Datasets”.

---

## 🧰 Requirements

- Python 3.7+
- `virtualenv` or `venv` for isolated environments
- Dependencies: see requirements.txt 

---

## 🔧 Setup Instructions

```bash
git clone <INSERT git address>
cd HD_Feature_Selection
python3 -m venv fs_env
source fs_env/bin/activate
pip install -r requirements.txt
```
### 📂 Datasets

- Place datasets in datasets/ folder located at the root of the repository:

        HD_Feature_Selection/
        │
        ├── datasets/
        │   ├── Lung_GSE18842.csv
        │   └── Lung_GSE19804.csv
- Provided datasets: golub_all_aml.csv.
- Additional datasets (GSE18842, GSE19804 and others) can be downloaded manually or automatically via the config.yaml file.

* Download Datasets

    You can download the datasets manually or using the code by specifying config parameteres in the script (described later):

    🔹 Option 1: Manual Download

        Visit the CuMiDa repository and search for:
        - GSE18842
        - GSE19804

        For TCGA LUAD/LUSC, use the GDC Data Portal. 

    🔹 Option 2: Command-line Download

    ```bash
    cd datasets/

    wget https://sbcb.inf.ufrgs.br/data/cumida/Genes/Lung/GSE18842/Lung_GSE18842.csv

    wget https://sbcb.inf.ufrgs.br/data/cumida/Genes/Lung/GSE19804/Lung_GSE19804.csv
    ```

    🔹 Option 3: Download using the script

    If you want the dataset to be downloaded autmotically, specify these two parameters in the config.yaml file
    ```bash
    download_datasets: 1  # If 1, then datasets will be downloaded automatically
                          # If 0, then datasets should be downloaded manually and placed in the datasets folder
    dataset_urls:
        - https://sbcb.inf.ufrgs.br/data/cumida/Genes/Leukemia/GSE28497/Leukemia_GSE28497.csv
        - # If you provide multiple links, the script will download them all if not already downloaded. 
    ```

### ⚙️ Configuration

To reproduce Figure 1 from the paper, please ensure 

1. You have GSE18842 and GSE19804 datasets inside datasets/ folder.

2. Your configuration (config.yaml) contains the following:

```bash
    train_test_sep: 1                                       # 0 = internal train-test split; 1 = external test set
    dataset: "Lung_GSE18842"                                # Identifier for the dataset used      
    filepath: "./datasets/Lung_GSE18842.csv"                # Path to the input data file   
    test_dataset: "Lung_GSE19804"                           # If train_test_sep is 1, then provide identifier for the test dataset
    test_filepath: "./datasets/Lung_GSE19804.csv"           # If train_test_sep is 1, then provide path to the test data file   
target: 'type'                                              # Column name for target labels
    
    
    model_name: "RF"                                        # Model used: RF = Random Forest
    opt_model: 0                                            # Use an optimized (1) /default (0) config for the model
    random_state: 42                                        # Seed for reproducibility
    debug: 0                                                # 1 = Print detailed debug info; 0 = minimal output
    num_runs: 20                                            # Number of repeated runs (for averaging)
    remove_cols: True                                       # Whether to remove previously selected features. If number of features are more than 20,000 - set it to true

    plot_acc: 1                                              # Plot accuracy vs. number of features
    plot_auc: 1                                              # Plot AUC vs. number of features

    interactive: 0                                           # 1 = Interactive code execution, asking for confirmationl 0: no confirmation required
    shuffle_cols: 0                                          # If 1, feature columns are shuffled before selection

    feature_ticks_ranges:                                    # Number of features to try in each experiment run
    - [1, 51, 1]
    - [60, 201, 10]
    - [300, 2001, 100]

    annotate_paper_results: 0                                # Whether to add marker for a published paper's result. If 1, then paper, x_from_paper and y_from_paper should be defined.
    # paper: "Lall (2020)"                                   # Citation or label for the paper
    # x_from_paper: 35                                       # Feature count used in the paper
    # y_from_paper: .86                                      # Reported accuracy in the paper
```

We have also provided a dataset golub_all_aml.csv with this repository. To run the code with this dataset and also annotate the accuracy plot, use the following config (already included, so NO action needed):

```bash
    train_test_sep: 0              # 0 = internal train-test split; 1 = external test set
    dataset: "ALL_AML"             # Identifier for the dataset used
    filepath: "./datasets/golub_all_aml.csv"  # Path to the input data file
    target: "type"                 # Column name for target labels

    model_name: "RF"               # Model used: RF = Random Forest
    opt_model: 1                   # Use an optimized/default config for the model
    random_state: 42               # Seed for reproducibility
    debug: 0                       # 1 = Print detailed debug info; 0 = minimal output
    num_runs: 20                   # Number of repeated runs (for averaging)
    remove_cols: False             # Whether to remove previously selected features

    plot_acc: 1                    # Plot accuracy vs. number of features
    plot_auc: 1                    # Plot AUC vs. number of features

    interactive: 0                 # 1 = Interactive code execution, asking for confirmationl 0: no confirmation required
    shuffle_cols: 0                # If 1, feature columns are shuffled before selection

    feature_ticks_ranges:          # Number of features to try in each experiment run
    - [1, 51, 1]
    - [60, 201, 10]
    - [300, 2001, 100]

    annotate_paper_results: 1      # Whether to add marker for a published paper's result. If 1, then paper, x_from_paper and y_from_paper should be defined.
    paper: "Lall (2020)"           # Citation or label for the paper
    x_from_paper: 35               # Feature count used in the paper
    y_from_paper: 0.86             # Reported accuracy or AUC in the paper

```

### ▶️ Run Experiments

```bash
python3 runner_code.py
```

### 📊 Output 

Outputs will be saved in Plots/[DATASET] and include:

    - Accuracy vs. feature count
    - AUC vs. feature count
    - CSV summary of results

📎 License

This repository is distributed under the MIT License.