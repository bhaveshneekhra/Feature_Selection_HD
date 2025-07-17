# HD_Feature_Selection

This repository contains code and configuration to reproduce the experiments and results presented in the paper on feature selection using high-dimensional lung cancer gene expression datasets.

---

## 🧰 Requirements

- Python 3.7+
- `virtualenv` or `venv` for isolated environments

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone <INSERT git address>
cd HD_Feature_Selection
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv fs_env
source fs_env/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
📂 Dataset Preparation

Directory structure

        All datasets should be placed inside the datasets/ folder located at the root of the repository:

            HD_Feature_Selection/
            │
            ├── datasets/
            │   ├── Lung_GSE18842.csv
            │   └── Lung_GSE19804.csv

Download Datasets

        You can download the datasets manually or via wget:

        🔹 Option 1: Manual Download

            Visit the CuMiDa repository and search for:

                •	GSE18842
                •	GSE19804

            For TCGA LUAD/LUSC, use the GDC Data Portal. 

        🔹 Option 2: Command-line Download

            ```bash
            cd datasets/

            wget https://sbcb.inf.ufrgs.br/data/cumida/Genes/Lung/GSE18842/Lung_GSE18842.csv

            wget https://sbcb.inf.ufrgs.br/data/cumida/Genes/Lung/GSE19804/Lung_GSE19804.csv
            
            ```

⚙️ Configuration

To reproduce Figure 1 from the paper, ensure your configuration (config.yaml) contains the following:

    train_test_sep: 1
    dataset: "Lung_GSE18842"
    filepath: "./datasets/Lung_GSE18842.csv"
    test_dataset: "Lung_GSE19804"
    test_filepath: "./datasets/Lung_GSE19804.csv"
    target: 'type'
    model_name: "RF"
    opt_model: 0
    random_state: 42
    debug: 0
    num_runs: 20
    remove_cols: True

    plot_acc: 1
    plot_auc: 0

    interactive: 0
    shuffle_cols: 0

    feature_ticks_ranges:
    - [1, 51, 1]
    - [60, 201, 10]
    - [300, 2001, 100]

    annotate_paper_results: 0
    # paper: "Lall (2020)"
    # x_from_paper: 35
    # y_from_paper: .86

▶️ Run the Code

```bash
python3 runner_code.py
```

📊 Output

This will reproduce accuracy plots as in Figure 1 of the referenced paper. You can enable AUC plots or paper annotation in the config file if needed.

📎 License

This repository is distributed under the MIT License.

## 📬 Contact

For queries, please contact **Bhavesh Neekhra** at  
`FirstnameLastname [at] gmail [dot] com`