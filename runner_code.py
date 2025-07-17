
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

import zipfile
import fs_utils

import time, os
import subprocess

dataset = fs_utils.config["dataset"]
download_datasets = fs_utils.config["download_datasets"]

model_name = fs_utils.config['model_name']
opt_model = fs_utils.config['opt_model']
random_state = fs_utils.config['random_state']
debug = fs_utils.config['debug']
num_runs = fs_utils.config['num_runs']

plot_acc = fs_utils.config['plot_acc']
plot_auc = fs_utils.config['plot_auc']

print(f'for dataset: {dataset}, the common setting are:')

print(f'Whether to download the datasets: {download_datasets}')
print(f'model to use for training: {model_name}')
print(f'Whether to optimise the model: {opt_model}')
print(f'random state: {random_state}')
print(f'Whether to print debug level information: {debug}')
print(f'For each randomly selected subset, how many runs: {num_runs}')
print(f'While plotting, whether to plot Accuracy: {plot_acc}')
print(f'While plotting, whether to plot AUC: {plot_auc}')

filepath = fs_utils.config['filepath']

if fs_utils.config.get("download_datasets", 0) == 1:
    urls = fs_utils.config.get("dataset_urls", [])
    download_dir = "datasets"
    os.makedirs(download_dir, exist_ok=True)

    for url in urls:
        filename = os.path.basename(url)
        filepath = os.path.join(download_dir, filename)

        if os.path.exists(filepath):
            print(f"[✓] Skipping download: {filename} already exists.")
        else:
            print(f"[↓] Downloading: {filename}")
            try:
                subprocess.run(["wget", "-O", filepath, url], check=True)
                print(f"[✔] Downloaded: {filename}")
            except subprocess.CalledProcessError:
                print(f"[✗] Failed to download: {url}")


if fs_utils.config['dataset'] == 'TCGA' or fs_utils.config['dataset'] == 'Data_Bischoff2021_Lung' or fs_utils.config['dataset'] == 'Lung Single-cell RNA-Seq':
    data_final = pd.read_parquet(filepath)
    
elif fs_utils.config['dataset'] == 'Arcene':
    data_final = pd.read_csv(filepath, header=None)
    data_final.rename(columns={10000: 'type'}, inplace=True)

elif fs_utils.config['dataset'] == 'Gisette':
    if fs_utils.config.get("download_datasets", 0) == 1:
        zip_path = "datasets/gisette.zip"
        extract_dir = "datasets/gisette"

        if not os.path.exists(extract_dir):
            # Create the directory if it doesn't exist
            os.makedirs(extract_dir, exist_ok=True)
        # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

        if fs_utils.config['debug']:
            print(f"Unzipped to: {extract_dir}")

    X_train = pd.read_csv(filepath, sep=r'\s+', engine='python', header=None)
    y_train = pd.read_csv('./datasets/gisette/GISETTE/gisette_train.labels', header=None)

    X_train.columns = [f'feat_{i}' for i in range(X_train.shape[1])]
    y_train.columns = ['type']
    data_final = pd.concat([X_train, y_train], axis=1)

elif fs_utils.config['dataset'] == 'Madelon':
    X_train = pd.read_csv(filepath, sep=r'\s+', engine='python', header=None)
    y_train = pd.read_csv('./dataset/NIPS_2003/MADELON/madelon_train.labels',header=None)

    X_train.columns = [f'feat_{i}' for i in range(X_train.shape[1])]
    y_train.columns = ['type']

    data_final = pd.concat([X_train, y_train], axis=1)
else:
    data_final = pd.read_csv(filepath)

if fs_utils.config["dataset"]=="Colon" or fs_utils.config["dataset"]=="Alon1999_Colon":
    try:
        data_final.drop('Unnamed: 0', axis=1, inplace=True)
        data_final.rename(columns={'NA': 'type'}, inplace=True)
    
    except:
        # print("Check the column names")
        pass

elif fs_utils.config["dataset"]=="Leukemia_GSE28497" or fs_utils.config["dataset"]=="Colorectal_GSE44076" or fs_utils.config["dataset"]=="Liver_GSE76427" \
    or fs_utils.config["dataset"]=="Breast_GSE45827" or fs_utils.config['dataset']=="Colorectal_GSE21510" or fs_utils.config['dataset']=="Renal_GSE53757"\
        or fs_utils.config['dataset']=="Breast_GSE22820" or fs_utils.config['dataset']=="Brain_GSE50161" or fs_utils.config['dataset']=='Prostate_GSE6919_U95B'\
            or fs_utils.config['dataset']=='Lung_GSE19804' or fs_utils.config['dataset']=='Throat_GSE42743' or fs_utils.config['dataset']=='Ovary_GSE6008' \
                or fs_utils.config['dataset']=='Lung_GSE18842':
    try:
        data_final.set_index('samples', inplace=True)
    except:
        # print("Check the column names")
        pass
elif fs_utils.config["dataset"]=="GSE3365" or fs_utils.config["dataset"]=="GSE4115":
    try:
        data_final.set_index('Unnamed: 0', inplace=True)
        data_final.rename(columns={'disease': 'type'}, inplace=True)
    except:
        # print("Check the column names")
        pass

elif fs_utils.config["dataset"]=="GSE30219_LUAD_LUSC":
    try:
        data_final.set_index('name', inplace=True)
    except:
        # print("Check the column names")
        pass
elif fs_utils.config["dataset"] == "Lung Single-cell RNA-Seq":
    data_final.rename(columns={'sample': 'type'}, inplace=True)

if data_final['type'].isnull().sum() > 0:
    data_final.dropna(subset=['type'], inplace=True)

print(f"{data_final['type'].isnull().sum()} rows with null type values removed.")
print(f"Dataset shape: {data_final.shape}")

if fs_utils.config['train_test_sep']:
    print("As there is a separate test dataset, we need to load it now...")
    
    if fs_utils.config['test_dataset'] == 'Arcene_test':
        # Arcene test dataset has no header
        # so we need to load it with header=None
        # and then rename the last column to 'type'
        # as it is the label column
        # and the first 10000 columns are features
        # and the last column is the label
        print("Loading Arcene test dataset...")
        test_df = pd.read_csv(fs_utils.config['test_filepath'], header=None)
        test_df.rename(columns={10000: 'type'}, inplace=True)
        
    elif fs_utils.config['test_dataset'] == 'Gisette_validation':
        print("Loading Gisette test dataset...")
        X_test = pd.read_csv(fs_utils.config['test_filepath'], sep=r'\s+', engine='python', header=None)
        y_test = pd.read_csv('datasets/gisette/gisette_valid.labels', header=None)

        X_test.columns = [f'feat_{i}' for i in range(X_test.shape[1])]
        y_test.columns = ['type']

        test_df = pd.concat([X_test, y_test], axis=1)

    elif fs_utils.config['test_dataset'] == 'Madelon_Validation':
        print("Loading Madelon test dataset...")

        X_test = pd.read_csv(fs_utils.config['test_filepath'], sep=r'\s+', engine='python', header=None)
        y_test = pd.read_csv('../Data/NIPS_2003/Madelon/madelon/madelon_valid.labels',header=None)

        X_test.columns = [f'feat_{i}' for i in range(X_test.shape[1])]
        y_test.columns = ['type']

        test_df = pd.concat([X_test, y_test], axis=1)

    else:
        test_df = pd.read_csv(fs_utils.config['test_filepath'])
    
    try:
        test_df.set_index('samples', inplace=True)
    except: 
        # print("Check the column names")
        pass

    y = test_df[fs_utils.config['target']]
    # encoding labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    test_df[fs_utils.config['target']] = y

    print(f'Test set has the shape: {test_df.shape}')

if fs_utils.config['train_test_sep']:
    print(f'Test set has the shape: {test_df.shape}')

print(f"\n\nDataset {fs_utils.config['dataset']} at a glance:")
print("-"*25)
print("Number of samples: "+ str(data_final.shape[0]))
print("Number of genes: "+str(data_final.shape[1]-1))
print("Number of classes: ", len(data_final[fs_utils.config['target']].value_counts()))
print("-"*25)
print("Class-wise distribution: ", (data_final[fs_utils.config['target']].value_counts()))
print("-"*25)
print(data_final.shape)
print("-"*25)


if fs_utils.config['train_test_sep']:
    print(f"\n\nTest dataset {fs_utils.config['test_dataset']} at a glance:")
    print("-"*25)
    print("Number of samples: "+ str(test_df.shape[0]))
    print("Number of genes: "+str(test_df.shape[1]-1))
    print("Number of classes: ", len(test_df[fs_utils.config['target']].value_counts()))
    print("-"*25)
    print("Class-wise distribution: ", (test_df[fs_utils.config['target']].value_counts()))
    print("-"*25)
    print(test_df.shape)
    print("-"*25)


col = data_final.pop(fs_utils.config['target'])
data_final[fs_utils.config['target']] = col

if fs_utils.config['train_test_sep']:
    test_col = test_df.pop(fs_utils.config['target'])
    test_df[fs_utils.config['target']] = test_col


y = data_final[fs_utils.config['target']]
# encoding labels
le = LabelEncoder()
y = le.fit_transform(y)
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

data_final[fs_utils.config['target']] = y

# Start the timer for checking model accuracy with all features
# If train-test separation is enabled, use the test_df for evaluation

print("Training model on all features")

start = time.perf_counter()

if fs_utils.config['train_test_sep']:   
    acc_full_feature, auc_full_feature = fs_utils.check_model_acc_full_feature(df = data_final, 
                                                                train_test_sep = 1, 
                                                                test_df = test_df,
                                                                debug=0
                                                                )
elif fs_utils.config['train_test_sep'] == False:
    acc_full_feature, auc_full_feature = fs_utils.check_model_acc_full_feature(df = data_final, debug=0)

# End the timer for checking model accuracy with all features
end = time.perf_counter()

# Print the results

print(f"\nAccuracy with all features: {acc_full_feature:.4f}")
print(f"AUC with all features: {auc_full_feature:.4f}")

print("-"*25)
print(f'\n\nExecution time for Model {fs_utils.config["model_name"]} and dataset {fs_utils.config["dataset"]} with {data_final.shape[1]-1} features: {end - start:.4f} seconds')

perc_sel = 0.01 # percentage of features to select randomly in each run

start = time.perf_counter()

if fs_utils.config['train_test_sep']:
    batch_numbers, accuracy_values, auc_values = fs_utils.test_random_features_model_accuracy(
                                df = data_final,
                                batch_size = int(perc_sel* data_final.shape[1]),
                                debug = 0,
                                train_test_sep = 1, 
                                test_df = test_df
                                )
                                
else:
    batch_numbers, accuracy_values, auc_values = fs_utils.test_random_features_model_accuracy(
                                df = data_final,
                                batch_size = int(perc_sel* data_final.shape[1]-1),
                                debug = 0,
                                )

end = time.perf_counter()

print(f"\n\nAccuracy with random features: {np.mean(accuracy_values):.4f} +/- {np.std(accuracy_values):.4f}")
print(f"AUC with random features: {np.mean(auc_values):.4f} +/- {np.std(auc_values):.4f}")
print(f'Execution time for Model {fs_utils.config["model_name"]} and dataset {fs_utils.config["dataset"]} with {int(perc_sel*data_final.shape[1])-1} ({perc_sel*100:.0f}%) features: {end - start:.4f} seconds')

start = time.perf_counter()

if fs_utils.config['train_test_sep']:
    result_csv_file = fs_utils.get_result_random_sets(data_final, train_test_sep=1, test_df=test_df)
elif fs_utils.config['train_test_sep'] == False: 
    result_csv_file = fs_utils.get_result_random_sets(data_final)

end = time.perf_counter()

print(f'\n\nExecution time for Model {fs_utils.config["model_name"]} and dataset {fs_utils.config["dataset"]} with various random subsets:{end - start:.4f} seconds')

result_df = pd.read_csv(result_csv_file)

num_samples = data_final.shape[0]
num_features = data_final.shape[1]-1

fs_utils.plot_random_features_acc_auc(num_samples, num_features, acc_full_feature, auc_full_feature, result_csv_file, plot_acc=1, plot_auc=0, debug=0)

fs_utils.plot_random_features_acc_auc(num_samples, num_features, acc_full_feature, auc_full_feature, result_csv_file, plot_acc=0, plot_auc=1, debug=0)

if fs_utils.config['opt_model']:
    result_csv_path = f"./Plots/{fs_utils.config['dataset']}/{fs_utils.config['model_name']}/{fs_utils.config['dataset']}_{fs_utils.config['model_name']}_(O)_random_features_acc_auc_results.csv"
else:
    result_csv_path = f"./Plots/{fs_utils.config['dataset']}/{fs_utils.config['model_name']}/{fs_utils.config['dataset']}_{fs_utils.config['model_name']}_(D)_random_features_acc_auc_results.csv"

if fs_utils.config['annotate_paper_results']:
    x_from_paper, y_from_paper = fs_utils.config['x_from_paper'], fs_utils.config['y_from_paper']
    print(f"Selected features from paper: {x_from_paper}, Accuracy: {y_from_paper:.2f}")
    
    fs_utils.get_area_above_curve(result_csv_path, x_from_paper, y_from_paper, debug=0)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

y = data_final[fs_utils.config["target"]]  
X = data_final.drop(columns=[fs_utils.config['target']])

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

class SubsetFeatureModel(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, feature_idx):
        self.base_estimator = base_estimator
        self.feature_idx = feature_idx

    def fit(self, X, y):
        self.model_ = clone(self.base_estimator)
        self.model_.fit(X.iloc[:, self.feature_idx], y)
        return self

    def predict(self, X):
        return self.model_.predict(X.iloc[:, self.feature_idx])

    def predict_proba(self, X):
        return self.model_.predict_proba(X.iloc[:, self.feature_idx])

# for _ in range(5):
# Parameters
k = 55  # number of features per model
for n_models in [5, 7, 9, 11, 13]:  # ensemble sizes
    # Create random feature subsets
    rng = np.random.default_rng(42)
    # Create random feature subsets
    feature_indices_list = [rng.choice(X.shape[1], size=k, replace=False) for _ in range(n_models)]

    # Prepare estimators
    estimators = []
    for i, idx in enumerate(feature_indices_list):
        clf = LogisticRegression(max_iter=1000, solver='liblinear')
        # clf = RandomForestClassifier(random_state=42)
        estimators.append((f'model_{i}', SubsetFeatureModel(clf, idx)))

    # Voting classifier ensemble
    ensemble = VotingClassifier(estimators=estimators, voting='soft')

    # Cross-validation performance
    scores = cross_val_score(ensemble, X, y, cv=5, scoring='accuracy')
    # scores.mean(), scores.std()

    print(f'with {n_models} RF models and {k} features each:')
    print(f"Mean accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressionCV

# --- Random Feature Voting Ensemble ---
def random_feature_voting_ensemble(X, y, base_models, n_features=35, test_size=0.2, seed=42):
    np.random.seed(seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=seed)
    
    predictions = []
    for model in base_models:
        feat_idx = np.random.choice(X.shape[1], n_features, replace=False)
        clf = clone(model)
        clf.fit(X_train.iloc[:, feat_idx], y_train)
        preds = clf.predict_proba(X_test.iloc[:, feat_idx])
        predictions.append(preds)

    # Soft voting: average probabilities
    avg_preds = np.mean(predictions, axis=0)
    final_preds = np.argmax(avg_preds, axis=1)

    acc = accuracy_score(y_test, final_preds)
    return acc, final_preds

def random_feature_ensemble():
    # Define base models
    lr = LogisticRegression(max_iter=1000, solver='liblinear')
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    xgb = XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric='logloss', random_state=1)

    base_models = [lr, rf, xgb] * 3  # 9 total base models

    num_features = int(np.round(X.shape[1]* 0.01))
    # Run voting ensemble
    acc_voting, preds_voting = random_feature_voting_ensemble(X, y, base_models, n_features=num_features)
    print(f"Voting Ensemble Accuracy: {acc_voting:.4f}")

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
import numpy as np

def cross_validated_random_feature_ensemble(X, y, base_models, n_features=35, n_splits=5, seed=42, mode='voting'):
    """
    Cross-validated evaluation of random-feature ensemble (voting or stacking).

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        base_models (list): List of scikit-learn base models.
        n_features (int): Number of features for each model.
        n_splits (int): Number of CV folds.
        seed (int): Random seed.
        mode (str): 'voting' or 'stacking'.

    Returns:
        Tuple: (mean accuracy, std accuracy, list of fold accuracies)
    """
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        predictions = []

        for model in base_models:
            feat_idx = np.random.choice(X.shape[1], n_features, replace=False)
            clf = clone(model)
            clf.fit(X_train.iloc[:, feat_idx], y_train)

            if mode == 'voting':
                predictions.append(clf.predict_proba(X_test.iloc[:, feat_idx]))
            elif mode == 'stacking':
                predictions.append((clf.predict_proba(X_train.iloc[:, feat_idx]), clf.predict_proba(X_test.iloc[:, feat_idx])))

        if mode == 'voting':
            avg_preds = np.mean(predictions, axis=0)
            final_preds = np.argmax(avg_preds, axis=1)

        elif mode == 'stacking':
            meta_train = np.hstack([p[0] for p in predictions])
            meta_test = np.hstack([p[1] for p in predictions])

            meta_model = LogisticRegressionCV(max_iter=1000)
            meta_model.fit(meta_train, y_train)
            final_preds = meta_model.predict(meta_test)

        acc = accuracy_score(y_test, final_preds)
        accs.append(acc)

    return accs

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Define your base models
lr = LogisticRegression(max_iter=1000, solver='liblinear')
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
xgb = XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric='logloss', random_state=1)
base_models = [lr, rf, xgb] *3 # total 9 base models

# n_features = int(np.round(X.shape[1]* 0.01))  # number of features for each model

n_features = [300, 400, 500, 600, 700, 800, 900, 1000]  # different feature sizes for each model
seed = 42  # random seed for reproducibility

for n_feat in n_features:
    print(f"Running ensemble with {n_feat} features per model and random seed {seed}...")

    # Voting ensemble
    start_time = time.perf_counter()
    accs = cross_validated_random_feature_ensemble(X, y, base_models, n_features=n_feat, seed=seed, mode='voting')
    end_time = time.perf_counter()
    # Stacking ensemble
    # stacking_mean, stacking_std, _ = cross_validated_random_feature_ensemble(X, y, base_models, n_features=n_feat, seed=seed, mode='stacking')

    voting_mean = np.mean(accs)
    voting_median = np.median(accs)
    voting_std = np.std(accs)

    print(f"Voting Ensemble - Mean: {voting_mean:.4f}, Median: {voting_median:.4f}, Std: {voting_std:.4f}, Time taken: {end_time - start_time:.2f} seconds")

print(f'for dataset {fs_utils.config["dataset"]} ({data_final.shape[0]} samples, {data_final.shape[1]-1} features)')
print(f'{y.value_counts()}')
print(f"Random Feature Ensemble with {n_feat} features per model and random seed {seed}:")
print(f"Voting Ensemble: Mean Accuracy = {voting_mean:.4f}, Median Accuracy = {voting_median:.4f}, Std = {voting_std:.4f}")
