from logging import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, TextArea
import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier 

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import label_binarize

from scipy.interpolate import UnivariateSpline
from scipy.optimize import root_scalar
from scipy.integrate import quad

import math
import sys
import os 
import yaml
from itertools import chain
from tqdm import tqdm

print("Python version: {}". format(sys.version))
print("pandas version: {}". format(pd.__version__))
print("NumPy version: {}". format(np.__version__))

print('-'*25)

def set_nature_style():
    plt.rcParams.update({
         # --- Figure & Font ---
    'figure.figsize': (6.8, 4.5),       # ~85mm x 55mm = 1 column in Nature
    'figure.dpi': 300,                  # High-resolution display
    'savefig.dpi': 600,                 # High-resolution export
    'font.size': 7,                     # Base font size (Nature uses small fonts)
    'font.family': 'sans-serif',       
    'font.sans-serif': ['Arial'],     
    'axes.titlesize': 7,               # Subplot titles
    'axes.labelsize': 7,               # Axis labels
    'xtick.labelsize': 6,              # Tick labels
    'ytick.labelsize': 6,
    'legend.fontsize': 6,              # Legend text
    'legend.frameon': True,         

    # --- Axes Style ---
    'axes.spines.top': False,          # Minimal style (Nature style)
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 2,
    'ytick.major.size': 2,

    # --- Save options ---
    'savefig.bbox': 'tight',           # Trim whitespace
    'savefig.transparent': True,       # Transparent background
    })

# plt.rcParams.update({
#     'font.size': 14,            # Base font size
#     'axes.titlesize': 16,       # Title of each subplot
#     'axes.labelsize': 16,       # X/Y axis labels
#     'xtick.labelsize': 14,
#     'ytick.labelsize': 14,
#     'legend.fontsize': 14
# })

def load_config(path="config.yaml", validate=True, confirm=False):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if validate:
        _validate_config(config)

    print("loaded config:\n")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    if confirm:    
        resp = input("\ncontinue with these settings? [y/n]: ").strip().lower()
        if resp not in ("y", "yes"):
            print("aborting.")
            sys.exit(0)

    return config

def _validate_config(cfg):
    assert "dataset" in cfg and isinstance(cfg["dataset"], str), "dataset must be a string"
    assert "target" in cfg and isinstance(cfg["target"], str), "target must be a string"
    assert cfg.get("model_name") in {"RF", "XGB", "SVM"}, "invalid model_name"
    assert isinstance(cfg.get("opt_model", 0), int), "opt_model must be int"
    assert isinstance(cfg.get("random_state", 42), int), "random_state must be int"
    assert isinstance(cfg.get("num_runs", 1), int) and cfg["num_runs"] > 0, "num_runs must be positive int"
    assert isinstance(cfg.get("remove_cols", False), bool), "remove_cols must be bool"
    assert cfg.get("plot_acc", 0) in {0,1}, "plot_acc must be 0 or 1"
    assert cfg.get("plot_auc", 1) in {0,1}, "plot_auc must be 0 or 1"

config = load_config()

if config['train_test_sep']:
    save_fig_path = "./Plots/"+config['dataset']+"/"+"/"+config['model_name']+"_train_test_sep/"
else:
    save_fig_path = "./Plots/"+config['dataset']+"/"+"/"+config['model_name']+"/"

np.random.seed(config['random_state'])

# set_nature_style() # Uncomment to set Nature style for plots

# plt.rcdefaults() # Uncomment to reset the plot parameters 

def assert_fraction(value):
    assert isinstance(value, (float, int)), "value must be a number"
    assert 0 < value < 1, "value must be a fraction between 0 and 1 (exclusive)"
    # print("value is valid")


def check_scaling(X):
    if isinstance(X, pd.DataFrame):
        X = X.values

    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    medians = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)

    # print("For StandardScaler:")
    # print(f'mean per feature (should be ~0) :\n{means[:5]}')
    # print(f'std per feature (should be ~1) :\n{stds[:5]}')
    print('-'*25)
    print("For RobustScaler:")
    print(f'median per feature (should be ~0) :\n{medians[:5]}')
    print(f'iqr per feature (should be ~1) :\n{iqr[:5]}')
    print('-'*25)

def get_optimized_rf_model(X_train, y_train, X_test, y_test, random_state=config['random_state']):
    # RandomForestClassifier(n_estimators=10, random_state=random_state),


    opt = RandomizedSearchCV(
        RandomForestClassifier(),
        {
            'n_estimators':[20,50, 100, 200, 500],
            'criterion':['log_loss', 'gini','entropy']
        },
        n_iter=100,
        cv=5,
        error_score='raise',
        n_jobs=-1,
        verbose=True, 
        scoring='accuracy'
    )

    opt.fit(X_train, y_train)


    print("Best model: %s" % opt.best_estimator_)
    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(X_test, y_test))
    return opt.best_estimator_

def get_model(model_name=config['model_name'], dataset=config['dataset'], random_state=config['random_state'], opt_model=config['opt_model']):
  
  if not(opt_model):
      model_dict = {
            'Logistic Regression':lambda: LogisticRegression(class_weight='balanced', n_jobs=-1, random_state=random_state),
            'LR':lambda: LogisticRegression(class_weight='balanced', n_jobs=-1, random_state=random_state),

            'SVM':lambda: LinearSVC(dual=True, class_weight='balanced', random_state=random_state),

            'Decision Tree':lambda: DecisionTreeClassifier(class_weight='balanced', random_state=random_state),
            'DT':lambda: DecisionTreeClassifier(class_weight='balanced', random_state=random_state),

            'Random Forest': lambda: RandomForestClassifier(class_weight='balanced', random_state=random_state),
            'RF':lambda: RandomForestClassifier(class_weight='balanced', random_state=random_state),

            "Neural Network":lambda: MLPClassifier(hidden_layer_sizes=[256, 128], batch_size=32, random_state=random_state),
            "NN":lambda: MLPClassifier(hidden_layer_sizes=[256, 128], batch_size=32, random_state=random_state),
            "MLP":lambda: MLPClassifier(hidden_layer_sizes=[256, 128], batch_size=32, random_state=random_state),

            "XGBoost": lambda: XGBClassifier(objective='binary:logistic', random_state=random_state),
            "XGB": lambda: XGBClassifier(objective='binary:logistic', random_state=random_state),

            'GBM': lambda: GradientBoostingClassifier(random_state=random_state),

            'HistGB': lambda: HistGradientBoostingClassifier(random_state=random_state), 

            'Ridge': lambda: RidgeClassifier(class_weight='balanced', random_state=random_state),

            'SGD': lambda: SGDClassifier(class_weight='balanced', random_state=random_state),
            }
  elif opt_model:
      if dataset == "GSE4115":
          model_dict = {
            'Random Forest': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 4, min_samples_split = 2, n_estimators = 300, random_state=random_state),
            'RF': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 4, min_samples_split = 2, n_estimators = 300, random_state=random_state),
          }
      elif dataset == "ALL_AML":
          model_dict = {
            'Random Forest': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 2, min_samples_split = 2, n_estimators = 100, random_state=random_state),
            'RF': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 2, min_samples_split = 2, n_estimators = 100, random_state=random_state),
            'GBM': lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=random_state)
          }
      elif dataset =="Colon":
          model_dict = {
            'Random Forest': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 200, random_state=random_state),
            'RF': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 2, min_samples_split = 2, n_estimators = 200, random_state=random_state),
          }
      elif dataset == "Arcene":
          model_dict = {
            'Random Forest': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 1, min_samples_split = 10, n_estimators = 200, random_state=random_state),
            'RF': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 1, min_samples_split = 10, n_estimators = 200, random_state=random_state)
          }
      elif dataset == "Data_Bischoff2021_Lung":
            model_dict = {
                'Random Forest': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 200, random_state=random_state),
                'RF': lambda: RandomForestClassifier(bootstrap = False, max_depth = None, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 200, random_state=random_state)
            }

  model_name = model_name.strip()
  
  if 'model_dict' in locals():
    if model_name not in model_dict:
            print("Model name should be one of", list(model_dict.keys()))
            print("Please check the entered model name for spelling errors.")
            sys.exit()
    else:
        model = model_dict[model_name]()
        return model
  else:
      print("No model dictionary found for the current context.")
      sys.exit()


def check_model_acc_full_feature(df, debug=config['debug'], average='macro', opt_model=config['opt_model'], train_test_sep=config['train_test_sep'], test_df = pd.DataFrame()):
    ''' 
        Parameters:
                df (pandas DataFrame): A dataframe containing features and a target (in the last column)
                debug (bool): Whether to print debug information
                average (str): The averaging method to use for multi-class classification. Defaults to 'macro'.
                opt_model (bool): Whether to use an optimized model. Defaults to False. 
                train_test_sep (bool): Whether to use a separate test set for evaluation. Defaults to False.
                test_df (pandas DataFrame): A separate test dataframe to use if train_test_sep is True.

        Returns:
                avg_acc (float): Average accuracy for the 3/5-fold cross-validation  
                avg_auc (float): Average AUC for the 3/5-fold cross-validation

    '''   
    target = config['target']
    model_name=config['model_name']
    dataset=config['dataset']
    if debug:
        print(f"Running the experiment with the following parameters:\n"
            f"Dataset: {dataset}, Model: {model_name}, Average: {average}, \n"
            f"Are train and test sets different?: {train_test_sep}, \n"
            f"Print Debug Info: {debug}")
    
    if not(train_test_sep):
        
        y = df[target]
        x = df.drop(columns=[target])

    elif train_test_sep:
        if debug:
            print(f'Using a separate test set for {dataset} dataset.')
        y = df[target]
        x = df.drop(columns=[target])

        y_test_sep = test_df[target]
        X_test_sep = test_df.drop(columns=[target])

    n_classes = len(np.unique(y))

    acc_list = []
    auc_list = []
    b_acc_list = []
    f1_list = []
    prec_list = []
    rec_list = []

    if train_test_sep == 0: 
        # If train and test are 80:20 split of the same dataset, then we do 3/5-fold cross-validation. 
        # For cases, where train and test sets are seprately available, we use them as such
        if x.shape[0] > 100:
            if debug:
                print(f"Using StratifiedKFold with 5 folds for {dataset} dataset")
            fold_n = 5
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['random_state'])
        else:
            if debug:
                print(f"Using StratifiedKFold with 3 folds for {dataset} dataset")
            fold_n = 3
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=config['random_state'])

        for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
            if config['shuffle_cols']:
                if debug:
                    print("Shuffling columns")
                selected_columns = np.random.choice(x.columns, size=int(1* x.shape[1]), replace=False)
                x = x[selected_columns]

                if debug:
                    print(f"Fold {fold+1}: Selected {len(selected_columns)} columns for training and testing.")

                assert len(selected_columns) == df.shape[1]-1
                assert selected_columns.dtype == object  # should be column names
                assert not x[selected_columns].isnull().any().any()  # sanity check

            X_train_raw, X_test_raw = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            
            tree_models = ["RF", "DT", "XGB", "LGB"]
            if config['model_name'] not in tree_models:
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X_train_raw)
                X_test = scaler.transform(X_test_raw)
                if debug:
                    print("For training data:")
                    check_scaling(X_train)
                    print("For test data:")
                    check_scaling(X_test)

            else:
                X_train = X_train_raw
                X_test = X_test_raw    
            

            if debug:
                print(f"fold {fold}: train={len(train_idx)}, test={len(test_idx)}")    
                print("\n\n******** Train ************\n\n")
                print(X_train.shape, y_train.shape)
                print("\n\n********* Test ***********\n\n")
                print(X_test.shape, y_test.shape)
                print("\n\n**************************\n\n")
            
            # fit model
            model = get_model(model_name=model_name, dataset=dataset, random_state=config['random_state'], opt_model=opt_model)
            if debug:
                print(f"Fitting model: {model_name} on fold {fold+1}")
            model.fit(X_train, y_train)
            
            # make predictions
            predictions = model.predict(X_test)


            try:
                y_scores = model.predict_proba(X_test)
            except AttributeError:
                y_scores = model.decision_function(X_test)

            y_scores = np.array(y_scores)
            y_test = np.array(y_test)
        
            full_probs = np.zeros((len(X_test), n_classes))
            for i, c in enumerate(model.classes_):  # these are class *indices*, already encoded
                full_probs[:, c] = y_scores[:, i]
        
            # roc & pr auc for binary
            if n_classes == 2:
                if y_scores.ndim == 2:
                    y_scores = y_scores[:, 1]  # prob or score for positive class
                
                roc_auc = roc_auc_score(y_test, y_scores)
                pr_auc = average_precision_score(y_test, y_scores)
                if debug:
                    print(f"Binary classification: AUC={roc_auc}, PR AUC={pr_auc}")
                    print(f"y_scores shape: {y_scores.shape}, y_test shape: {y_test.shape}")  
                    
                    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
                    roc_auc = auc(fpr, tpr)

                    plt.figure(figsize=(6, 5))
                    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
                    plt.plot([0, 1], [0, 1], 'k--', label="Chance level (AUC = 0.5)")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("Receiver Operating Characteristic (ROC)")
                    plt.legend(loc="lower right")
                    plt.show()

            else:
                try:
                    if y_scores.shape[1] != n_classes:
                        raise ValueError(f"Expected y_scores to have shape (n_samples, {n_classes}), got {y_scores.shape}")
                except IndexError as e:
                    print(f"Error: {e}")
                    

                y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

                if average is None:         # Per class AUCs
                    roc_auc = [roc_auc_score(y_test_bin[:, i], full_probs[:, i]) for i in range(n_classes)]
                    pr_auc = [average_precision_score(y_test_bin[:, i], full_probs[:, i]) for i in range(n_classes)]
                else:
                    roc_auc = roc_auc_score(y_test, full_probs, multi_class='ovo', average=average)
                    pr_auc = average_precision_score(y_test_bin, full_probs, average=average)

                if debug:
                    print("Multiclass classification")
                    print("Number of classes: ", len(np.unique(y_test)))
                    print(f"AUC={roc_auc}, PR AUC={pr_auc}")
                    print(f"y_scores shape: {full_probs.shape}, y_test shape: {y_test.shape}")
                    print(f"y scores: {full_probs}")
                    print(f"y test: {y_test}")


            auc_list.append(roc_auc)

            if debug:
                print(classification_report(y_test, predictions, zero_division=0))
            
        
            accuracy = accuracy_score(y_test, predictions)
            acc_list.append(accuracy)

            b_accuracy = balanced_accuracy_score(y_test, predictions)
            b_acc_list.append(b_accuracy)

            f1score = f1_score(y_test, predictions, average='weighted', zero_division=0)
            f1_list.append(f1score)

            precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
            prec_list.append(precision)

            recall = recall_score(y_test, predictions, average='weighted')
            rec_list.append(recall)

            if debug:
                print(f'Accuracy: {accuracy}', end='\t')
                print(f'AUC: {roc_auc}', end='\t')
                print(f'Balanced Accuracy: {b_accuracy}', end='\t')
                print(f'F1 score: {f1score}', end='\t')
                print(f'Precision: {precision}', end='\t')
                print(f'Recall: {recall}')


        avg_acc = np.average(acc_list)
        avg_auc = np.average(auc_list)

        if debug:
            print(f"\n\nFor {dataset} dataset, Average Accuracy of {fold_n}-fold cross-validation: {avg_acc:.4f}")
            print(f"For {dataset} dataset, Average AUC of {fold_n}-fold cross-validation: {avg_auc:.4f}")


        if debug:
            print("--"* 25)
            print(f"\n\n{fold_n}-fold cross-validation results:")
            print("Accuracy: ", avg_acc, end='\t')
            print("Balanced Accuracy: ", np.average(b_acc_list), end='\t')
            print("F1 score: ", np.average(f1_list), end='\t')
            print("Precision: ", np.average(prec_list), end='\t')
            print("Recall: ", np.average(rec_list))
            print("--"* 25)
        
        return avg_acc, avg_auc

    if train_test_sep:

        if debug:
            print(f'Because train and test sets are different, we will evaluate the model on a separate test set with all features.')
        model_sep = get_model(model_name=model_name, dataset=dataset, random_state=config['random_state'], opt_model=opt_model)
        model_sep.fit(x, y)

        predictions_test = model_sep.predict(X_test_sep)

        if debug:
            print(classification_report(y_test_sep, predictions_test))
        accuracy_test = accuracy_score(y_test_sep, predictions_test)
        
        try:
            y_scores_sep = model_sep.predict_proba(X_test_sep)
        except AttributeError:
            y_scores_sep = model_sep.decision_function(X_test_sep)

        y_scores_sep = np.array(y_scores_sep)
        y_test_sep = np.array(y_test_sep)

        full_probs = np.zeros((len(X_test_sep), n_classes))
        for i, c in enumerate(model_sep.classes_):  # these are class *indices*, already encoded
            full_probs[:, c] = y_scores_sep[:, i]

        # roc & pr auc for binary
        if n_classes == 2:
            if y_scores_sep.ndim == 2:
                y_scores_sep = y_scores_sep[:, 1]  # prob or score for positive class

            roc_auc = roc_auc_score(y_test_sep, y_scores_sep)
            pr_auc = average_precision_score(y_test_sep, y_scores_sep)
            if debug:
                print(f"Binary classification: AUC={roc_auc}, PR AUC={pr_auc}")
                print(f"y_scores shape: {y_scores_sep.shape}, y_test shape: {y_test_sep.shape}")

                fpr, tpr, thresholds = roc_curve(y_test_sep, y_scores_sep)
                roc_auc = auc(fpr, tpr)

                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
                plt.plot([0, 1], [0, 1], 'k--', label="Chance level (AUC = 0.5)")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("Receiver Operating Characteristic (ROC)")
                plt.legend(loc="lower right")
                plt.show()

        else:
            try:
                if y_scores_sep.shape[1] != n_classes:
                    raise ValueError(f"Expected y_scores to have shape (n_samples, {n_classes}), got {y_scores_sep.shape}")
            except IndexError as e:
                print(f"Error: {e}")
                

            y_test_bin = label_binarize(y_test_sep, classes=np.arange(n_classes))

            if average is None:         # Per class AUCs
                roc_auc = [roc_auc_score(y_test_bin[:, i], full_probs[:, i]) for i in range(n_classes)]
                pr_auc = [average_precision_score(y_test_bin[:, i], full_probs[:, i]) for i in range(n_classes)]
            else:
                roc_auc = roc_auc_score(y_test_bin, full_probs, multi_class='ovo', average=average)
                pr_auc = average_precision_score(y_test_bin, full_probs, average=average)

            if debug:
                print("Multiclass classification")
                print("Number of classes: ", len(np.unique(y_test_sep)))
                print(f"AUC={roc_auc}, PR AUC={pr_auc}")
                print(f"y_scores shape: {full_probs.shape}, y_test shape: {y_test_sep.shape}")
                print(f"y scores: {full_probs}")
                print(f"y test: {y_test_sep}")


        y_pred_prob_sep = model_sep.predict_proba(X_test_sep)[:, 1]
        # auc_test = roc_auc_score(y_test_sep, y_pred_prob_sep)

        if debug:
            print("Accuracy on separate test set: ", accuracy_test, end='\t')
            print("AUC on separate test set: ", roc_auc, end='\t')

        if n_classes == 2 and debug:
            y_true = y_test_sep
            y_pred_proba = y_pred_prob_sep  # probs for class 1

            # split by true class
            probs_class0 = [p for p, y in zip(y_pred_proba, y_true) if y == 0]
            probs_class1 = [p for p, y in zip(y_pred_proba, y_true) if y == 1]

            plt.figure(figsize=(8, 4))
            sns.histplot(probs_class0, color='blue', label='True class 0', bins=20, stat='density', kde=True)
            sns.histplot(probs_class1, color='red', label='True class 1', bins=20, stat='density', kde=True)

            plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold 0.5')
            plt.xlabel('Predicted Probability for Class 1')
            plt.ylabel('Density')
            plt.title('Predicted Probability Distribution by True Class')
            plt.legend()
            plt.tight_layout()
            plt.show()

            thresholds = np.linspace(0.3, 0.8, 100)
            f1_scores = []

            best_thresh = 0
            best_f1 = 0
            for t in thresholds:
                preds = (y_pred_proba >= t).astype(int)
                f1 = f1_score(y_true, preds)
                f1_scores.append(f1)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = t

            print(f"Best threshold: {best_thresh:.3f}, Best F1: {best_f1:.3f}")

            # use chosen threshold
            preds = (y_pred_proba >= best_thresh).astype(int)

            # now compute accuracy
            acc = accuracy_score(y_true, preds)
            print(f"Accuracy at threshold {best_thresh:.3f}: {acc:.4f}")

        return accuracy_test, roc_auc


def random_features_model_accuracy(df, batch_size=20, debug=config['debug'], train_test_sep=config['train_test_sep'], test_df = pd.DataFrame()):
    '''
        Parameters:
                df (pandas.DataFrame): Original unmodified dataframe containing features and a target (in the last column)
                batch_size (int): Number of features to randomly select in each batch
                debug (bool): Whether to print debug information
                train_test_sep (bool): Whether to use a separate test set for evaluation. 
                test_df (pandas.DataFrame): A separate test dataframe to use if train_test_sep is True.

        Returns:
                batch_numbers (list): A list of batch numbers
                accuracy_values (list): A list of accuracy values corresponding to each batch
                auc_values (list): A list of AUC values corresponding to each batch (if applicable).
    '''
    target=config['target']
    remove_cols=config['remove_cols']
    stop_at = config['num_runs']   # Number of runs per random feature set

    # Initialize list to store accuracy values
    accuracy_values = []
    auc_values = []
    batch_numbers = []

    # Initialize batch counter
    batch_number = 1

    y = df[target]
    n_classes = len(np.unique(y))
    
    n_cols = df.shape[1]-1

    if debug:
        print(f"Number of classes in target variable '{target}': {n_classes}")
        print(f"Type of the target variable 'n_classes': {type(n_classes)}")
        

    df_rand_feat = df.drop(columns=[target]).copy()  # Drop the target column to get features only


    while len(df_rand_feat.columns) >= batch_size:
        # Randomly select columns
        selected_columns = np.random.choice(df_rand_feat.columns, size=batch_size, replace=False)

        # Extract the selected features
        X_batch = df_rand_feat[selected_columns]

        if not(train_test_sep):
        # Split the data into training and testing sets
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_batch, y, stratify=y, test_size=0.2, random_state=config['random_state'])

        elif train_test_sep:

            X_train_raw = X_batch
            y_train = y

            X_test_raw = test_df[selected_columns]
            y_test = test_df[target]

        tree_models = ["RF", "DT", "XGB", "LGB"]
        if config['model_name'] not in tree_models:
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            if debug:
                print("--"* 25)
                print("For training data:")
                check_scaling(X_train)
                print("--"* 25)
                print("For test data:")
                check_scaling(X_test)
                print("--"* 25)

        else:
            X_train = X_train_raw
            X_test = X_test_raw

        model = get_model(config['model_name'], opt_model=config['opt_model'])
        model.fit(X_train, y_train)
        
        if n_classes == 2:
            y_pred_prob = model.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, y_pred_prob)
            auc_values.append(auc)
        else:
            y_pred_prob = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovo')
            auc_values.append(auc)

        # Predict on the test set and calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store the accuracy value
        accuracy_values.append(accuracy)
        batch_numbers.append(batch_number)

        if debug: 
            try:
                print(f"Batch {batch_number} - Accuracy: {accuracy:.4f}")
                print(f"AUC: {auc:.4f}")
            except:
                pass
        
        if debug and n_classes == 2:
            
            y_true = y_test
            y_pred_proba = y_pred_prob  # probs for class 1

            # split by true class
            probs_class0 = [p for p, y in zip(y_pred_proba, y_true) if y == 0]
            probs_class1 = [p for p, y in zip(y_pred_proba, y_true) if y == 1]

            plt.figure(figsize=(8, 4))
            sns.histplot(probs_class0, color='blue', label='True class 0', bins=20, stat='density', kde=True)
            sns.histplot(probs_class1, color='red', label='True class 1', bins=20, stat='density', kde=True)

            plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold 0.5')
            plt.xlabel('Predicted Probability for Class 1')
            plt.ylabel('Density')
            plt.title('Predicted Probability Distribution by True Class')
            plt.legend()
            plt.tight_layout()
            plt.show()

            thresholds = np.linspace(0.3, 0.8, 100)
            f1_scores = []

            best_thresh = 0
            best_f1 = 0
            for t in thresholds:
                preds = (y_pred_proba >= t).astype(int)
                f1 = f1_score(y_true, preds)
                f1_scores.append(f1)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = t

            print(f"Best threshold: {best_thresh:.3f}, Best F1: {best_f1:.3f}")

            # use chosen threshold
            preds = (y_pred_proba >= best_thresh).astype(int)

            # now compute accuracy
            acc = accuracy_score(y_true, preds)
            print(f"Accuracy at threshold {best_thresh:.3f}: {acc:.4f}")


        # Remove the selected columns from the DataFrame
        if remove_cols:
            if debug:
                print("Removing columns")
            df_rand_feat = df_rand_feat.drop(columns=selected_columns).copy()  # Use copy to avoid SettingWithCopyWarning
        
        # Increment the batch counter
        batch_number += 1

        if batch_number > stop_at:   # Stop after a certain number of runs
            break 

    if n_cols >= 20000 and remove_cols == False:
        print("*-*-"*25)
        print("As you have more than 40,000 features, you could also set remove_cols as false in the config file")
        print("*-*-"*25)
        
    return batch_numbers, accuracy_values, auc_values

def get_result_random_sets(df, debug=config['debug'], train_test_sep=0, test_df = pd.DataFrame()):

    dataset = config['dataset']
    model_name = config['model_name']
    num_runs = config['num_runs']
    remove_cols = config['remove_cols']

    range_tuples = config["feature_ticks_ranges"]   # Sizes of random subsets

    if config['opt_model']:
        to_opt_model = "Optimized"
    else:
        to_opt_model = "Default"

    print(f"""Running the experiment with the following parameters:
          Dataset: {dataset}, 
          Are train and test sets different?: {bool(train_test_sep)},
          Model: {model_name}({to_opt_model}), 
          Number of runs per expr: {num_runs}, 
          Remove columns?: {remove_cols}, 
          Print Debug Info: {bool(debug)}
          """)
    if config['interactive']:
        # If interactive mode is enabled, ask for confirmation before proceeding
        print("Interactive mode is enabled. Please confirm the settings before proceeding.")
        resp = input("\ncontinue with these settings? [y/n]: ").strip().lower()
        if resp not in ("y", "yes"):
            print("aborting.")
            sys.exit(0)
    else:
        print("Interactive mode is disabled. Proceeding with the settings without confirmation.")

    batch_list = []

    batch_mean_acc_list = []

    batch_std_acc_list = []

    batch_mean_auc_list = []

    batch_std_auc_list = []


    if debug: 
        print(f"Mean and standard deviation are for {num_runs} runs.")
    
    length = sum(len(range(start, stop, step)) for start, stop, step in range_tuples)
    pbar = tqdm(total=length, desc="Batching")

    if train_test_sep and test_df.empty:
        raise ValueError("Test set is empty, but 'train_test_sep' is True.")

    for batch in chain(*(range(start, stop, step) for start, stop, step in range_tuples)):
        
        if debug:
            print(f"\n For random gene set size {batch}:")

        if train_test_sep == 0:    
            batch_numbers, accuracy_random, auc_random = random_features_model_accuracy(df = df, 
                                                                                        batch_size = batch, 
                                                                                        debug = debug)
        elif train_test_sep == 1:
            # If train_test_sep is 1, we use a separate test set
            # We assume test_df is already provided and has the same structure as df
            batch_numbers, accuracy_random, auc_random = random_features_model_accuracy(df = df, 
                                                                                        batch_size = batch, 
                                                                                        debug = debug, 
                                                                                        train_test_sep = train_test_sep, 
                                                                                        test_df = test_df)


        mean_acc = round(np.mean(accuracy_random), 5)
        std_acc = round(np.std(accuracy_random), 5)
        if debug:
            print(f"Accuracy Mean: {mean_acc}, STD: {std_acc}")

        if len(auc_random) > 0:
            mean_auc = round(np.mean(auc_random), 5)
            std_auc = round(np.std(auc_random), 5)
            if debug:
                print(f"AUC Mean: {mean_auc}, STD: {std_auc}")
        else:
            mean_auc = np.nan
            std_auc = np.nan
            if debug:
                print(f"AUC Mean: {mean_auc}, STD: {std_auc}")

        
        batch_list.append(batch)
        batch_mean_acc_list.append(mean_acc)
        batch_std_acc_list.append(std_acc)

        batch_mean_auc_list.append(mean_auc)  
        batch_std_auc_list.append(std_auc)

        pbar.update(1)
    pbar.close()

    print(f"Experiment with model: {model_name}, Dataset: {dataset}")

    if train_test_sep:
        print(f'For this experiment, train and test sets are different.')

    if remove_cols:
        print(f"For each of the {num_runs} runs, genes were removed successively for model training.")
    else:
        pass

    result_df = pd.DataFrame({
        'Random Gene Set Size': batch_list,
        'Accuracy Mean': batch_mean_acc_list,
        'Accuracy STD': batch_std_acc_list,
        'AUC Mean': batch_mean_auc_list,
        'AUC STD': batch_std_auc_list
    })

    if train_test_sep:
        result_csv_path = f'./Plots/{dataset}/{model_name}_train_test_sep/' 
    else:
        result_csv_path = f'./Plots/{dataset}/{model_name}/' 

    if config['opt_model']:
        to_opt_model = "O"
    else:
        to_opt_model = "D"
    
    file_name = f'{dataset}_{model_name}_({to_opt_model})_random_features_acc_auc_results.csv'
    
    if not os.path.exists(result_csv_path):
        os.makedirs(result_csv_path)  

    result_file = os.path.join(result_csv_path, file_name)

    result_df.to_csv(result_file, index=False)
    
    print(f"Saved results to CSV file at {result_file}")

    return result_file

def plot_random_features_acc_auc(num_samples, num_features, acc_full_feature, auc_full_feature, result_csv_file, plot_acc=config['plot_acc'], plot_auc=config['plot_auc'], debug=config['debug']):

    dataset = config['dataset']
    model_name = config['model_name']
    num_runs = config['num_runs']
    remove_cols = config['remove_cols']
    train_test_sep = config['train_test_sep']

    if train_test_sep:
        print(f'For this experiment, train and test sets are different.')
        test_dataset = config['test_dataset']
        print(f'Train dataset: {dataset}')
        print(f'Test dataset: {test_dataset}')
    else:
        print(f'For this experiment, train and test sets are the same from {dataset} dataset.')

    if config['remove_cols']:
        set_selection_method = "w_o_replacement"
    else:
        set_selection_method =  "w_replacement"

    result_df = pd.read_csv(result_csv_file)
    if debug:
        print(f"Loaded results from {result_csv_file}")
        print(result_df.head())

    if plot_acc:
        metric = "Accuracy"
        reference_value = round(acc_full_feature,3)
        x1 = result_df['Random Gene Set Size'][0:50] 
        y1 = result_df['Accuracy Mean'][0:50] 
        y1_std = result_df['Accuracy STD'][0:50] 


        x2 = result_df['Random Gene Set Size'][50:65] 
        y2 = result_df['Accuracy Mean'][50:65]
        y2_std = result_df['Accuracy STD'][50:65]


        x3 = result_df['Random Gene Set Size'][65:84]
        y3 = result_df['Accuracy Mean'][65:84]
        y3_std = result_df['Accuracy STD'][65:84]


    elif plot_auc:
        metric = "AUC"
        reference_value = round(auc_full_feature,3)
        x1 = result_df['Random Gene Set Size'][0:50]
        y1 = result_df['AUC Mean'][0:50]
        y1_std = result_df['AUC STD'][0:50]


        x2 = result_df['Random Gene Set Size'][50:65]
        y2 = result_df['AUC Mean'][50:65]
        y2_std = result_df['AUC STD'][50:65]


        x3 = result_df['Random Gene Set Size'][65:84]
        y3 = result_df['AUC Mean'][65:84]
        y3_std = result_df['AUC STD'][65:84]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(16, 8))

    # First axis for 1-50
    ax1.errorbar(x1, y1, yerr=y1_std, fmt='-o', capsize=5, color='blue')
    
    ax1.set_xlim(min(x1), max(x1))
    ax1.set_ylim(0, 1.01)
    ax1.set_xticks(range(min(x1)-1, max(x1)+1, 10))    
    ax1.tick_params(axis='x', rotation=45, labelsize=11)
    ax1.set_yticks((np.arange(0, 1.01, step=0.1)))

    ax1.set_title(f'Range {min(x1)}-{max(x1)} with step size {x1.iloc[1] - x1.iloc[0]}', fontsize=12)
    ax1.set_ylabel(f'{metric} (mean ± std over {num_runs} Runs)', fontsize=16)

    # Second axis for 51-200

    ax2.errorbar(x2, y2, yerr=y2_std, fmt='-o', capsize=5, color='red')
    
    ax2.set_xlim(min(x2), max(x2))

    ax2.set_xticks(range(min(x2), max(x2)+1, 20))
    ax2.tick_params(axis='x', rotation=45, labelsize=11)

    ax2.set_title(f'Range {min(x2)}-{max(x2)} with step size {x2.iloc[1] - x2.iloc[0]}', fontsize=12)
    ax2.set_xlabel('Number of randomly selected features', fontsize=16)
    

    # Third axis for 200-2000
    ax3.errorbar(x3, y3, yerr=y3_std, fmt='-o', capsize=5, color='green')
    
    ax3.set_xlim(min(x3), max(x3))

    ax3.set_title(f'Range {min(x3)}-{max(x3)} with step size {x3.iloc[1] - x3.iloc[0]}', fontsize=12)
    ax3_x_ticks = list(range(min(x3)+100, max(x3)+100, 200))
    ax3.set_xticks(ax3_x_ticks)
    ax3.tick_params(axis='x', rotation=45, labelsize=11)

    # Hide the spines between the two axes
    ax1.spines['right'].set_visible(False)

    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax3.spines['left'].set_visible(False)

    ax1.axhline(y=reference_value, color='b', linestyle='--', label=f'{metric} with all features ({num_features})\n: {reference_value}')
    ax2.axhline(y=reference_value, color='b', linestyle='--', label=f'{metric} with all features ({num_features})\n: {reference_value}')
    ax3.axhline(y=reference_value, color='b', linestyle='--', label=f'{metric} \n with all features ({num_features}): {reference_value}')

    ax1.axhline(y=reference_value-0.02, color='g', linestyle='--', label=f'within 2%: {reference_value-0.02}')
    ax2.axhline(y=reference_value-0.02, color='g', linestyle='--', label=f'within 2%: {reference_value-0.02:.3f}')
    ax3.axhline(y=reference_value-0.02, color='g', linestyle='--', label=f'within 2%: {reference_value-0.02:.3f}')
    
    ax1.axhline(y=reference_value-0.05, color='grey', linestyle='--', label=f'within 5%: {reference_value-0.05:.3f}')
    ax2.axhline(y=reference_value-0.05, color='grey', linestyle='--', label=f'within 5%: {reference_value-0.05:.3f}')
    ax3.axhline(y=reference_value-0.05, color='grey', linestyle='--', label=f'within 5%: {reference_value-0.05:.3f}')
    
    if plot_acc:
        if config['annotate_paper_results']:
            x_from_paper = config['x_from_paper']
            y_from_paper = config['y_from_paper']
            Paper = config['paper']
            print(f"Annotating paper results: {Paper}, x: {x_from_paper}, y: {y_from_paper}")
            ax1.annotate(f'Paper: {Paper} \nSelected Features: {x_from_paper} \nAccuracy: {y_from_paper:.2f}', 
                        xy=(x_from_paper, y_from_paper), 
                        #  xytext=(x_from_paper+10, y_from_paper+0.08),
                        xytext=(0.6, 0.3),
                        textcoords='axes fraction',
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
            ax1.scatter(x_from_paper, y_from_paper, color='red', marker='x', s=100, label=f'{x_from_paper} (acc={y_from_paper:.2f})')

    if config['opt_model']:
        model_opt = "O"
    else:
        model_opt = "D"

    if train_test_sep:
        fig.suptitle(f'{metric} vs Random Features \n Train Dataset: {dataset} ({num_samples} samples, {num_features} features) Test Dataset: {test_dataset}')   
    else:
        fig.suptitle(f'{metric} vs Random Features; {dataset} Dataset ({num_samples} samples, {num_features} features)')   

    ax2.legend(loc='lower center', fontsize=11)


    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)  

    fname = dataset+"_"+model_name+"_("+model_opt+")_"+metric+"_"+set_selection_method.replace(" ","_")+".png"
    file_path = os.path.join(save_fig_path, fname)
    
    print(f"Saving figure to {file_path}")
    plt.savefig(file_path, dpi=600)
    plt.show()


def cv_spline_scores(x, y, s_values, k=3, cv_folds=5):
    scores = []
    for s in s_values:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=config['random_state'])
        fold_scores = []
        for train_idx, test_idx in kf.split(x):
            spline = UnivariateSpline(x[train_idx], y[train_idx], s=s, k=k)
            y_pred = spline(x[test_idx])
            fold_scores.append(mean_squared_error(y[test_idx], y_pred))
        scores.append(np.mean(fold_scores))
    return np.array(scores)

def get_area_above_curve(result_csv_path, x_from_paper, y_from_paper, debug=1):

    """
    Computes the area above the curve (AAC) for a given model and dataset.
    
    Parameters:
        df (pandas DataFrame): A dataframe containing features and a target (in the last column)
        target (str): The name of the target column in the dataframe
        model_name (str): The name of the model to be used
        dataset (str): The name of the dataset
        debug (bool): Whether to print debug information
        opt_model (int): Whether to use an optimized model or not
        train_test_sep (int): If 1, uses a separate test set provided in `test_df`
        test_df (pandas DataFrame): Separate test set dataframe, required if `train_test_sep` is set to 1

    Returns:
        float: The area above the curve (AAC) for the given model and dataset
    """   
    plot_data = pd.read_csv(result_csv_path)

    x_data = plot_data['Random Gene Set Size']
    
    if config['plot_acc']:
        metric = "Accuracy"
        y_data = plot_data['Accuracy Mean']
    elif config['plot_auc']:
        metric = "AUC"
        y_data = plot_data['AUC Mean']
    else:
        raise ValueError("Please set config['plot_acc'] or config['plot_auc'] to True.") 

    assert_fraction(y_from_paper)
    
    if debug:
        print(f"No. of datapoints to approximate the spline fit : {len(x_data)}")

    # search range
    s_values = np.logspace(-2, 3, 50)
    scores = cv_spline_scores(x_data, y_data, s_values)
    best_s = s_values[np.argmin(scores)]

    if debug:
        print(f"Best smoothing factor (s): {best_s:.2g}")
        # fit final spline with best_s
        spline = UnivariateSpline(x_data, y_data, s=best_s)

        # generate points for plotting
        x_fit = np.linspace(np.min(x_data), np.max(x_data), 100)
        y_fit = spline(x_fit)

        # plot
        plt.figure(figsize=(8, 5))
        
        plt.plot(x_data, y_data, 'o', label='data')
        plt.plot(x_fit, y_fit, label=f'spline fit (s={best_s:.2g})')
        
        plt.xlabel('number of features')
        plt.ylabel(f"{metric} + ' value'")
        
        plt.axhline(y=y_from_paper, color='r', linestyle='--', label=f'{metric} from paper: {y_from_paper:.3f}')
        plt.axvline(x=x_from_paper, color='b', linestyle='--', label=f'Features from paper: {x_from_paper}')

        plt.title(f'Spline fit to {metric} data for {config["dataset"]} with {config["model_name"]} model')
        
        plt.xlim(x_data.min(), x_data.max())
        plt.ylim(0, 1.01)
        
        plt.xticks(np.arange(0, x_data.max() + 1, step=200))
        plt.yticks(np.arange(0, 1.01, step=0.1))
        
        plt.legend()
        
        plt.show()

    # interpolate f(t)
    f = UnivariateSpline(x_data, y_data, s=best_s)

    # define function to find root of f(t) - y
    def diff(t):    
        return f(t) - y_from_paper

    # pick a bracket (must ensure f(t) crosses y in this interval)
    # bracket = [x_from_paper + 1e-3, x_data.iloc[-1]]
    bracket = [0, x_data.iloc[-1]]

    try:
        # find root
        print(f"Finding root in the interval {bracket}")
        res = root_scalar(diff, bracket=bracket, method='brentq')
    except ValueError as e:
        print(f"Error: {e}")
        res = None

    if res is not None:
        x_prime = res.root if res.converged else None
        print(f"Similar result can be achieved with {x_prime} features.")
    else:
        x_prime = None
        print("No root found.")
        
    if debug:
        print(f"x: {x_from_paper}, y: {y_from_paper}, x': {x_prime}")

        # compute shaded area between f(t) and y from x to x'

    if x_prime is not None:
        area, _ = quad(lambda t: y_from_paper - f(t), x_from_paper, x_prime)
        print(f"Area above the curve with the point ({x_from_paper}, {y_from_paper:.2f}) from x = {x_from_paper} to x' = {x_prime:.2f} ==> {area:.5f}")
    else:
        print("curve never crosses y again after x: area = 0")
        area = 0

    plt.plot(x_data, f(x_data), label='best spline fit')
    plt.plot(x_from_paper, y_from_paper, 'ro')
    plt.xlim(x_data.min(), x_data.max())
    plt.ylim(0, 1.01)

    try:
        plt.plot(x_prime, f(x_prime), 'go')
        plt.annotate(f" Min. Features needed \n to match: {x_prime:.0f}", xy=(x_prime, f(x_prime)), xytext=(x_prime+800, f(x_prime)-0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=10,
                    horizontalalignment='right',
                    verticalalignment='bottom')
        plt.annotate(f"{metric} from a \n published study: ({x_from_paper:.0f}, {y_from_paper:.4f})", xy=(x_from_paper, y_from_paper), xytext=(x_from_paper+300, y_from_paper+0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=10,
                    horizontalalignment='right',
                    verticalalignment='bottom')

        plt.xlim(x_data.min(), x_data.max())
        plt.ylim(0, 1.01)

        plt.xticks(np.arange(0, x_data.max() + 1, step=200))
        plt.yticks(np.arange(0, 1.01, step=0.1))
        
        plt.xlabel('No of genes randomly selected for model training')
        plt.ylabel(f'Mean {metric} over 20 runs')

        plt.axhline(y=y_from_paper, color='r', linestyle='--')
        plt.axvline(x=x_from_paper, color='b', linestyle='--')

        # shade area between curve and y_ref from x_known to x_root
        x_shade = x_data[(x_data >= x_from_paper) & (x_data <= x_from_paper)]
        y_shade = f(x_shade)
        x_poly = np.concatenate(([x_from_paper], x_shade, [x_prime]))
        y_poly = np.concatenate(([y_from_paper], y_shade, [y_from_paper]))
        plt.fill(x_poly, y_poly, color='skyblue', alpha=0.4, label=f'Approx. Area: {area:.2f}')
    except:
        print("Random set can not match the performance of the published study.")

    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)

    fname = f"{config['dataset']}_{config['model_name']}_({'O' if config['opt_model'] else 'D'})_{metric}_area_plot.png"

    file_path = os.path.join(save_fig_path, fname)

    plt.savefig(file_path, dpi=600)

    plt.legend(loc='best')
    plt.show()

    return area

def test_random_features_model_accuracy(df, batch_size=20, debug=config['debug'], train_test_sep=config['train_test_sep'], test_df = pd.DataFrame()):
    '''
        Parameters:
                df (pandas.DataFrame): Original unmodified dataframe containing features and a target (in the last column)
                batch_size (int): Number of features to randomly select in each batch
                debug (bool): Whether to print debug information
                train_test_sep (bool): Whether to use a separate test set for evaluation. 
                test_df (pandas.DataFrame): A separate test dataframe to use if train_test_sep is True.

        Returns:
                batch_numbers (list): A list of batch numbers
                accuracy_values (list): A list of accuracy values corresponding to each batch
                auc_values (list): A list of AUC values corresponding to each batch (if applicable).
    '''
    target=config['target']
    remove_cols=config['remove_cols']
    stop_at = config['num_runs']   # Number of runs per random feature set

    # Initialize list to store accuracy values
    accuracy_values = []
    auc_values = []
    batch_numbers = []

    # accuracy_cis = []
    # auc_cis = []

    # Initialize batch counter
    batch_number = 1

    y = df[target]
    n_classes = len(np.unique(y))

    if debug:
        print(f"Number of classes in target variable '{target}': {n_classes}")
        print(f"Type of the target variable 'n_classes': {type(n_classes)}")
        

    df_rand_feat = df.drop(columns=[target]).copy()  # Drop the target column to get features only

    while len(df_rand_feat.columns) >= batch_size:
        # Randomly select columns
        selected_columns = np.random.choice(df_rand_feat.columns, size=batch_size, replace=False)

        # Extract the selected features
        X_batch = df_rand_feat[selected_columns]

        if not(train_test_sep):
            df_random_features = pd.concat([X_batch, y], axis=1)
            accuracy, auc = check_model_acc_full_feature(df = df_random_features, debug=0)

        elif train_test_sep:

            train_df_random_features = pd.concat([X_batch, y], axis=1)
            test_df_random_features = pd.concat([test_df[selected_columns], test_df[target]], axis=1)

            accuracy, auc = check_model_acc_full_feature(df = train_df_random_features, 
                                         train_test_sep = 1, 
                                         test_df = test_df_random_features,
                                         debug=0
            )
        
        batch_numbers.append(batch_number)
        auc_values.append(auc)
        accuracy_values.append(accuracy)
        

        if debug: 
            try:
                print(f"Batch {batch_number} - Accuracy: {accuracy:.4f}")
                print(f"AUC: {auc:.4f}")
            except:
                pass

        # Remove the selected columns from the DataFrame
        if remove_cols:
            if debug:
                print("Removing columns")
            df_rand_feat = df_rand_feat.drop(columns=selected_columns).copy()  # Use copy to avoid SettingWithCopyWarning
        
        # Increment the batch counter
        batch_number += 1

        if batch_number > stop_at:   # Stop after a certain number of runs
            break 

    return batch_numbers, accuracy_values, auc_values
