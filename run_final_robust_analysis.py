import pandas as pd
import numpy as np
import time
import warnings
import re
import joblib
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, 
                              ExtraTreesClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

try:
    from catboost import CatBoostClassifier
except ImportError:
    print("[WARN] CatBoost not installed. Skipping.")

warnings.filterwarnings('ignore')

# ==========================================
# 1. Configuration
# ==========================================
INPUT_FILE = 'Merged Research Data By PID with Demographics.csv'
RESULTS_FILE = 'final_robust_analysis_results.csv'
CV_FOLDS = 5
RANDOM_STATE = 42
N_JOBS = -1

# Drug Lists
SSRI_LIST = ['Citalopram', 'Escitalopram', 'Fluoxetine', 'Paroxetine', 'Sertraline']
SNRI_LIST = ['Venlafaxine', 'Duloxetine', 'Desvenlafaxine']

# ==========================================
# 2. Data Cleaning
# ==========================================
def clean_dosage(val):
    try:
        if pd.isna(val): return 0.0
        if isinstance(val, (int, float)): return float(val)
        val_str = str(val).strip()
        numbers = re.findall(r"(\d+(\.\d+)?)", val_str)
        if not numbers: return 0.0
        return max([float(n[0]) for n in numbers])
    except:
        return 0.0

def load_data(filepath):
    print(f"[INFO] Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Map column names if needed (Small dataset uses 'DRUG', large uses 'DRUG_NAME')
    if 'DRUG' in df.columns: df.rename(columns={'DRUG': 'DRUG_NAME'}, inplace=True)
    if 'DOSAGE' in df.columns: df.rename(columns={'DOSAGE': 'DRUG_DOSAGE'}, inplace=True)

    # Filter Placebo
    if 'DRUG_NAME' in df.columns:
        df = df[df['DRUG_NAME'] != 'Placebo'].copy()

    # Encode Sex
    if 'SEX' in df.columns:
        df['SEX'] = df['SEX'].map({'F': 0, 'M': 1})
        if not df['SEX'].mode().empty:
             df['SEX'] = df['SEX'].fillna(df['SEX'].mode()[0])
        else:
             df['SEX'] = df['SEX'].fillna(0)
    
    # Clean Dosage
    df['DRUG_DOSAGE'] = df['DRUG_DOSAGE'].apply(clean_dosage)
    
    # Ensure HAM-D columns are numeric
    hamd_cols = [c for c in df.columns if str(c).startswith('V1-HAMD')]
    for col in hamd_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    # Extract Trial ID
    if 'UNIQUEID' in df.columns:
        df['Trial_ID'] = df['UNIQUEID'].astype(str).apply(lambda x: x.split('-')[0] if '-' in x else 'Unknown')
    else:
        df['Trial_ID'] = 'Unknown'

    print(f"[INFO] Data Loaded. Shape: {df.shape}")
    return df, hamd_cols

# ==========================================
# 3. Labeling Schemes
# ==========================================
def get_label_standard(row):
    drug = str(row['DRUG_NAME']).strip()
    if drug in SSRI_LIST: return 0
    if drug in SNRI_LIST: return 1
    return -1

def get_label_v1(row):
    drug = str(row['DRUG_NAME']).strip()
    dose = row['DRUG_DOSAGE']
    if drug == 'Venlafaxine' and dose >= 150: return 1
    if drug == 'Paroxetine' and dose >= 50: return 1
    if drug == 'Duloxetine' and dose >= 60: return 1
    if drug in SSRI_LIST or drug in SNRI_LIST: return 0
    return -1

def get_label_v2(row):
    drug = str(row['DRUG_NAME']).strip()
    dose = row['DRUG_DOSAGE']
    if drug == 'Venlafaxine' and dose > 150: return 1
    if drug == 'Paroxetine' and dose > 50: return 1
    if drug == 'Duloxetine' and dose > 60: return 1
    if drug in SSRI_LIST or drug in SNRI_LIST: return 0
    return -1

# ==========================================
# 4. Model Zoo (Regularized to prevent Overfitting)
# ==========================================
def get_models():
    models = []
    
    # 1. KNN (Simple baseline)
    models.append(('KNN', KNeighborsClassifier(n_neighbors=15))) # Increased neighbors for smoothing
    
    # 2. Logistic Regression (Strong Regularization)
    models.append(('LogisticRegression', LogisticRegression(C=0.1, max_iter=3000, random_state=RANDOM_STATE)))

    # 3. Linear SVM
    models.append(('Linear_SVM', SVC(kernel="linear", C=0.01, probability=True, random_state=RANDOM_STATE)))
    
    # 4. RBF SVM
    models.append(('RBF_SVM', SVC(gamma='auto', C=1.0, probability=True, random_state=RANDOM_STATE)))
    
    # 5. Decision Tree (Pruned)
    models.append(('DecisionTree', DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=RANDOM_STATE)))
    
    # 6. Random Forest (Regularized)
    # Reduced depth, increased min samples to prevent memorization
    models.append(('RandomForest', RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_split=10, min_samples_leaf=5, random_state=RANDOM_STATE)))
    
    # 7. MLP (Simple Architecture)
    models.append(('MLP', MLPClassifier(hidden_layer_sizes=(32,), alpha=0.1, max_iter=1000, random_state=RANDOM_STATE)))
    
    # 8. AdaBoost (Low learning rate)
    models.append(('AdaBoost', AdaBoostClassifier(algorithm="SAMME", n_estimators=50, learning_rate=0.5, random_state=RANDOM_STATE)))
    
    # 9. Naive Bayes
    models.append(('NaiveBayes', GaussianNB()))
    
    # 10. QDA
    models.append(('QDA', QuadraticDiscriminantAnalysis(reg_param=0.1)))
    
    # 11. Extra Trees (Regularized)
    models.append(('ExtraTrees', ExtraTreesClassifier(n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=RANDOM_STATE)))
    
    # 12. Gradient Boosting (Regularized)
    models.append(('GradientBoosting', GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE)))
    
    # 13. Hist Gradient Boosting (L2 Reg)
    models.append(('HistGradientBoosting', HistGradientBoostingClassifier(max_depth=4, l2_regularization=1.0, learning_rate=0.05, random_state=RANDOM_STATE)))
    
    # 14. CatBoost (if available)
    try:
        from catboost import CatBoostClassifier
        models.append(('CatBoost', CatBoostClassifier(verbose=0, depth=4, l2_leaf_reg=5, random_state=RANDOM_STATE)))
    except: pass
    
    # Ensembles (Using simpler base learners)
    rf_base = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=RANDOM_STATE)
    mlp_base = MLPClassifier(hidden_layer_sizes=(16,), alpha=0.1, max_iter=500, random_state=RANDOM_STATE)
    
    # 15. Stacking
    models.append(('Stacking', StackingClassifier(
        estimators=[('rf', rf_base), ('mlp', mlp_base)],
        final_estimator=LogisticRegression(C=0.1), cv=3)))
    
    # 16. Voting
    models.append(('Voting', VotingClassifier(
        estimators=[('rf', rf_base), ('mlp', mlp_base)], voting='soft')))
    
    return models

# ==========================================
# 5. Main Analysis Loop
# ==========================================
def run_analysis():
    start_time = time.time()
    df_raw, hamd_cols = load_data(INPUT_FILE)
    
    schemes = {
        'Standard': get_label_standard,
        'V1_Inclusive': get_label_v1,
        'V2_Exclusive': get_label_v2
    }
    
    feature_sets = {
        'Symptoms_Only': hamd_cols,
        'With_Demo': ['AGE', 'SEX'] + hamd_cols
    }
    
    smote_opts = [False, True]
    cv_types = {
        'StratifiedKFold': StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        'GroupKFold': GroupKFold(n_splits=CV_FOLDS)
    }
    
    results = []
    models = get_models()
    
    total_steps = len(schemes) * len(feature_sets) * len(smote_opts) * len(cv_types) * len(models)
    step = 0
    
    print(f"\n[INFO] Starting Robust Analysis. Total Steps: {total_steps}")
    
    for scheme_name, label_func in schemes.items():
        # Prepare Labels
        df = df_raw.copy()
        df['Label'] = df.apply(label_func, axis=1)
        df_model = df[df['Label'] != -1].copy()
        y = df_model['Label'].astype(int)
        groups = df_model['Trial_ID']
        
        if len(y.unique()) < 2:
            print(f"[SKIP] {scheme_name} has 1 class.")
            continue
            
        for feat_name, feats in feature_sets.items():
            valid_feats = [f for f in feats if f in df_model.columns]
            X = df_model[valid_feats]
            
            for use_smote in smote_opts:
                for cv_name, cv_splitter in cv_types.items():
                    for model_name, model_inst in models:
                        step += 1
                        print(f"[{step}/{total_steps}] {scheme_name} | {model_name} | {cv_name}")
                        
                        try:
                            # Pipeline Construction
                            steps = [('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]
                            
                            if use_smote:
                                # Careful SMOTE: fewer neighbors to avoid overfitting noise
                                min_samples = y.value_counts().min()
                                k = min(3, min_samples - 1) if min_samples > 1 else 1
                                steps.append(('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=k)))
                            
                            steps.append(('classifier', model_inst))
                            pipeline = ImbPipeline(steps)
                            
                            # Metrics including Train scores to check overfitting
                            scoring = {
                                'AUC': 'roc_auc', 'Acc': 'accuracy', 'F1': 'f1_weighted',
                                'Prec': 'precision_weighted', 'Recall': 'recall_weighted'
                            }
                            
                            # Cross Validate
                            if cv_name == 'GroupKFold':
                                scores = cross_validate(pipeline, X, y, groups=groups, cv=cv_splitter, scoring=scoring, return_train_score=True)
                            else:
                                scores = cross_validate(pipeline, X, y, cv=cv_splitter, scoring=scoring, return_train_score=True)
                            
                            # Store Results
                            res = {
                                'Scheme': scheme_name,
                                'Feature_Set': feat_name,
                                'SMOTE': use_smote,
                                'CV_Type': cv_name,
                                'Model': model_name,
                                # Test Metrics
                                'Test_AUC': np.nanmean(scores['test_AUC']),
                                'Test_Acc': np.nanmean(scores['test_Acc']),
                                'Test_F1': np.nanmean(scores['test_F1']),
                                'Test_Prec': np.nanmean(scores['test_Prec']),
                                'Test_Recall': np.nanmean(scores['test_Recall']),
                                # Train Metrics (Critical for Overfitting check)
                                'Train_AUC': np.nanmean(scores['train_AUC']),
                                'Train_Acc': np.nanmean(scores['train_Acc']),
                                'Train_F1': np.nanmean(scores['train_F1'])
                            }
                            results.append(res)
                            
                            # Log Gap
                            gap = res['Train_Acc'] - res['Test_Acc']
                            print(f"   -> Test Acc: {res['Test_Acc']:.3f} | Train Acc: {res['Train_Acc']:.3f} | Gap: {gap:.3f}")
                            
                        except Exception as e:
                            print(f"   [ERROR] {e}")

    # Save
    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_FILE, index=False)
    print("\n" + "="*40)
    print(f"Complete. Results saved to {RESULTS_FILE}")
    print(f"Total Time: {(time.time()-start_time)/60:.1f} min")

if __name__ == "__main__":
    run_analysis()