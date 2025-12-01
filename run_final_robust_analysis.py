import json
import time
import warnings
import re
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler

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
BOOTSTRAP_CI_FILE = 'metric_bootstrap_intervals.csv'
FEATURE_STABILITY_FILE = 'counterfactual_feature_stability.json'
SHAP_SUMMARY_FILE = 'shap_global_importance.csv'
LIME_SUMMARY_FILE = 'lime_local_explanations.json'
CV_FOLDS = 5
RANDOM_STATE = 42
RANDOM_SEEDS = [42, 1337, 2025]
BOOTSTRAP_SAMPLES = 2000
COUNTERFACTUAL_SAMPLES = 25
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
# 4. Utility Functions
# ==========================================
def bootstrap_ci(scores, alpha=0.05, n_bootstrap=BOOTSTRAP_SAMPLES):
    """
    Compute bootstrap confidence interval for a list/array of scores.
    Returns (mean, lower, upper).
    """
    arr = np.array(scores)
    if arr.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(RANDOM_STATE)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=arr.shape[0], replace=True)
        boot_means.append(np.nanmean(sample))
    lower = np.nanpercentile(boot_means, 100 * (alpha / 2))
    upper = np.nanpercentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.nanmean(arr)), float(lower), float(upper)

def prepare_training_data(df_raw, label_func, feature_list):
    df = df_raw.copy()
    df['Label'] = df.apply(label_func, axis=1)
    df_model = df[df['Label'] != -1].copy()
    y = df_model['Label'].astype(int)
    valid_features = [f for f in feature_list if f in df_model.columns]
    X = df_model[valid_features]
    return df_model, X, y

# ==========================================
# 4. Model Zoo (Regularized to prevent Overfitting)
# ==========================================
def get_models(seed):
    models = []
    
    # 1. KNN (Simple baseline)
    models.append(('KNN', KNeighborsClassifier(n_neighbors=15))) # Increased neighbors for smoothing
    
    # 2. Logistic Regression (Strong Regularization)
    models.append(('LogisticRegression', LogisticRegression(C=0.1, max_iter=3000, random_state=seed)))

    # 3. Linear SVM
    models.append(('Linear_SVM', SVC(kernel="linear", C=0.01, probability=True, random_state=seed)))
    
    # 4. RBF SVM
    models.append(('RBF_SVM', SVC(gamma='auto', C=1.0, probability=True, random_state=seed)))
    
    # 5. Decision Tree (Pruned)
    models.append(('DecisionTree', DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=seed)))
    
    # 6. Random Forest (Regularized)
    # Reduced depth, increased min samples to prevent memorization
    models.append(('RandomForest', RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_split=10, min_samples_leaf=5, random_state=seed)))
    
    # 7. MLP (Simple Architecture)
    models.append(('MLP', MLPClassifier(hidden_layer_sizes=(32,), alpha=0.1, max_iter=1000, random_state=seed)))
    
    # 8. AdaBoost (Low learning rate)
    models.append(('AdaBoost', AdaBoostClassifier(algorithm="SAMME", n_estimators=50, learning_rate=0.5, random_state=seed)))
    
    # 9. Naive Bayes
    models.append(('NaiveBayes', GaussianNB()))
    
    # 10. QDA
    models.append(('QDA', QuadraticDiscriminantAnalysis(reg_param=0.1)))
    
    # 11. Extra Trees (Regularized)
    models.append(('ExtraTrees', ExtraTreesClassifier(n_estimators=150, max_depth=6, min_samples_leaf=5, random_state=seed)))
    
    # 12. Gradient Boosting (Regularized)
    models.append(('GradientBoosting', GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.85, random_state=seed)))
    
    # 13. Hist Gradient Boosting (L2 Reg)
    models.append(('HistGradientBoosting', HistGradientBoostingClassifier(max_depth=4, l2_regularization=1.0, learning_rate=0.05, random_state=seed)))
    
    # 14. CatBoost (if available)
    try:
        from catboost import CatBoostClassifier
        models.append(('CatBoost', CatBoostClassifier(verbose=0, depth=4, l2_leaf_reg=5, random_state=seed)))
    except: pass
    
    # Ensembles (Using simpler base learners)
    rf_base = RandomForestClassifier(n_estimators=75, max_depth=4, random_state=seed)
    mlp_base = MLPClassifier(hidden_layer_sizes=(16,), alpha=0.1, max_iter=500, random_state=seed)
    
    # 15. Stacking
    models.append(('Stacking', StackingClassifier(
        estimators=[('rf', rf_base), ('mlp', mlp_base)],
        final_estimator=LogisticRegression(C=0.1), cv=3)))
    
    # 16. Voting
    models.append(('Voting', VotingClassifier(
        estimators=[('rf', rf_base), ('mlp', mlp_base)], voting='soft')))

    return models

# ==========================================
# 5. Explanatory Analyses (Counterfactual stability + SHAP/LIME)
# ==========================================
def fit_reference_pipeline(df_raw, hamd_cols, scheme_name='Standard', feature_key='With_Demo', seed=RANDOM_STATE):
    schemes = {
        'Standard': get_label_standard,
        'V1_Inclusive': get_label_v1,
        'V2_Exclusive': get_label_v2
    }

    feature_sets = {
        'Symptoms_Only': hamd_cols,
        'With_Demo': ['AGE', 'SEX'] + hamd_cols
    }

    label_func = schemes[scheme_name]
    feature_list = feature_sets[feature_key]

    _, X, y = prepare_training_data(df_raw, label_func, feature_list)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_split=8, min_samples_leaf=4, random_state=seed
    )
    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    pipeline.fit(X, y)
    return pipeline, X, y


def _generate_counterfactuals_robust(exp, query_df, desired_total):
    """Try multiple DiCE configurations to avoid empty counterfactual sets."""
    # Primary attempt
    try:
        return exp.generate_counterfactuals(query_df, total_CFs=desired_total, desired_class="opposite")
    except Exception:
        pass

    # Fallback: allow a larger search and k-d tree neighbor search
    try:
        return exp.generate_counterfactuals(
            query_df,
            total_CFs=desired_total * 2,
            desired_class="opposite",
            method="kdtree",
        )
    except Exception:
        return None


def compute_counterfactual_feature_changes(pipeline, X, y, seed):
    try:
        import dice_ml
    except ImportError:
        print("[WARN] dice-ml not installed. Skipping counterfactual stability analysis.")
        return None

    df_for_dice = X.copy()
    df_for_dice['Label'] = y.values

    data = dice_ml.Data(dataframe=df_for_dice, continuous_features=list(X.columns), outcome_name='Label')
    model = dice_ml.Model(model=pipeline, backend='sklearn')
    exp = dice_ml.Dice(data, model, method='random')

    rng = np.random.default_rng(seed)
    sample_df = df_for_dice.sample(min(COUNTERFACTUAL_SAMPLES, len(df_for_dice)), random_state=seed)
    feature_counts = Counter()

    for _, row in sample_df.iterrows():
        query_x = row.drop('Label')
        query_df = pd.DataFrame([query_x])

        cf_examples = _generate_counterfactuals_robust(exp, query_df, desired_total=3)
        if cf_examples is None:
            print(
                f"   [WARN] Counterfactual generation failed for seed {seed}: No counterfactuals found for any of the query points!"
            )
            continue

        if not cf_examples.cf_examples_list:
            continue

        cf_df = cf_examples.cf_examples_list[0].final_cfs_df
        if cf_df is None or cf_df.empty:
            continue

        for _, cf_row in cf_df.iterrows():
            for feat in X.columns:
                if not np.isclose(cf_row[feat], query_x[feat]):
                    feature_counts[feat] += 1
    return feature_counts


def run_counterfactual_stability(df_raw, hamd_cols):
    stability_payload = {'per_run_top5': [], 'jaccard_summary': []}
    schemes = ['Standard', 'V1_Inclusive', 'V2_Exclusive']

    for scheme in schemes:
        top_sets = []
        for seed in RANDOM_SEEDS:
            pipeline, X, y = fit_reference_pipeline(df_raw, hamd_cols, scheme_name=scheme, feature_key='With_Demo', seed=seed)
            feature_counts = compute_counterfactual_feature_changes(pipeline, X, y, seed)
            if feature_counts is None:
                return None
            top_features = [f for f, _ in feature_counts.most_common(5)]
            top_sets.append(set(top_features))
            stability_payload['per_run_top5'].append({
                'Scheme': scheme,
                'Seed': seed,
                'Top5': top_features
            })

        # Jaccard stability across seeds for this scheme
        jaccards = []
        for i in range(len(top_sets)):
            for j in range(i+1, len(top_sets)):
                if len(top_sets[i] | top_sets[j]) == 0:
                    continue
                score = len(top_sets[i] & top_sets[j]) / len(top_sets[i] | top_sets[j])
                jaccards.append(score)
        if jaccards:
            stability_payload['jaccard_summary'].append({
                'Scheme': scheme,
                'Mean_Jaccard_Top5': float(np.mean(jaccards))
            })
    return stability_payload


def run_shap_global_importance(pipeline, X, scheme_name):
    try:
        import shap
    except ImportError:
        print("[WARN] SHAP not installed. Skipping SHAP analysis.")
        return None

    sample = X.sample(min(500, len(X)), random_state=RANDOM_STATE)
    explainer = shap.Explainer(pipeline.predict_proba, sample)
    shap_values = explainer(sample)
    abs_vals = np.abs(shap_values.values[..., 1]).mean(axis=0)
    df_shap = pd.DataFrame({
        'Scheme': scheme_name,
        'Feature': sample.columns,
        'MeanAbsSHAP': abs_vals
    }).sort_values(by='MeanAbsSHAP', ascending=False)
    return df_shap


def run_lime_local_explanations(pipeline, X, scheme_name):
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        print("[WARN] LIME not installed. Skipping LIME analysis.")
        return None

    explainer = LimeTabularExplainer(
        training_data=X.values,
        feature_names=list(X.columns),
        mode='classification',
        discretize_continuous=False  # avoid discretizer truncnorm issues on low-variance features
    )

    instances = X.sample(min(10, len(X)), random_state=RANDOM_STATE)
    explanations = []
    for idx, row in instances.iterrows():
        try:
            exp = explainer.explain_instance(
                row.values,
                pipeline.predict_proba,
                num_features=min(10, X.shape[1]),
            )
            explanations.append({
                'Scheme': scheme_name,
                'InstanceIndex': int(idx),
                'FeatureWeights': exp.as_list()
            })
        except Exception as e:
            print(f"   [WARN] LIME explanation failed for instance {idx}: {e}")
            continue
    return explanations

# ==========================================
# 6. Main Analysis Loop
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
    bootstrap_rows = []

    total_steps = len(schemes) * len(feature_sets) * len(smote_opts) * len(cv_types) * len(get_models(RANDOM_STATE)) * len(RANDOM_SEEDS)
    step = 0

    print(f"\n[INFO] Starting Robust Analysis. Total Steps: {total_steps}")

    for seed in RANDOM_SEEDS:
        models = get_models(seed)
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
                            print(f"[{step}/{total_steps}] Seed {seed} | {scheme_name} | {model_name} | {cv_name}")

                            try:
                                # Pipeline Construction
                                steps = [('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]

                                if use_smote:
                                    # Careful SMOTE: fewer neighbors to avoid overfitting noise
                                    min_samples = y.value_counts().min()
                                    k = min(3, min_samples - 1) if min_samples > 1 else 1
                                    steps.append(('smote', SMOTE(random_state=seed, k_neighbors=k)))

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
                                    'Seed': seed,
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

                                # Bootstrap confidence intervals on held-out folds
                                for metric_key, score_key in [('AUC', 'test_AUC'), ('Acc', 'test_Acc'), ('F1', 'test_F1'), ('Prec', 'test_Prec'), ('Recall', 'test_Recall')]:
                                    mean_score, low, high = bootstrap_ci(scores[score_key])
                                    res[f'Test_{metric_key}_CI_Low'] = low
                                    res[f'Test_{metric_key}_CI_High'] = high
                                    bootstrap_rows.append({
                                        'Seed': seed,
                                        'Scheme': scheme_name,
                                        'Feature_Set': feat_name,
                                        'SMOTE': use_smote,
                                        'CV_Type': cv_name,
                                        'Model': model_name,
                                        'Metric': metric_key,
                                        'Mean': mean_score,
                                        'CI_Low': low,
                                        'CI_High': high
                                    })

                                results.append(res)

                                # Log Gap
                                gap = res['Train_Acc'] - res['Test_Acc']
                                print(f"   -> Test Acc: {res['Test_Acc']:.3f} | Train Acc: {res['Train_Acc']:.3f} | Gap: {gap:.3f}")

                            except Exception as e:
                                print(f"   [ERROR] {e}")

    # Save primary metrics
    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_FILE, index=False)

    # Save bootstrap confidence intervals
    if bootstrap_rows:
        pd.DataFrame(bootstrap_rows).to_csv(BOOTSTRAP_CI_FILE, index=False)
        print(f"[INFO] Bootstrap CIs saved to {BOOTSTRAP_CI_FILE}")

    # Counterfactual stability across seeds and labeling schemes
    stability_payload = run_counterfactual_stability(df_raw, hamd_cols)
    if stability_payload is not None:
        with open(FEATURE_STABILITY_FILE, 'w') as f:
            json.dump(stability_payload, f, indent=2)
        print(f"[INFO] Counterfactual stability saved to {FEATURE_STABILITY_FILE}")

    # SHAP & LIME comparisons on primary scheme
    pipeline, X_ref, _ = fit_reference_pipeline(df_raw, hamd_cols, scheme_name='Standard', feature_key='With_Demo', seed=RANDOM_STATE)

    shap_df = run_shap_global_importance(pipeline, X_ref, scheme_name='Standard')
    if shap_df is not None:
        shap_df.to_csv(SHAP_SUMMARY_FILE, index=False)
        print(f"[INFO] SHAP global importance saved to {SHAP_SUMMARY_FILE}")

    lime_payload = run_lime_local_explanations(pipeline, X_ref, scheme_name='Standard')
    if lime_payload is not None:
        with open(LIME_SUMMARY_FILE, 'w') as f:
            json.dump(lime_payload, f, indent=2)
        print(f"[INFO] LIME local explanations saved to {LIME_SUMMARY_FILE}")

    print("\n" + "="*40)
    print(f"Complete. Results saved to {RESULTS_FILE}")
    print(f"Total Time: {(time.time()-start_time)/60:.1f} min")

if __name__ == "__main__":
    run_analysis()
