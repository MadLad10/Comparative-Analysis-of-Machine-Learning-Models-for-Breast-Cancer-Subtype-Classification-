import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, GridSearchCV, 
    train_test_split, cross_validate
)
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA, SparsePCA
from sklearn.feature_selection import (
    SelectKBest, f_classif, RFE, VarianceThreshold,
    SelectPercentile, mutual_info_classif
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, f1_score, accuracy_score
)

import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

import optuna

def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(['samples', 'type'], axis=1)
    y = df['type']
    sample_ids = df['samples']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y, y_encoded, sample_ids, le

class DataPreprocessor:
    def __init__(self, variance_threshold=0.01, max_features=10000):
        self.variance_threshold = variance_threshold
        self.max_features = max_features
        self.variance_selector = None
        self.feature_selector = None
        self.scaler = None
        self.selected_features = None

    def fit_transform(self, X, y):
        self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)
        X_var = self.variance_selector.fit_transform(X)
        
        max_features = min(self.max_features, X_var.shape[1])
        self.feature_selector = SelectKBest(f_classif, k=max_features)
        X_selected = self.feature_selector.fit_transform(X_var, y)
        
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_selected)
        
        self.selected_features = self.feature_selector.get_support()
        
        return X_scaled

    def transform(self, X):
        X_var = self.variance_selector.transform(X)
        X_selected = self.feature_selector.transform(X_var)
        X_scaled = self.scaler.transform(X_selected)
        return X_scaled

class DimensionalityReducer:
    def __init__(self):
        self.reducers = {}
        self.explained_variance_ratios = {}

    def fit_transform_pca(self, X, y, methods=['pca']):
        results = {}
        
        if 'pca' in methods:
            for n_components in [50, 100, 200]:
                if n_components < min(X.shape):
                    pca = PCA(n_components=n_components, random_state=42)
                    X_pca = pca.fit_transform(X)
                    variance_explained = pca.explained_variance_ratio_.sum()
                    self.reducers[f'pca_{n_components}'] = pca
                    self.explained_variance_ratios[f'pca_{n_components}'] = variance_explained
                    results[f'pca_{n_components}'] = X_pca
        return results

class NearestShrunkenCentroids(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.overall_centroid = np.mean(X, axis=0)
        centroids = [np.mean(X[y == cls], axis=0) for cls in self.classes_]
        pooled_std = np.std(X, axis=0)
        pooled_std[pooled_std == 0] = 1
        self.centroids = []
        for c in centroids:
            diff = c - self.overall_centroid
            shrunken_diff = np.sign(diff) * np.maximum(0, np.abs(diff) - self.threshold * pooled_std)
            self.centroids.append(self.overall_centroid + shrunken_diff)
        self.centroids = np.array(self.centroids)
        return self

    def predict(self, X):
        distances = np.array([np.sum((X - c) ** 2, axis=1) for c in self.centroids]).T
        return self.classes_[np.argmin(distances, axis=1)]

class DiagonalLDA(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.means = np.array([np.mean(X[y == cls], axis=0) for cls in self.classes_])
        self.vars = np.array([np.var(X[y == cls], axis=0) + 1e-6 for cls in self.classes_])
        self.priors = np.array([np.mean(y == cls) for cls in self.classes_])
        return self

    def predict(self, X):
        log_probs = []
        for mean, var, prior in zip(self.means, self.vars, self.priors):
            diff = X - mean
            lp = np.log(prior) - 0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(diff**2 / var, axis=1)
            log_probs.append(lp)
        return self.classes_[np.argmax(np.vstack(log_probs).T, axis=1)]

class ModelEvaluator:
    def __init__(self, cv_folds=5, random_state=42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results = {}

    def evaluate_model(self, model, X, y, model_name):
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scoring = ['accuracy', 'f1_macro']
        cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring)
        self.results[model_name] = cv_results
        print(f"{model_name:30} -> Acc: {cv_results['test_accuracy'].mean():.3f} +/- {cv_results['test_accuracy'].std():.3f}")
        return cv_results

    def get_summary_table(self):
        return pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy_Mean': [self.results[m]['test_accuracy'].mean() for m in self.results],
            'Accuracy_Std': [self.results[m]['test_accuracy'].std() for m in self.results],
            'F1_Mean': [self.results[m]['test_f1_macro'].mean() for m in self.results],
            'F1_Std': [self.results[m]['test_f1_macro'].std() for m in self.results]
        }).sort_values('Accuracy_Mean', ascending=False)

def optimize_random_forest(X, y, n_trials=10):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 2, 30)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        return scores.mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return {"RandomForest": (study.best_params, study.best_value)}

def optimize_svm(X, y, n_trials=10):
    def objective(trial):
        C = trial.suggest_loguniform("C", 1e-3, 1e2)
        gamma = trial.suggest_loguniform("gamma", 1e-4, 1e0)
        clf = SVC(C=C, gamma=gamma, kernel="rbf", random_state=42)
        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        return scores.mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return {"SVM": (study.best_params, study.best_value)}

def optimize_xgboost(X, y, n_trials=15):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
            "random_state": 42
        }
        clf = xgb.XGBClassifier(**params)
        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        return scores.mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return {"XGBoost": (study.best_params, study.best_value)}

def optimize_catboost(X, y, n_trials=15):
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 50, 300),
            "depth": trial.suggest_int("depth", 2, 10),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
            "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-3, 10.0),
            "verbose": 0,
            "random_state": 42
        }
        clf = CatBoostClassifier(**params)
        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        return scores.mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return {"CatBoost": (study.best_params, study.best_value)}

def optimize_lightgbm(X, y, n_trials=15):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "verbosity": -1
        }
        clf = LGBMClassifier(**params)
        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        return scores.mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return {"LightGBM": (study.best_params, study.best_value)}

def optimize_logistic_regression(X, y, n_trials=15):
    def objective(trial):
        C = trial.suggest_loguniform("C", 1e-3, 1e2)
        penalty = trial.suggest_categorical("penalty", ["l2"])
        solver = "lbfgs"
        clf = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=2000, random_state=42)
        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        return scores.mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return {"LogisticRegression": (study.best_params, study.best_value)}

def optimize_pam(X, y, n_trials=10):
    def objective(trial):
        threshold = trial.suggest_uniform("threshold", 0.1, 2.0)
        clf = NearestShrunkenCentroids(threshold=threshold)
        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        return scores.mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return {"PAM": (study.best_params, study.best_value)}

def run_optimization(X, y):
    results = {}
    
    print("Optimizing Random Forest...")
    results.update(optimize_random_forest(X, y, n_trials=10))
    
    print("Optimizing SVM...")
    results.update(optimize_svm(X, y, n_trials=10))
    
    print("Optimizing XGBoost...")
    results.update(optimize_xgboost(X, y, n_trials=15))
    
    print("Optimizing CatBoost...")
    results.update(optimize_catboost(X, y, n_trials=15))

    print("Optimizing LightGBM...")
    results.update(optimize_lightgbm(X, y, n_trials=15))

    print("Optimizing Logistic Regression...")
    results.update(optimize_logistic_regression(X, y, n_trials=15))

    print("Optimizing PAM...")
    results.update(optimize_pam(X, y, n_trials=10))

    return results

def retrain_models(optimization_results, X, y):
    optimized_models = {}
    evaluator = ModelEvaluator()
    
    for model_name, (best_params, best_score) in optimization_results.items():
        print(f"{model_name} - Best CV Score: {best_score:.4f}")
        
        if model_name == 'XGBoost':
            model = xgb.XGBClassifier(**best_params)
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(**best_params, random_state=42)
        elif model_name == 'SVM':
            model = SVC(**best_params, random_state=42)
        elif model_name == 'CatBoost':
            model = CatBoostClassifier(**best_params)
        elif model_name == 'LightGBM':
            model = LGBMClassifier(**best_params)
        elif model_name == 'LogisticRegression':
            model = LogisticRegression(**best_params, random_state=42)
        elif model_name == 'PAM':
            model = NearestShrunkenCentroids(**best_params)
        
        cv_results = evaluator.evaluate_model(model, X, y, f"{model_name}_Optimized")
        optimized_models[model_name] = model
    
    return optimized_models, evaluator

def plot_comparison(baseline_results, optimized_results):
    baseline_df = baseline_results.get_summary_table()
    optimized_df = optimized_results.get_summary_table()
    
    optimized_df['Model'] = optimized_df['Model'].str.replace('_Optimized', '')
    
    common_models = set(baseline_df['Model']) & set(optimized_df['Model'])
    baseline_filtered = baseline_df[baseline_df['Model'].isin(common_models)]
    optimized_filtered = optimized_df[optimized_df['Model'].isin(common_models)]
    
    comparison_df = pd.merge(baseline_filtered, optimized_filtered, on='Model', suffixes=('_Baseline', '_Optimized'))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(comparison_df))
    width = 0.35
    
    ax1.bar(x - width/2, comparison_df['Accuracy_Mean_Baseline'], width, 
            label='Baseline', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, comparison_df['Accuracy_Mean_Optimized'], width,
            label='Optimized', alpha=0.8, color='orange')
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy: Baseline vs Optimized')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df['Model'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    improvement = comparison_df['Accuracy_Mean_Optimized'] - comparison_df['Accuracy_Mean_Baseline']
    colors = ['green' if imp > 0 else 'red' for imp in improvement]
    
    ax2.bar(comparison_df['Model'], improvement * 100, color=colors, alpha=0.7)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Accuracy Improvement (%)')
    ax2.set_title('Accuracy Improvement After Optimization')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return comparison_df

def run_pipeline(file_path, use_smote=True, max_features=5000):
    print("Starting Microarray Analysis Pipeline")
    print("=" * 80)

    X, y, y_encoded, sample_ids, le = load_data(file_path)
    print(f"Dataset loaded: {X.shape}")
    print(f"Target classes: {y.value_counts()}")

    preprocessor = DataPreprocessor(max_features=max_features)
    X_processed = preprocessor.fit_transform(X, y_encoded)
    print(f"After preprocessing: {X_processed.shape}")

    scaler_full = RobustScaler()
    X_scaled_full = scaler_full.fit_transform(X)
    print(f"Full data scaled: {X_scaled_full.shape}")

    if use_smote:
        print("Applying SMOTE")
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_balanced, y_balanced = smote.fit_resample(X_processed, y_encoded)
        X_full_balanced, y_full_balanced = smote.fit_resample(X_scaled_full, y_encoded)
        print(f"After SMOTE: {np.bincount(y_balanced)}")
    else:
        X_balanced, y_balanced = X_processed, y_encoded
        X_full_balanced, y_full_balanced = X_scaled_full, y_encoded

    dim_reducer = DimensionalityReducer()
    reduced_data = dim_reducer.fit_transform_pca(X_balanced, y_balanced)
    
    if 'pca_200' in reduced_data:
        X_pca = reduced_data['pca_200']
    elif 'pca_100' in reduced_data:
        X_pca = reduced_data['pca_100']
    else:
        X_pca = reduced_data['pca_50']

    print(f"PCA data shape: {X_pca.shape}")
    print(f"Statistical filtered data shape: {X_balanced.shape}")
    print(f"Full data shape: {X_full_balanced.shape}")

    print("BASELINE MODEL EVALUATION")
    print("=" * 60)
    
    baseline_evaluator = ModelEvaluator()
    
    standard_models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=False),
        'LightGBM': LGBMClassifier(random_state=42, verbosity=-1),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB()
    }
    
    print("Standard Models on PCA Data:")
    for name, model in standard_models.items():
        try:
            baseline_evaluator.evaluate_model(model, X_pca, y_balanced, name)
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
    
    print("Specialized Microarray Models:")
    
    pam_full = NearestShrunkenCentroids(threshold=0.5)
    baseline_evaluator.evaluate_model(pam_full, X_full_balanced, y_full_balanced, 'PAM_Full')
    
    pam_statistical = NearestShrunkenCentroids(threshold=0.5)
    baseline_evaluator.evaluate_model(pam_statistical, X_balanced, y_balanced, 'PAM_Statistical')
    
    pam_pca = NearestShrunkenCentroids(threshold=0.5)
    baseline_evaluator.evaluate_model(pam_pca, X_pca, y_balanced, 'PAM_PCA')
    
    dlda_full = DiagonalLDA()
    baseline_evaluator.evaluate_model(dlda_full, X_full_balanced, y_full_balanced, 'DiagonalLDA_Full')
    
    dlda_statistical = DiagonalLDA()
    baseline_evaluator.evaluate_model(dlda_statistical, X_balanced, y_balanced, 'DiagonalLDA_Statistical')
    
    dlda_pca = DiagonalLDA()
    baseline_evaluator.evaluate_model(dlda_pca, X_pca, y_balanced, 'DiagonalLDA_PCA')

    print("BASELINE RESULTS SUMMARY:")
    baseline_summary = baseline_evaluator.get_summary_table()
    print(baseline_summary)

    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    optimization_results = run_optimization(X_pca, y_balanced)
    
    print("OPTIMIZATION RESULTS:")
    for model_name, (best_params, best_score) in optimization_results.items():
        print(f"{model_name:20} -> Best CV Score: {best_score:.4f}")

    optimized_models, optimized_evaluator = retrain_models(optimization_results, X_pca, y_balanced)

    print("OPTIMIZED RESULTS SUMMARY:")
    optimized_summary = optimized_evaluator.get_summary_table()
    print(optimized_summary)

    print("CREATING COMPARISON CHARTS")
    print("=" * 60)
    
    comparison_df = plot_comparison(baseline_evaluator, optimized_evaluator)
    
    print("FINAL COMPARISON TABLE:")
    print(comparison_df[['Model', 'Accuracy_Mean_Baseline', 'Accuracy_Mean_Optimized']])
    
    best_model_idx = comparison_df['Accuracy_Mean_Optimized'].idxmax()
    best_model_name = comparison_df.loc[best_model_idx, 'Model']
    best_accuracy = comparison_df.loc[best_model_idx, 'Accuracy_Mean_Optimized']
    
    print("BEST PERFORMING MODEL:")
    print(f"Model: {best_model_name}")
    print(f"Optimized Accuracy: {best_accuracy:.4f}")
    
    return {
        'baseline_summary': baseline_summary,
        'optimized_summary': optimized_summary,
        'comparison_df': comparison_df,
        'optimization_results': optimization_results,
        'best_model': best_model_name,
        'best_accuracy': best_accuracy
    }

if __name__ == "__main__":
    file_path = r".....\Breast_GSE45827.csv"
    
    results = run_pipeline(file_path, use_smote=True, max_features=5000)
    
    print("=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    print("KEY INSIGHTS:")
    print(f"Best Model: {results['best_model']}")
    print(f"Best Accuracy: {results['best_accuracy']:.4f}")
    print(f"Total Models Evaluated: {len(results['baseline_summary']) + len(results['optimized_summary'])}")
    print("Specialized models (PAM, DiagonalLDA) tested on full, statistical, and PCA data")

    print("Memory-efficient preprocessing used (max 5000 features)")
