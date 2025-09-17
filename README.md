# Microarray Analysis Pipeline

A comprehensive machine learning pipeline for microarray gene expression data classification with specialized preprocessing, dimensionality reduction, and model optimization.

## Overview

This pipeline is designed specifically for microarray data analysis, implementing both standard machine learning approaches and specialized bioinformatics methods like Prediction Analysis of Microarrays (PAM) and Diagonal Linear Discriminant Analysis (DiagonalLDA).

## Features

- **Specialized Preprocessing**: Variance thresholding, statistical feature selection, and robust scaling
- **Dimensionality Reduction**: PCA with multiple component options
- **Class Balancing**: SMOTE oversampling for imbalanced datasets
- **Multiple Model Types**: 
  - Standard ML models (Random Forest, SVM, XGBoost, etc.)
  - Specialized microarray methods (PAM, DiagonalLDA)
- **Hyperparameter Optimization**: Optuna-based automated tuning
- **Comprehensive Evaluation**: Cross-validation with multiple metrics
- **Visualization**: Performance comparison charts

## Requirements

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
catboost
lightgbm
imbalanced-learn
optuna
```

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost lightgbm imbalanced-learn optuna
```

## Data Format

The pipeline expects a CSV file with the following structure:
- `samples`: Sample identifiers
- `type`: Target class labels
- Remaining columns: Gene expression features

Example:
```
samples,type,gene1,gene2,gene3,...
sample1,cancer,0.5,1.2,0.8,...
sample2,normal,0.3,0.9,1.1,...
```

## Usage

### Basic Usage

```python
from project443 import run_pipeline

# Run the complete pipeline
results = run_pipeline("path/to/your/data.csv")
```

### Advanced Usage

```python
# Customize preprocessing and SMOTE settings
results = run_pipeline(
    file_path="data.csv",
    use_smote=True,        # Apply SMOTE for class balancing
    max_features=5000      # Maximum features after selection
)
```

### Manual Pipeline Steps

```python
from project443 import *

# Load and preprocess data
X, y, y_encoded, sample_ids, le = load_data("data.csv")

# Preprocess features
preprocessor = DataPreprocessor(max_features=5000)
X_processed = preprocessor.fit_transform(X, y_encoded)

# Apply dimensionality reduction
dim_reducer = DimensionalityReducer()
reduced_data = dim_reducer.fit_transform_pca(X_processed, y_encoded)

# Evaluate models
evaluator = ModelEvaluator()
model = RandomForestClassifier(random_state=42)
results = evaluator.evaluate_model(model, X_processed, y_encoded, "RF")
```

## Pipeline Components

### 1. Data Preprocessing (`DataPreprocessor`)
- **Variance Thresholding**: Removes low-variance features
- **Statistical Feature Selection**: SelectKBest with f_classif
- **Robust Scaling**: Handles outliers better than standard scaling

### 2. Dimensionality Reduction (`DimensionalityReducer`)
- **PCA**: Principal Component Analysis with 50, 100, or 200 components
- **Variance Explanation**: Tracks explained variance ratios

### 3. Specialized Classifiers

#### Nearest Shrunken Centroids (PAM)
- Implements Prediction Analysis of Microarrays
- Shrinks class centroids toward overall centroid
- Performs automatic feature selection

#### Diagonal Linear Discriminant Analysis
- Assumes diagonal covariance matrices
- Suitable for high-dimensional, small-sample data
- Computationally efficient

### 4. Model Optimization
- **Optuna Integration**: Bayesian optimization for hyperparameters
- **Supported Models**: Random Forest, SVM, XGBoost, CatBoost, LightGBM, Logistic Regression, PAM
- **Cross-Validation**: Stratified k-fold validation

### 5. Evaluation and Visualization
- **Metrics**: Accuracy, F1-score (macro)
- **Comparison Charts**: Baseline vs optimized performance
- **Summary Tables**: Comprehensive results overview

## Model Performance Strategy

The pipeline tests models on three different data representations:

1. **Full Data**: All features with robust scaling
2. **Statistical Filtered**: Top features selected by statistical tests
3. **PCA Reduced**: Principal components (50-200 dimensions)

This multi-representation approach helps identify the optimal feature space for each model type.

## Output

The pipeline returns a dictionary containing:

```python
{
    'baseline_summary': DataFrame,      # Baseline model results
    'optimized_summary': DataFrame,     # Optimized model results  
    'comparison_df': DataFrame,         # Performance comparison
    'optimization_results': dict,       # Best hyperparameters
    'best_model': str,                 # Best performing model name
    'best_accuracy': float             # Best accuracy achieved
}
```

## Example Output

```
Starting Microarray Analysis Pipeline
================================================================================
Dataset loaded: (100, 20000)
Target classes: cancer    60
                normal    40
After preprocessing: (100, 5000)
...

BEST PERFORMING MODEL:
Model: XGBoost
Optimized Accuracy: 0.8750
```

## Performance Considerations

- **Memory Efficient**: Limits features to prevent memory issues
- **Parallel Processing**: Uses all available CPU cores where possible
- **Progress Tracking**: Shows optimization progress with progress bars
- **Error Handling**: Gracefully handles model failures

## Customization

### Adding New Models

```python
def optimize_new_model(X, y, n_trials=10):
    def objective(trial):
        # Define hyperparameter search space
        param = trial.suggest_float("param", 0.1, 1.0)
        clf = YourModel(param=param)
        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        return scores.mean()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return {"YourModel": (study.best_params, study.best_value)}
```

### Custom Preprocessing

```python
class CustomPreprocessor(DataPreprocessor):
    def __init__(self, custom_param=0.5):
        super().__init__()
        self.custom_param = custom_param
    
    def fit_transform(self, X, y):
        # Add custom preprocessing steps
        X_custom = your_custom_function(X, self.custom_param)
        return super().fit_transform(X_custom, y)
```

## Best Practices

1. **Feature Selection**: Start with statistical filtering before PCA
2. **Class Balancing**: Use SMOTE for imbalanced datasets
3. **Validation**: Always use stratified cross-validation
4. **Scaling**: RobustScaler works better than StandardScaler for gene expression
5. **Optimization**: Increase n_trials for better hyperparameter search

## Troubleshooting

### Common Issues

**Memory Error**: Reduce `max_features` parameter
```python
results = run_pipeline(file_path, max_features=1000)
```

**SMOTE Error**: Reduce k_neighbors if classes are too small
```python
# In the code, modify SMOTE initialization:
smote = SMOTE(random_state=42, k_neighbors=1)
```

**Convergence Issues**: Increase max_iter for iterative algorithms
```python
LogisticRegression(max_iter=2000)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Citation

If you use this pipeline in your research, please cite:

```
Microarray Analysis Pipeline
https://github.com/yourusername/microarray-pipeline
```

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@domain.com].
