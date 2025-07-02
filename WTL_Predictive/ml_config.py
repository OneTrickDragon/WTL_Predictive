
MODEL_CONFIG = {
    'success_prediction': {
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': 5,
        'early_stopping_rounds': 50,
        'models': {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8
            },
            'logistic_regression': {
                'max_iter': 1000,
                'class_weight': 'balanced'
            }
        }
    },
    'anomaly_detection': {
        'contamination': 0.1,  # Expected proportion of anomalies
        'n_neighbors': 20,
        'methods': ['isolation_forest', 'local_outlier_factor', 'one_class_svm']
    }
}

# Feature engineering settings
FEATURE_CONFIG = {
    'time_windows': [7, 14, 30],  # Days for rolling features
    'aggregations': ['sum', 'mean', 'std', 'max', 'min'],
    'threshold_percentiles': [25, 50, 75, 90]
}

# Success mapping for GS projects
SUCCESS_MAPPING = {
    'Success': 1,
    'Negotiation': 0,
    'In Progress': -1,  # Exclude from training
    'Fail': 0,
    'Unknown': -1  # Exclude from training
}

# Anomaly thresholds
ANOMALY_THRESHOLDS = {
    'cost_variance': 0.3,  # 30% deviation from expected
    'efficiency_zscore': 3,  # 3 standard deviations
    'hours_per_dept_ratio': 5,  # 5x normal ratio
}

# Model evaluation metrics
EVALUATION_METRICS = [
    'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
]

# File paths
ML_DATA_PATH = 'ml_data/'
MODEL_PATH = 'models/'
RESULTS_PATH = 'ml_results/'

# Logging
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}