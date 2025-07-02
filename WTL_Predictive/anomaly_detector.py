import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import joblib
import os

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ml_config import MODEL_CONFIG, ANOMALY_THRESHOLDS, MODEL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect anomalous projects using multiple methods"""
    
    def __init__(self, method: str = 'isolation_forest'):
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.anomaly_scores = None
        self.anomalies = None
        self.contamination = MODEL_CONFIG['anomaly_detection']['contamination']
        
        # Create model directory
        os.makedirs(MODEL_PATH, exist_ok=True)
        
    def fit(self, X: pd.DataFrame) -> 'AnomalyDetector':
        """Fit anomaly detection model"""
        logger.info(f"Fitting {self.method} anomaly detector...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and fit model
        self.model = self._create_model()
        
        if self.method == 'local_outlier_factor':
            # LOF requires predict on same data
            self.anomaly_scores = self.model.fit_predict(X_scaled)
        else:
            self.model.fit(X_scaled)
            self.anomaly_scores = self.model.decision_function(X_scaled)
        
        # Identify anomalies
        self.anomalies = self._identify_anomalies(X)
        
        logger.info(f"Found {len(self.anomalies)} anomalies out of {len(X)} projects")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomalies on new data"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        
        if self.method == 'local_outlier_factor':
            # LOF doesn't support predict on new data by default
            logger.warning("LOF doesn't support prediction on new data. Returning fitted anomalies.")
            return self.anomaly_scores[:len(X)]
        else:
            predictions = self.model.predict(X_scaled)
            return predictions
    
    def get_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores for data"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        
        if self.method == 'local_outlier_factor':
            return self.anomaly_scores[:len(X)]
        else:
            return self.model.decision_function(X_scaled)
    
    def _create_model(self):
        """Create anomaly detection model"""
        if self.method == 'isolation_forest':
            return IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        elif self.method == 'local_outlier_factor':
            return LocalOutlierFactor(
                n_neighbors=MODEL_CONFIG['anomaly_detection']['n_neighbors'],
                contamination=self.contamination
            )
        elif self.method == 'one_class_svm':
            return OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='auto'
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _identify_anomalies(self, X: pd.DataFrame) -> pd.DataFrame:
        """Identify and characterize anomalies"""
        if self.method == 'local_outlier_factor':
            anomaly_mask = self.anomaly_scores == -1
        else:
            # For other methods, use threshold
            threshold = np.percentile(self.anomaly_scores, self.contamination * 100)
            anomaly_mask = self.anomaly_scores < threshold
        
        anomalies = X[anomaly_mask].copy()
        anomalies['anomaly_score'] = self.anomaly_scores[anomaly_mask]
        
        return anomalies
    
    def explain_anomalies(self, X: pd.DataFrame, feature_names: List[str], 
                         top_n: int = 5) -> pd.DataFrame:
        """Explain why projects are anomalous"""
        explanations = []
        
        # Get mean and std for each feature
        feature_stats = pd.DataFrame({
            'mean': X.mean(),
            'std': X.std()
        })
        
        # For each anomaly, find most unusual features
        for idx in self.anomalies.index:
            project_data = X.loc[idx]
            
            # Calculate z-scores
            z_scores = (project_data - feature_stats['mean']) / feature_stats['std']
            z_scores_abs = z_scores.abs().sort_values(ascending=False)
            
            # Get top unusual features
            unusual_features = z_scores_abs.head(top_n)
            
            explanation = {
                'project_index': idx,
                'anomaly_score': self.anomalies.loc[idx, 'anomaly_score'],
                'unusual_features': unusual_features.to_dict(),
                'most_unusual': unusual_features.index[0],
                'deviation': z_scores[unusual_features.index[0]]
            }
            
            explanations.append(explanation)
        
        return pd.DataFrame(explanations)


class MultiMethodAnomalyDetector:
    """Combine multiple anomaly detection methods"""
    
    def __init__(self):
        self.detectors = {
            method: AnomalyDetector(method)
            for method in MODEL_CONFIG['anomaly_detection']['methods']
        }
        self.consensus_anomalies = None
        
    def fit(self, X: pd.DataFrame) -> 'MultiMethodAnomalyDetector':
        """Fit all anomaly detectors"""
        logger.info("Fitting multiple anomaly detection methods...")
        
        # Fit each detector
        for method, detector in self.detectors.items():
            detector.fit(X)
        
        # Find consensus anomalies
        self._find_consensus_anomalies(X)
        
        return self
    
    def _find_consensus_anomalies(self, X: pd.DataFrame):
        """Find projects flagged by multiple methods"""
        anomaly_counts = pd.Series(index=X.index, data=0)
        
        for method, detector in self.detectors.items():
            if method == 'local_outlier_factor':
                anomaly_mask = detector.anomaly_scores == -1
            else:
                predictions = detector.predict(X)
                anomaly_mask = predictions == -1
            
            anomaly_counts[anomaly_mask] += 1
        
        # Projects flagged by at least 2 methods
        consensus_threshold = len(self.detectors) // 2 + 1
        self.consensus_anomalies = X[anomaly_counts >= consensus_threshold].copy()
        self.consensus_anomalies['detection_count'] = anomaly_counts[anomaly_counts >= consensus_threshold]
        
        logger.info(f"Consensus anomalies: {len(self.consensus_anomalies)} projects")
    
    def get_anomaly_report(self, X: pd.DataFrame, 
                          project_info: Optional[pd.DataFrame] = None) -> Dict:
        """Generate comprehensive anomaly report"""
        report = {
            'summary': {
                'total_projects': len(X),
                'methods_used': list(self.detectors.keys()),
                'consensus_anomalies': len(self.consensus_anomalies)
            },
            'method_results': {},
            'consensus_details': []
        }
        
        # Results by method
        for method, detector in self.detectors.items():
            report['method_results'][method] = {
                'anomalies_found': len(detector.anomalies),
                'percentage': len(detector.anomalies) / len(X) * 100
            }
        
        # Consensus anomaly details
        if project_info is not None and not self.consensus_anomalies.empty:
            # Merge with project information
            anomaly_details = self.consensus_anomalies.merge(
                project_info[['ProjectCode', 'ProjectName', 'Status', 'ContractPrice']],
                left_index=True,
                right_index=True,
                how='left'
            )
            
            report['consensus_details'] = anomaly_details.to_dict('records')
        
        return report


class ProjectAnomalyAnalyzer:
    """Analyze specific types of project anomalies"""
    
    def __init__(self):
        self.cost_anomalies = None
        self.efficiency_anomalies = None
        self.pattern_anomalies = None
        
    def analyze_cost_anomalies(self, projects_df: pd.DataFrame) -> pd.DataFrame:
        """Identify cost-related anomalies"""
        cost_anomalies = []
        
        # Unusual cost-to-revenue ratio
        cost_ratio = projects_df['PurchaseCost'] / projects_df['ContractPrice'].replace(0, 1)
        median_ratio = cost_ratio.median()
        unusual_ratio = np.abs(cost_ratio - median_ratio) > ANOMALY_THRESHOLDS['cost_variance']
        
        # Negative profit margins
        negative_margin = projects_df['Profit'] < 0
        
        # Combine conditions
        anomaly_mask = unusual_ratio | negative_margin
        
        if anomaly_mask.any():
            anomalies = projects_df[anomaly_mask].copy()
            anomalies['anomaly_type'] = 'cost'
            anomalies['cost_ratio_deviation'] = cost_ratio[anomaly_mask] - median_ratio
            cost_anomalies.append(anomalies)
        
        self.cost_anomalies = pd.concat(cost_anomalies) if cost_anomalies else pd.DataFrame()
        return self.cost_anomalies
    
    def analyze_efficiency_anomalies(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Identify efficiency-related anomalies"""
        efficiency_anomalies = []
        
        # Extremely low hours per task
        if 'TasksPerHour' in features_df.columns:
            tasks_per_hour_zscore = (
                (features_df['TasksPerHour'] - features_df['TasksPerHour'].mean()) /
                features_df['TasksPerHour'].std()
            )
            unusual_efficiency = np.abs(tasks_per_hour_zscore) > ANOMALY_THRESHOLDS['efficiency_zscore']
            
            if unusual_efficiency.any():
                anomalies = features_df[unusual_efficiency].copy()
                anomalies['anomaly_type'] = 'efficiency'
                anomalies['efficiency_zscore'] = tasks_per_hour_zscore[unusual_efficiency]
                efficiency_anomalies.append(anomalies)
        
        self.efficiency_anomalies = pd.concat(efficiency_anomalies) if efficiency_anomalies else pd.DataFrame()
        return self.efficiency_anomalies
    
    def analyze_pattern_anomalies(self, work_hours_df: pd.DataFrame) -> pd.DataFrame:
        """Identify unusual work patterns"""
        pattern_anomalies = []
        
        # Projects with unusual department combinations
        dept_patterns = work_hours_df.groupby('ProjectCode')['Department'].apply(
            lambda x: tuple(sorted(x.unique()))
        )
        
        # Find rare patterns
        pattern_counts = dept_patterns.value_counts()
        rare_patterns = pattern_counts[pattern_counts == 1].index
        
        rare_pattern_projects = dept_patterns[dept_patterns.isin(rare_patterns)]
        
        if not rare_pattern_projects.empty:
            anomalies = pd.DataFrame({
                'ProjectCode': rare_pattern_projects.index,
                'anomaly_type': 'pattern',
                'department_pattern': rare_pattern_projects.values
            })
            pattern_anomalies.append(anomalies)
        
        self.pattern_anomalies = pd.concat(pattern_anomalies) if pattern_anomalies else pd.DataFrame()
        return self.pattern_anomalies
    
    def get_comprehensive_anomaly_report(self) -> Dict:
        """Get report of all anomaly types"""
        report = {
            'cost_anomalies': {
                'count': len(self.cost_anomalies) if self.cost_anomalies is not None else 0,
                'details': self.cost_anomalies.to_dict('records') if self.cost_anomalies is not None else []
            },
            'efficiency_anomalies': {
                'count': len(self.efficiency_anomalies) if self.efficiency_anomalies is not None else 0,
                'details': self.efficiency_anomalies.to_dict('records') if self.efficiency_anomalies is not None else []
            },
            'pattern_anomalies': {
                'count': len(self.pattern_anomalies) if self.pattern_anomalies is not None else 0,
                'details': self.pattern_anomalies.to_dict('records') if self.pattern_anomalies is not None else []
            }
        }
        
        return report


def main():
    """Test anomaly detection module"""
    logger.info("Anomaly detection module loaded successfully")


if __name__ == "__main__":
    main()