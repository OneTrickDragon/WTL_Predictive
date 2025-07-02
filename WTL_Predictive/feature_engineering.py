import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ml_config import FEATURE_CONFIG, SUCCESS_MAPPING, ANOMALY_THRESHOLDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features for ML models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def create_features(self, work_hours_df: pd.DataFrame, 
                       projects_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set for ML"""
        logger.info("Starting feature engineering...")
        
        # Create base features
        features_df = self._create_base_features(work_hours_df, projects_df)
        
        # Add temporal features
        features_df = self._add_temporal_features(features_df, work_hours_df)
        
        # Add department interaction features
        features_df = self._add_department_features(features_df, work_hours_df)
        
        # Add efficiency features
        features_df = self._add_efficiency_features(features_df)
        
        # Add anomaly indicators
        features_df = self._add_anomaly_indicators(features_df)
        
        # Store feature names
        self.feature_names = [col for col in features_df.columns 
                             if col not in ['ProjectCode', 'ProjectType', 'Status', 'Success']]
        
        logger.info(f"Created {len(self.feature_names)} features")
        return features_df
    
    def _create_base_features(self, work_hours_df: pd.DataFrame, 
                             projects_df: pd.DataFrame) -> pd.DataFrame:
        """Create base features from project data"""
        # Aggregate work hours data
        project_features = work_hours_df.groupby('ProjectCode').agg({
            'Hours': ['sum', 'mean', 'std', 'max', 'min', 'count'],
            'Department': ['nunique', 'count'],
            'Task': 'nunique'
        })
        
        # Flatten column names
        project_features.columns = ['_'.join(col).strip() for col in project_features.columns]
        project_features = project_features.reset_index()
        
        # Merge with project information
        features_df = projects_df.merge(project_features, on='ProjectCode', how='left')
        
        # Add basic financial features
        features_df['CostRevenueRatio'] = (
            features_df['PurchaseCost'] / features_df['ContractPrice'].replace(0, 1)
        )
        features_df['RevenuePerHour'] = (
            features_df['ContractPrice'] / features_df['Hours_sum'].replace(0, 1)
        )
        features_df['CostPerHour'] = (
            features_df['PurchaseCost'] / features_df['Hours_sum'].replace(0, 1)
        )
        
        # Add project type encoding
        features_df['IsGS'] = (features_df['ProjectType'] == 'GS').astype(int)
        
        # Add success label for GS projects
        features_df['Success'] = features_df['Status'].map(SUCCESS_MAPPING)
        
        return features_df
    
    def _add_temporal_features(self, features_df: pd.DataFrame, 
                              work_hours_df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        # Convert date strings to datetime
        work_hours_df['DateStart'] = pd.to_datetime(
            work_hours_df['Date'].str.split('-').str[0], 
            format='%m/%d', 
            errors='coerce'
        )
        
        # Get project timeline features
        timeline_features = work_hours_df.groupby('ProjectCode').agg({
            'DateStart': ['min', 'max']
        })
        timeline_features.columns = ['StartDate', 'EndDate']
        timeline_features['ProjectDuration'] = (
            timeline_features['EndDate'] - timeline_features['StartDate']
        ).dt.days
        timeline_features = timeline_features.reset_index()
        
        # Calculate work intensity over time
        for window in FEATURE_CONFIG['time_windows']:
            # Hours in first N days
            window_hours = work_hours_df.groupby('ProjectCode', group_keys=False).apply(
                lambda x: x[x['DateStart'] <= x['DateStart'].min() + timedelta(days=window)]['Hours'].sum()
            )
            timeline_features[f'Hours_First_{window}_Days'] = window_hours.values
        
        # Merge timeline features
        features_df = features_df.merge(
            timeline_features[['ProjectCode', 'ProjectDuration'] + 
                            [f'Hours_First_{w}_Days' for w in FEATURE_CONFIG['time_windows']]],
            on='ProjectCode',
            how='left'
        )
        
        # Work pattern features
        features_df['WorkIntensityEarly'] = (
            features_df['Hours_First_7_Days'] / features_df['Hours_sum'].replace(0, 1)
        )
        features_df['WorkAcceleration'] = (
            features_df['Hours_First_14_Days'] - features_df['Hours_First_7_Days']
        ) / 7  # Average daily acceleration
        
        return features_df
    
    def _add_department_features(self, features_df: pd.DataFrame, 
                                work_hours_df: pd.DataFrame) -> pd.DataFrame:
        """Add department interaction features"""
        # Department participation matrix
        dept_matrix = work_hours_df.pivot_table(
            values='Hours',
            index='ProjectCode',
            columns='Department',
            aggfunc='sum',
            fill_value=0
        )
        
        # Department diversity score (entropy)
        dept_proportions = dept_matrix.div(dept_matrix.sum(axis=1), axis=0)
        dept_diversity = -1 * (dept_proportions * np.log2(dept_proportions + 1e-10)).sum(axis=1)
        
        features_df = features_df.merge(
            dept_diversity.rename('DepartmentDiversity').reset_index(),
            on='ProjectCode',
            how='left'
        )
        
        # Key department involvement
        key_departments = ['设计部', '项目管理部', '施工部', 'IT部']
        for dept in key_departments:
            if dept in dept_matrix.columns:
                dept_hours = dept_matrix[dept].reset_index()
                dept_hours.columns = ['ProjectCode', f'{dept}_Hours']
                features_df = features_df.merge(dept_hours, on='ProjectCode', how='left')
                features_df[f'{dept}_Hours'] = features_df[f'{dept}_Hours'].fillna(0)
                features_df[f'{dept}_Proportion'] = (
                    features_df[f'{dept}_Hours'] / features_df['Hours_sum'].replace(0, 1)
                )
        
        # Department collaboration score
        dept_pairs = work_hours_df.groupby(['ProjectCode', 'Date']).agg({
            'Department': 'nunique'
        }).reset_index()
        collab_score = dept_pairs.groupby('ProjectCode')['Department'].mean()
        
        features_df = features_df.merge(
            collab_score.rename('AvgDeptPerPeriod').reset_index(),
            on='ProjectCode',
            how='left'
        )
        
        return features_df
    
    def _add_efficiency_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add efficiency-related features"""
        # Task efficiency
        features_df['TasksPerHour'] = (
            features_df['Task_nunique'] / features_df['Hours_sum'].replace(0, 1)
        )
        
        # Department efficiency
        features_df['HoursPerDepartment'] = (
            features_df['Hours_sum'] / features_df['Department_nunique'].replace(0, 1)
        )
        
        # Work distribution features
        features_df['WorkConcentration'] = (
            features_df['Hours_max'] / features_df['Hours_mean'].replace(0, 1)
        )
        features_df['WorkVariability'] = (
            features_df['Hours_std'] / features_df['Hours_mean'].replace(0, 1)
        )
        
        # Cost efficiency compared to revenue
        features_df['MarginPotential'] = 1 - features_df['CostRevenueRatio']
        
        return features_df
    
    def _add_anomaly_indicators(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add features that might indicate anomalies"""
        # Extreme values indicators
        for col in ['RevenuePerHour', 'CostPerHour', 'WorkConcentration']:
            if col in features_df.columns:
                mean_val = features_df[col].mean()
                std_val = features_df[col].std()
                features_df[f'{col}_ZScore'] = (features_df[col] - mean_val) / std_val
                features_df[f'{col}_IsExtreme'] = (
                    np.abs(features_df[f'{col}_ZScore']) > ANOMALY_THRESHOLDS['efficiency_zscore']
                ).astype(int)
        
        # Unusual patterns
        features_df['UnusualCostRatio'] = (
            np.abs(features_df['CostRevenueRatio'] - features_df['CostRevenueRatio'].median()) > 
            ANOMALY_THRESHOLDS['cost_variance']
        ).astype(int)
        
        # Project complexity anomaly
        features_df['ComplexityScore'] = (
            features_df['Department_nunique'] * features_df['Task_nunique'] / 
            features_df['Hours_sum'].replace(0, 1)
        )
        
        return features_df
    
    def prepare_for_modeling(self, features_df: pd.DataFrame, 
                           target_col: str = 'Success',
                           scale_features: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for ML modeling"""
        # Remove rows with invalid target
        if target_col in features_df.columns:
            valid_mask = features_df[target_col] >= 0
            features_df = features_df[valid_mask].copy()
        
        # Select only numeric feature columns (exclude text columns)
        numeric_features = [col for col in self.feature_names 
                           if col in features_df.columns 
                           and features_df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        X = features_df[numeric_features].copy()
        
        # Handle missing values with median only for numeric columns
        numeric_medians = X.median()
        X = X.fillna(numeric_medians)
        
        # Scale features if requested
        if scale_features:
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=numeric_features, index=X.index)
        
        # Extract target
        y = features_df[target_col] if target_col in features_df.columns else None
        
        return X, y
    
    def get_feature_importance_df(self, importances: np.ndarray) -> pd.DataFrame:
        """Convert feature importances to DataFrame"""
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df


class FeatureSelector:
    """Select most relevant features for modeling"""
    
    def __init__(self, method: str = 'mutual_info'):
        self.method = method
        self.selected_features = []
        
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       n_features: int = 20) -> List[str]:
        """Select top N features"""
        from sklearn.feature_selection import mutual_info_classif, f_classif
        
        if self.method == 'mutual_info':
            scores = mutual_info_classif(X, y)
        elif self.method == 'f_score':
            scores, _ = f_classif(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Get top features
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': scores
        }).sort_values('score', ascending=False)
        
        self.selected_features = feature_scores.head(n_features)['feature'].tolist()
        
        logger.info(f"Selected {len(self.selected_features)} features using {self.method}")
        return self.selected_features


def main():
    """Test feature engineering"""
    # This would load your data and test the feature engineering
    logger.info("Feature engineering module loaded successfully")


if __name__ == "__main__":
    main()