import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import MinMaxScaler
from ml_config import ANOMALY_THRESHOLDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProjectRiskScorer:
    """Calculate multi-dimensional risk scores for projects"""
    
    def __init__(self):
        self.risk_components = {}
        self.risk_weights = {
            'financial_risk': 0.30,
            'efficiency_risk': 0.25,
            'complexity_risk': 0.20,
            'timeline_risk': 0.15,
            'anomaly_risk': 0.10
        }
        self.scaler = MinMaxScaler()
        self.use_percentile_scoring = True  # Use percentile-based scoring for better distribution
        
    def calculate_risk_scores(self, features_df: pd.DataFrame, 
                            anomaly_flags: Optional[pd.Series] = None) -> pd.DataFrame:
        """Calculate comprehensive risk scores for all projects"""
        logger.info("Calculating project risk scores...")
        
        # Calculate individual risk components
        self.risk_components['financial_risk'] = self._calculate_financial_risk(features_df)
        self.risk_components['efficiency_risk'] = self._calculate_efficiency_risk(features_df)
        self.risk_components['complexity_risk'] = self._calculate_complexity_risk(features_df)
        self.risk_components['timeline_risk'] = self._calculate_timeline_risk(features_df)
        
        # Add anomaly risk if provided
        if anomaly_flags is not None:
            self.risk_components['anomaly_risk'] = anomaly_flags.astype(float) * 100
        else:
            self.risk_components['anomaly_risk'] = 0
        
        # Calculate weighted total risk score
        total_risk = pd.Series(0, index=features_df.index)
        for component, weight in self.risk_weights.items():
            if component in self.risk_components:
                total_risk += self.risk_components[component] * weight
        
        # Create risk dataframe
        risk_df = pd.DataFrame({
            'ProjectCode': features_df['ProjectCode'],
            'TotalRiskScore': total_risk,
            'FinancialRisk': self.risk_components['financial_risk'],
            'EfficiencyRisk': self.risk_components['efficiency_risk'],
            'ComplexityRisk': self.risk_components['complexity_risk'],
            'TimelineRisk': self.risk_components['timeline_risk'],
            'AnomalyRisk': self.risk_components.get('anomaly_risk', 0)
        })
        
        # Add risk level categories
        risk_df['RiskLevel'] = pd.cut(
            risk_df['TotalRiskScore'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Critical']
        )
        
        # Add risk flags
        risk_df['IsHighRisk'] = risk_df['TotalRiskScore'] > 60
        
        logger.info(f"Calculated risk scores for {len(risk_df)} projects")
        
        return risk_df
    
    def _calculate_financial_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate financial risk component using percentile-based scoring"""
        risk = pd.Series(0, index=df.index)
        
        # Cost overrun risk (percentile-based)
        if 'CostRevenueRatio' in df.columns:
            cost_ratio = df['CostRevenueRatio'].fillna(df['CostRevenueRatio'].median())
            if self.use_percentile_scoring:
                # Higher percentile = higher risk
                percentiles = cost_ratio.rank(pct=True)
                risk += percentiles * 40  # Max 40 points from cost ratio
            else:
                # Original threshold-based approach
                risk += np.clip((cost_ratio - 0.7) * 100, 0, 40)
        
        # Negative margin risk
        if 'ProfitMargin' in df.columns:
            margin = df['ProfitMargin'].fillna(df['ProfitMargin'].median())
            if self.use_percentile_scoring:
                # Lower margin = higher percentile rank = higher risk
                margin_percentiles = (-margin).rank(pct=True)
                risk += margin_percentiles * 30  # Max 30 points
            else:
                risk += np.where(margin < 0, 40, 
                               np.where(margin < 10, 20 - margin * 2, 0))
        
        # Low revenue per hour risk
        if 'RevenuePerHour' in df.columns:
            rev_per_hour = df['RevenuePerHour'].fillna(df['RevenuePerHour'].median())
            if self.use_percentile_scoring:
                # Lower revenue = higher risk
                rev_percentiles = (-rev_per_hour).rank(pct=True)
                risk += rev_percentiles * 30  # Max 30 points
            else:
                low_revenue_threshold = rev_per_hour.quantile(0.25)
                risk += np.where(rev_per_hour < low_revenue_threshold, 20, 0)
        
        return np.clip(risk, 0, 100)
    
    def _calculate_efficiency_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate efficiency risk component using percentile-based scoring"""
        risk = pd.Series(0, index=df.index)
        
        # Low efficiency score risk
        if 'EfficiencyScore' in df.columns:
            efficiency = df['EfficiencyScore'].fillna(df['EfficiencyScore'].median())
            if self.use_percentile_scoring:
                # Lower efficiency = higher risk
                eff_percentiles = (-efficiency).rank(pct=True)
                risk += eff_percentiles * 40  # Max 40 points
            else:
                efficiency_zscore = (efficiency - efficiency.mean()) / efficiency.std()
                risk += np.where(efficiency_zscore < -1, 30, 
                               np.where(efficiency_zscore < 0, 15, 0))
        
        # High hours per task risk (inefficiency)
        if 'HoursPerTask' in df.columns:
            hours_per_task = df['HoursPerTask'].fillna(df['HoursPerTask'].median())
            if self.use_percentile_scoring:
                # More hours per task = higher risk
                task_percentiles = hours_per_task.rank(pct=True)
                risk += task_percentiles * 30  # Max 30 points
            else:
                high_hours_threshold = hours_per_task.quantile(0.75)
                risk += np.where(hours_per_task > high_hours_threshold, 20, 0)
        
        # Work variability risk
        if 'WorkVariability' in df.columns:
            variability = df['WorkVariability'].fillna(0)
            if self.use_percentile_scoring:
                var_percentiles = variability.rank(pct=True)
                risk += var_percentiles * 30  # Max 30 points
            else:
                risk += np.clip(variability * 20, 0, 30)
        
        return np.clip(risk, 0, 100)
    
    def _calculate_complexity_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate project complexity risk using percentile-based scoring"""
        risk = pd.Series(0, index=df.index)
        
        # Many departments risk
        if 'NumDepartments' in df.columns:
            num_depts = df['NumDepartments'].fillna(df['NumDepartments'].median())
            if self.use_percentile_scoring:
                dept_percentiles = num_depts.rank(pct=True)
                risk += dept_percentiles * 35  # Max 35 points
            else:
                risk += np.where(num_depts > 10, 30,
                               np.where(num_depts > 7, 20,
                                      np.where(num_depts > 5, 10, 0)))
        
        # High task count risk
        if 'NumTasks' in df.columns:
            num_tasks = df['NumTasks'].fillna(df['NumTasks'].median())
            if self.use_percentile_scoring:
                task_percentiles = num_tasks.rank(pct=True)
                risk += task_percentiles * 35  # Max 35 points
            else:
                high_task_threshold = num_tasks.quantile(0.80)
                risk += np.where(num_tasks > high_task_threshold, 25, 0)
        
        # Department diversity risk (both extremes)
        if 'DepartmentDiversity' in df.columns:
            diversity = df['DepartmentDiversity'].fillna(df['DepartmentDiversity'].median())
            if self.use_percentile_scoring:
                # Both very low and very high diversity are risky
                diversity_percentiles = diversity.rank(pct=True)
                extreme_diversity = np.abs(diversity_percentiles - 0.5) * 2  # Distance from median
                risk += extreme_diversity * 30  # Max 30 points
            else:
                risk += np.where((diversity < diversity.quantile(0.10)) | 
                               (diversity > diversity.quantile(0.90)), 25, 0)
        
        return np.clip(risk, 0, 100)
    
    def _calculate_timeline_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate timeline and work pattern risk"""
        risk = pd.Series(0, index=df.index)
        
        # Slow start risk
        if 'WorkIntensityEarly' in df.columns:
            early_intensity = df['WorkIntensityEarly'].fillna(df['WorkIntensityEarly'].median())
            risk += np.where(early_intensity < 0.1, 30,
                           np.where(early_intensity < 0.2, 15, 0))
        
        # Work acceleration risk (negative acceleration)
        if 'WorkAcceleration' in df.columns:
            acceleration = df['WorkAcceleration'].fillna(0)
            risk += np.where(acceleration < -5, 25,
                           np.where(acceleration < 0, 10, 0))
        
        # Project duration risk
        if 'ProjectDuration' in df.columns:
            duration = df['ProjectDuration'].fillna(df['ProjectDuration'].median())
            long_duration_threshold = duration.quantile(0.75)
            risk += np.where(duration > long_duration_threshold, 20, 0)
        
        # Missing timeline data risk
        if 'ProjectDuration' in df.columns:
            risk += np.where(df['ProjectDuration'].isna(), 25, 0)
        
        return np.clip(risk, 0, 100)
    
    def get_risk_summary(self, risk_df: pd.DataFrame) -> Dict:
        """Generate risk analysis summary"""
        summary = {
            'total_projects': len(risk_df),
            'risk_distribution': risk_df['RiskLevel'].value_counts().to_dict(),
            'high_risk_count': len(risk_df[risk_df['IsHighRisk']]),
            'average_risk_score': risk_df['TotalRiskScore'].mean(),
            'risk_components': {
                'financial': risk_df['FinancialRisk'].mean(),
                'efficiency': risk_df['EfficiencyRisk'].mean(),
                'complexity': risk_df['ComplexityRisk'].mean(),
                'timeline': risk_df['TimelineRisk'].mean(),
                'anomaly': risk_df['AnomalyRisk'].mean()
            }
        }
        
        # Identify top risk factors
        component_means = summary['risk_components']
        top_risk_factor = max(component_means, key=component_means.get)
        summary['primary_risk_driver'] = top_risk_factor
        
        return summary
    
    def get_high_risk_projects(self, risk_df: pd.DataFrame, 
                              project_info_df: pd.DataFrame, 
                              top_n: int = 10) -> pd.DataFrame:
        """Get detailed information about highest risk projects"""
        high_risk = risk_df.nlargest(top_n, 'TotalRiskScore')
        
        # Merge with project information
        if project_info_df is not None:
            high_risk = high_risk.merge(
                project_info_df[['ProjectCode', 'ProjectName', 'ProjectType', 
                               'ContractPrice', 'Status']],
                on='ProjectCode',
                how='left'
            )
        
        # Add risk explanation
        high_risk['PrimaryRiskFactor'] = high_risk[
            ['FinancialRisk', 'EfficiencyRisk', 'ComplexityRisk', 
             'TimelineRisk', 'AnomalyRisk']
        ].idxmax(axis=1).str.replace('Risk', '')
        
        return high_risk
    
    def generate_risk_recommendations(self, risk_df: pd.DataFrame, 
                                    features_df: pd.DataFrame) -> List[Dict]:
        """Generate actionable recommendations based on risk analysis"""
        recommendations = []
        
        # High financial risk projects
        high_financial = risk_df[risk_df['FinancialRisk'] > 60]
        if len(high_financial) > 0:
            recommendations.append({
                'category': 'Financial Risk',
                'severity': 'High',
                'finding': f'{len(high_financial)} projects with high financial risk',
                'recommendation': 'Review cost estimation and pricing strategies',
                'affected_projects': high_financial['ProjectCode'].tolist()[:5]
            })
        
        # High complexity projects
        high_complexity = risk_df[risk_df['ComplexityRisk'] > 60]
        if len(high_complexity) > 0:
            recommendations.append({
                'category': 'Complexity Risk',
                'severity': 'Medium',
                'finding': f'{len(high_complexity)} projects with high complexity',
                'recommendation': 'Consider breaking down into smaller phases or adding PM resources',
                'affected_projects': high_complexity['ProjectCode'].tolist()[:5]
            })
        
        # Timeline risk projects
        high_timeline = risk_df[risk_df['TimelineRisk'] > 60]
        if len(high_timeline) > 0:
            recommendations.append({
                'category': 'Timeline Risk',
                'severity': 'Medium',
                'finding': f'{len(high_timeline)} projects with timeline concerns',
                'recommendation': 'Implement early-stage project acceleration protocols',
                'affected_projects': high_timeline['ProjectCode'].tolist()[:5]
            })
        
        # Projects with multiple high risks
        multi_risk_mask = (
            (risk_df['FinancialRisk'] > 50) & 
            (risk_df['EfficiencyRisk'] > 50)
        ) | (
            (risk_df['ComplexityRisk'] > 50) & 
            (risk_df['TimelineRisk'] > 50)
        )
        multi_risk = risk_df[multi_risk_mask]
        
        if len(multi_risk) > 0:
            recommendations.append({
                'category': 'Multiple Risk Factors',
                'severity': 'Critical',
                'finding': f'{len(multi_risk)} projects with multiple high risk factors',
                'recommendation': 'Immediate review and intervention required',
                'affected_projects': multi_risk['ProjectCode'].tolist()[:5]
            })
        
        return recommendations


class RiskTrendAnalyzer:
    """Analyze risk trends and patterns"""
    
    def __init__(self):
        self.risk_history = []
    
    def analyze_risk_patterns(self, risk_df: pd.DataFrame, 
                            features_df: pd.DataFrame) -> Dict:
        """Analyze patterns in risk distribution"""
        analysis = {}
        
        # Risk by project type
        if 'ProjectType' in features_df.columns:
            risk_by_type = features_df.merge(
                risk_df[['ProjectCode', 'TotalRiskScore', 'RiskLevel']], 
                on='ProjectCode'
            ).groupby('ProjectType').agg({
                'TotalRiskScore': ['mean', 'std', 'min', 'max'],
                'RiskLevel': lambda x: x.value_counts().to_dict()
            })
            analysis['risk_by_project_type'] = risk_by_type
        
        # Risk by status (for GS projects)
        if 'Status' in features_df.columns:
            status_risk = features_df[features_df['Status'] != 'Unknown'].merge(
                risk_df[['ProjectCode', 'TotalRiskScore']], 
                on='ProjectCode'
            ).groupby('Status')['TotalRiskScore'].agg(['mean', 'count'])
            analysis['risk_by_status'] = status_risk
        
        # Department involvement in high-risk projects
        high_risk_projects = risk_df[risk_df['IsHighRisk']]['ProjectCode']
        analysis['high_risk_project_count'] = len(high_risk_projects)
        
        return analysis


def main():
    """Test risk scoring module"""
    logger.info("Risk scoring module loaded successfully")


if __name__ == "__main__":
    main()