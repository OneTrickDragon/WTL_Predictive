"""
ML Pipeline Orchestrator for WTL ML System
Coordinates feature engineering, model training, and predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

from ml_config import MODEL_CONFIG, RESULTS_PATH, MODEL_PATH
from feature_engineering import FeatureEngineer
from anomaly_detector import AnomalyDetector, MultiMethodAnomalyDetector, ProjectAnomalyAnalyzer
from risk_scorer import ProjectRiskScorer, RiskTrendAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPipeline:
    """Main ML pipeline orchestrator"""
    
    def __init__(self, work_hours_df: pd.DataFrame, projects_df: pd.DataFrame):
        self.work_hours_df = work_hours_df
        self.projects_df = projects_df
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.anomaly_detector = None
        self.risk_scorer = ProjectRiskScorer()
        
        # Data storage
        self.features_df = None
        self.risk_scores_df = None
        
        # Results
        self.results = {
            'feature_engineering': {},
            'anomaly_detection': {},
            'risk_scoring': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Create directories
        os.makedirs(RESULTS_PATH, exist_ok=True)
        os.makedirs(MODEL_PATH, exist_ok=True)
    
    def run_full_pipeline(self):
        """Run complete ML pipeline"""
        logger.info("Starting ML pipeline...")
        
        # Step 1: Feature Engineering
        self._engineer_features()
        
        # Step 2: Anomaly Detection
        self._detect_anomalies()
        
        # Step 3: Risk Scoring
        self._calculate_risk_scores()
        
        # Step 4: Generate insights
        self._generate_insights()
        
        # Step 5: Save results
        self._save_results()
        
        logger.info("ML pipeline completed successfully!")
    
    def _engineer_features(self):
        """Engineer features for ML"""
        logger.info("Engineering features...")
        
        # Create features
        self.features_df = self.feature_engineer.create_features(
            self.work_hours_df, 
            self.projects_df
        )
        
        # Log feature statistics
        self.results['feature_engineering'] = {
            'total_features': len(self.feature_engineer.feature_names),
            'total_projects': len(self.features_df),
            'feature_names': self.feature_engineer.feature_names[:20]  # Top 20
        }
        
        logger.info(f"Created {len(self.feature_engineer.feature_names)} features")
    
    def _detect_anomalies(self):
        """Detect anomalies in projects"""
        logger.info("Detecting anomalies...")
        
        # Prepare data for anomaly detection
        X_anomaly, _ = self.feature_engineer.prepare_for_modeling(
            self.features_df, 
            target_col=None,
            scale_features=True
        )
        
        # Multi-method anomaly detection
        multi_detector = MultiMethodAnomalyDetector()
        multi_detector.fit(X_anomaly)
        
        # Get anomaly report
        anomaly_report = multi_detector.get_anomaly_report(
            X_anomaly,
            self.features_df[['ProjectCode', 'ProjectName', 'Status', 'ContractPrice']]
        )
        
        # Specific anomaly analysis
        anomaly_analyzer = ProjectAnomalyAnalyzer()
        anomaly_analyzer.analyze_cost_anomalies(self.features_df)
        anomaly_analyzer.analyze_efficiency_anomalies(self.features_df)
        anomaly_analyzer.analyze_pattern_anomalies(self.work_hours_df)
        
        specific_anomalies = anomaly_analyzer.get_comprehensive_anomaly_report()
        
        # Store results
        self.results['anomaly_detection'] = {
            'multi_method_report': anomaly_report,
            'specific_anomalies': specific_anomalies,
            'total_anomalies_found': len(multi_detector.consensus_anomalies)
        }
        
        self.anomaly_detector = multi_detector
        
        logger.info(f"Found {len(multi_detector.consensus_anomalies)} consensus anomalies")
    
    def _calculate_risk_scores(self):
        """Calculate risk scores for all projects"""
        logger.info("Calculating risk scores...")
        
        # Get anomaly flags
        anomaly_flags = None
        if self.anomaly_detector and hasattr(self.anomaly_detector, 'consensus_anomalies'):
            anomaly_flags = self.features_df.index.isin(
                self.anomaly_detector.consensus_anomalies.index
            )
        
        # Calculate risk scores
        self.risk_scores_df = self.risk_scorer.calculate_risk_scores(
            self.features_df,
            anomaly_flags
        )
        
        # Get risk summary
        risk_summary = self.risk_scorer.get_risk_summary(self.risk_scores_df)
        
        # Get high risk projects
        high_risk_projects = self.risk_scorer.get_high_risk_projects(
            self.risk_scores_df,
            self.features_df,
            top_n=15
        )
        
        # Generate risk recommendations
        risk_recommendations = self.risk_scorer.generate_risk_recommendations(
            self.risk_scores_df,
            self.features_df
        )
        
        # Analyze risk patterns
        risk_analyzer = RiskTrendAnalyzer()
        risk_patterns = risk_analyzer.analyze_risk_patterns(
            self.risk_scores_df,
            self.features_df
        )
        
        # Store results
        self.results['risk_scoring'] = {
            'summary': risk_summary,
            'high_risk_projects': high_risk_projects.to_dict('records'),
            'recommendations': risk_recommendations,
            'risk_patterns': {
                'high_risk_count': risk_patterns.get('high_risk_project_count', 0)
            }
        }
        
        logger.info(f"Risk scoring complete. {risk_summary['high_risk_count']} high-risk projects identified")
    
    def _generate_insights(self):
        """Generate actionable insights from ML results"""
        insights = []
        
        # Risk scoring insights
        if 'risk_scoring' in self.results:
            risk_data = self.results['risk_scoring']
            
            # High risk projects
            high_risk_count = risk_data['summary']['high_risk_count']
            if high_risk_count > 0:
                primary_driver = risk_data['summary']['primary_risk_driver']
                insights.append({
                    'type': 'risk_alert',
                    'title': 'High Risk Projects Detected',
                    'description': f'{high_risk_count} projects identified as high risk. Primary driver: {primary_driver}',
                    'priority': 'critical',
                    'action': 'Immediate review and intervention recommended'
                })
            
            # Risk recommendations
            for rec in risk_data['recommendations'][:3]:  # Top 3 recommendations
                insights.append({
                    'type': 'risk_recommendation',
                    'title': rec['category'],
                    'description': rec['finding'],
                    'priority': rec['severity'].lower(),
                    'action': rec['recommendation']
                })
        
        # Anomaly insights
        if 'anomaly_detection' in self.results:
            anomaly_data = self.results['anomaly_detection']
            
            # Cost anomalies
            cost_anomalies = anomaly_data['specific_anomalies']['cost_anomalies']['count']
            if cost_anomalies > 0:
                insights.append({
                    'type': 'cost_anomaly',
                    'title': 'Cost Overrun Projects',
                    'description': f'{cost_anomalies} projects show unusual cost patterns',
                    'priority': 'high',
                    'action': 'Review cost estimation and control processes'
                })
            
            # Consensus anomalies
            consensus_count = anomaly_data.get('total_anomalies_found', 0)
            if consensus_count > 0:
                insights.append({
                    'type': 'general_anomaly',
                    'title': 'Unusual Projects Detected',
                    'description': f'{consensus_count} projects flagged by multiple anomaly detection methods',
                    'priority': 'medium',
                    'action': 'Investigate these projects for data quality or operational issues'
                })
        
        self.results['insights'] = insights
        logger.info(f"Generated {len(insights)} insights")
    
    def _save_results(self):
        """Save all results and models"""
        # Save results JSON
        results_file = os.path.join(RESULTS_PATH, f'ml_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        
        # Save predictions on full dataset
        if self.features_df is not None:
            predictions_file = os.path.join(RESULTS_PATH, 'project_analysis.csv')
            analysis_df = self.features_df[['ProjectCode', 'ProjectName', 'ProjectType', 'Status']].copy()
            
            # Add risk scores
            if self.risk_scores_df is not None:
                analysis_df = analysis_df.merge(
                    self.risk_scores_df[['ProjectCode', 'TotalRiskScore', 'RiskLevel', 
                                       'FinancialRisk', 'EfficiencyRisk', 'ComplexityRisk',
                                       'TimelineRisk', 'IsHighRisk']],
                    on='ProjectCode',
                    how='left'
                )
            
            # Add anomaly flags
            if self.anomaly_detector and hasattr(self.anomaly_detector, 'consensus_anomalies'):
                analysis_df['IsAnomaly'] = analysis_df.index.isin(
                    self.anomaly_detector.consensus_anomalies.index
                )
            
            analysis_df.to_csv(predictions_file, index=False)
            logger.info(f"Project analysis saved to {predictions_file}")
    
    def check_project_anomaly(self, project_features: pd.DataFrame) -> Dict:
        """Check if project is anomalous"""
        if self.anomaly_detector is None:
            raise ValueError("Anomaly detector not trained yet")
        
        # Prepare features
        X_new, _ = self.feature_engineer.prepare_for_modeling(
            project_features,
            target_col=None,
            scale_features=True
        )
        
        # Check with each detector
        anomaly_flags = {}
        for method, detector in self.anomaly_detector.detectors.items():
            if method != 'local_outlier_factor':  # LOF doesn't support new predictions
                is_anomaly = detector.predict(X_new)[0] == -1
                anomaly_flags[method] = bool(is_anomaly)
        
        # Consensus
        anomaly_count = sum(anomaly_flags.values())
        is_anomaly = anomaly_count >= len(anomaly_flags) // 2
        
        return {
            'is_anomaly': is_anomaly,
            'detection_methods': anomaly_flags,
            'consensus_score': anomaly_count / len(anomaly_flags)
        }
    def run_risk_diagnostics(self):
        """Run diagnostics on risk scoring to understand distribution"""
        if self.risk_scores_df is None or self.features_df is None:
            logger.warning("Must run full pipeline before diagnostics")
            return
        
        try:
            from risk_diagnostic import diagnose_risk_scores, create_calibrated_risk_scorer
            
            logger.info("Running risk scoring diagnostics...")
            
            # Run diagnostic
            diagnostic_results = diagnose_risk_scores(
                self.risk_scorer,
                self.features_df,
                self.risk_scores_df
            )
            
            # Get calibrated thresholds
            calibrated_thresholds = create_calibrated_risk_scorer(self.features_df)
            
            return diagnostic_results
            
        except ImportError:
            logger.warning("Risk diagnostic module not found. Create risk_diagnostic.py to use this feature.")
            return None


def main():
    """Test ML pipeline"""
    logger.info("ML Pipeline module loaded successfully")


if __name__ == "__main__":
    main()