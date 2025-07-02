import pandas as pd
import numpy as np
import logging
import argparse
import sys
import os
from datetime import datetime

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import DataLoader
from data_processor import DataProcessor
from ml_pipeline import MLPipeline
from ml_config import LOG_CONFIG, RESULTS_PATH

# Setup logging
logging.basicConfig(**LOG_CONFIG)
logger = logging.getLogger(__name__)


class WTLMLSystem:
    """Main ML system coordinator"""
    
    def __init__(self, excel_path: str = None):
        self.excel_path = excel_path
        self.data_loader = None
        self.data_processor = None
        self.ml_pipeline = None
        
        # Data storage
        self.work_hours_df = None
        self.projects_df = None
        self.financial_summary_df = None
        
    def load_and_prepare_data(self):
        """Load and prepare data for ML"""
        logger.info("Loading data for ML analysis...")
        
        # Load data
        self.data_loader = DataLoader(self.excel_path) if self.excel_path else DataLoader()
        work_hours, gs_projects, iss_projects = self.data_loader.load_all_data()
        
        self.work_hours_df = work_hours
        self.projects_df = self.data_loader.combine_projects()
        
        # Process data to get financial metrics
        self.data_processor = DataProcessor(self.work_hours_df, self.projects_df)
        self.financial_summary_df = self.data_processor.calculate_all_metrics()
        
        # Merge financial data back to projects
        self.projects_df = self.projects_df.merge(
            self.financial_summary_df[['ProjectCode', 'Profit', 'ProfitMargin', 'TotalCost', 
                                       'LaborCost', 'EfficiencyScore', 'TotalHours']],
            on='ProjectCode',
            how='left'
        )
        
        logger.info(f"Data loaded: {len(self.work_hours_df)} work records, {len(self.projects_df)} projects")
    
    def run_ml_analysis(self):
        """Run machine learning analysis"""
        logger.info("Starting ML analysis...")
        
        # Initialize ML pipeline
        self.ml_pipeline = MLPipeline(self.work_hours_df, self.projects_df)
        
        # Run full pipeline
        self.ml_pipeline.run_full_pipeline()
        
        # Print summary
        self._print_ml_summary()
    
    def _print_ml_summary(self):
        """Print ML analysis summary"""
        print("\n" + "=" * 60)
        print("WTL ML ANALYSIS COMPLETE")
        print("=" * 60)
        
        results = self.ml_pipeline.results
        
        # Feature Engineering Summary
        print("\nFeature Engineering:")
        print(f"  Total features created: {results['feature_engineering']['total_features']}")
        print(f"  Projects analyzed: {results['feature_engineering']['total_projects']}")
        
        # Risk Scoring Summary
        if 'risk_scoring' in results and results['risk_scoring']:
            risk_data = results['risk_scoring']['summary']
            print("\nRisk Scoring:")
            print(f"  High-risk projects: {risk_data['high_risk_count']}")
            print(f"  Average risk score: {risk_data['average_risk_score']:.1f}")
            print(f"  Primary risk driver: {risk_data['primary_risk_driver']}")
            
            print("\n  Risk Distribution:")
            for level, count in risk_data['risk_distribution'].items():
                print(f"    {level}: {count} projects")
        
        # Anomaly Detection Summary
        if 'anomaly_detection' in results:
            anomaly_data = results['anomaly_detection']
            print("\nAnomaly Detection:")
            print(f"  Consensus anomalies found: {anomaly_data['total_anomalies_found']}")
            
            specific = anomaly_data['specific_anomalies']
            print(f"  Cost anomalies: {specific['cost_anomalies']['count']}")
            print(f"  Efficiency anomalies: {specific['efficiency_anomalies']['count']}")
            print(f"  Pattern anomalies: {specific['pattern_anomalies']['count']}")
        
        # Insights
        if 'insights' in results:
            print("\nKey Insights:")
            for insight in results['insights'][:3]:
                print(f"  • {insight['title']}: {insight['description']}")
        
        print(f"\nResults saved to: {RESULTS_PATH}")
        print("=" * 60)
    
    def analyze_specific_project(self, project_code: str):
        """Analyze a specific project"""
        if self.ml_pipeline is None:
            raise ValueError("Must run ML analysis first")
        
        # Find project
        project_data = self.projects_df[self.projects_df['ProjectCode'] == project_code]
        
        if project_data.empty:
            print(f"Project {project_code} not found")
            return
        
        print(f"\nAnalysis for Project: {project_code}")
        print("-" * 40)
        
        # Basic info
        project = project_data.iloc[0]
        print(f"Project Name: {project['ProjectName']}")
        print(f"Type: {project['ProjectType']}")
        print(f"Status: {project.get('Status', 'Unknown')}")
        print(f"Contract Price: ¥{project['ContractPrice']:,.2f}")
        print(f"Total Cost: ¥{project.get('TotalCost', 0):,.2f}")
        print(f"Profit: ¥{project.get('Profit', 0):,.2f}")
        
        # Risk Analysis
        if self.ml_pipeline.risk_scores_df is not None:
            risk_info = self.ml_pipeline.risk_scores_df[
                self.ml_pipeline.risk_scores_df['ProjectCode'] == project_code
            ]
            
            if not risk_info.empty:
                risk = risk_info.iloc[0]
                print(f"\nRisk Analysis:")
                print(f"Total Risk Score: {risk['TotalRiskScore']:.1f}/100")
                print(f"Risk Level: {risk['RiskLevel']}")
                print(f"Risk Components:")
                print(f"  - Financial Risk: {risk['FinancialRisk']:.1f}")
                print(f"  - Efficiency Risk: {risk['EfficiencyRisk']:.1f}")
                print(f"  - Complexity Risk: {risk['ComplexityRisk']:.1f}")
                print(f"  - Timeline Risk: {risk['TimelineRisk']:.1f}")
                print(f"  - Anomaly Risk: {risk['AnomalyRisk']:.1f}")
        
        # Check if anomaly
        if hasattr(self.ml_pipeline.anomaly_detector, 'consensus_anomalies'):
            is_anomaly = project_data.index[0] in self.ml_pipeline.anomaly_detector.consensus_anomalies.index
            print(f"\nAnomaly Status: {'ANOMALY DETECTED' if is_anomaly else 'Normal'}")
        
        # Success prediction (GS only)
        if project['ProjectType'] == 'GS' and project.get('Status') not in ['Unknown', None]:
            print(f"\nActual Outcome: {project['Status']}")
    
    def generate_ml_report(self, output_file: str = None):
        """Generate comprehensive ML report"""
        if self.ml_pipeline is None:
            raise ValueError("Must run ML analysis first")
        
        if output_file is None:
            output_file = os.path.join(RESULTS_PATH, f'ml_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("WTL MACHINE LEARNING ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        results = self.ml_pipeline.results
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Projects Analyzed: {results['feature_engineering']['total_projects']}")
        report_lines.append(f"Features Engineered: {results['feature_engineering']['total_features']}")
        
        if 'success_prediction' in results and results['success_prediction']:
            report_lines.append(f"Success Prediction Accuracy: {results['success_prediction']['test_accuracy']:.1%}")
        
        if 'anomaly_detection' in results:
            report_lines.append(f"Anomalies Detected: {results['anomaly_detection']['total_anomalies_found']}")
        
        # Detailed Results
        report_lines.append("\n\nDETAILED FINDINGS")
        report_lines.append("=" * 80)
        
        # Risk Scoring Details
        if 'risk_scoring' in results:
            report_lines.append("\n\n1. RISK SCORING ANALYSIS")
            report_lines.append("-" * 40)
            
            risk_summary = results['risk_scoring']['summary']
            report_lines.append(f"\nTotal Projects Analyzed: {risk_summary['total_projects']}")
            report_lines.append(f"High-Risk Projects: {risk_summary['high_risk_count']}")
            report_lines.append(f"Average Risk Score: {risk_summary['average_risk_score']:.1f}/100")
            
            report_lines.append("\nRisk Distribution:")
            for level, count in risk_summary['risk_distribution'].items():
                report_lines.append(f"  {level}: {count} projects")
            
            report_lines.append("\nRisk Component Analysis:")
            for component, score in risk_summary['risk_components'].items():
                report_lines.append(f"  {component.title()}: {score:.1f}")
            
            # High risk projects
            if results['risk_scoring']['high_risk_projects']:
                report_lines.append("\nTop High-Risk Projects:")
                for proj in results['risk_scoring']['high_risk_projects'][:5]:
                    report_lines.append(
                        f"  - {proj['ProjectCode']}: Risk Score {proj['TotalRiskScore']:.1f} "
                        f"(Primary: {proj.get('PrimaryRiskFactor', 'Unknown')})"
                    )
            
            # Risk recommendations
            if results['risk_scoring']['recommendations']:
                report_lines.append("\nRisk-Based Recommendations:")
                for rec in results['risk_scoring']['recommendations']:
                    report_lines.append(f"\n{rec['category'].upper()} ({rec['severity']})")
                    report_lines.append(f"Finding: {rec['finding']}")
                    report_lines.append(f"Action: {rec['recommendation']}")
        
        # Anomaly Detection Details
        if 'anomaly_detection' in results:
            report_lines.append("\n\n2. ANOMALY DETECTION")
            report_lines.append("-" * 40)
            
            # Method results
            report_lines.append("\nDetection Method Results:")
            for method, stats in results['anomaly_detection']['multi_method_report']['method_results'].items():
                report_lines.append(f"  {method}: {stats['anomalies_found']} anomalies ({stats['percentage']:.1f}%)")
            
            # Specific anomalies
            specific = results['anomaly_detection']['specific_anomalies']
            report_lines.append("\nAnomaly Categories:")
            report_lines.append(f"  Cost Anomalies: {specific['cost_anomalies']['count']}")
            report_lines.append(f"  Efficiency Anomalies: {specific['efficiency_anomalies']['count']}")
            report_lines.append(f"  Pattern Anomalies: {specific['pattern_anomalies']['count']}")
            
            # Sample anomalies
            if results['anomaly_detection']['multi_method_report']['consensus_details']:
                report_lines.append("\nSample Anomalous Projects:")
                for i, anomaly in enumerate(results['anomaly_detection']['multi_method_report']['consensus_details'][:5]):
                    report_lines.append(f"  - {anomaly.get('ProjectCode', 'Unknown')}: {anomaly.get('ProjectName', 'Unknown')[:50]}...")
        
        # Insights and Recommendations
        if 'insights' in results:
            report_lines.append("\n\n3. KEY INSIGHTS AND RECOMMENDATIONS")
            report_lines.append("-" * 40)
            
            for insight in results['insights']:
                report_lines.append(f"\n{insight['title'].upper()}")
                report_lines.append(f"Priority: {insight['priority']}")
                report_lines.append(f"Description: {insight['description']}")
                if 'action' in insight:
                    report_lines.append(f"Recommended Action: {insight['action']}")
        
        # Technical Details
        report_lines.append("\n\n4. TECHNICAL DETAILS")
        report_lines.append("-" * 40)
        report_lines.append("Models Used:")
        report_lines.append("  - Risk Scoring: Multi-factor weighted scoring system")
        report_lines.append("  - Isolation Forest (Anomaly Detection)")
        report_lines.append("  - Local Outlier Factor (Anomaly Detection)")
        report_lines.append("  - One-Class SVM (Anomaly Detection)")
        
        report_lines.append("\nData Processing:")
        report_lines.append("  - Missing value imputation using Random Forest")
        report_lines.append("  - Feature scaling using StandardScaler")
        report_lines.append("  - Feature selection using mutual information")
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"ML report saved to {output_file}")
        print(f"\nReport saved to: {output_file}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='WTL ML System - Project Success Prediction & Anomaly Detection')
    parser.add_argument(
        '--excel',
        type=str,
        help='Path to Excel file (optional, uses default if not specified)'
    )
    parser.add_argument(
        '--analyze-project',
        type=str,
        help='Analyze specific project by code'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate detailed ML report'
    )
    
    args = parser.parse_args()
    
    # Initialize system
    ml_system = WTLMLSystem(excel_path=args.excel)
    
    try:
        # Load data
        ml_system.load_and_prepare_data()
        
        # Run ML analysis
        ml_system.run_ml_analysis()
        
        # Additional operations
        if args.analyze_project:
            ml_system.analyze_specific_project(args.analyze_project)
        
        if args.report:
            ml_system.generate_ml_report()
        
    except Exception as e:
        logger.error(f"Error in ML analysis: {e}")
        raise


if __name__ == "__main__":
    main()