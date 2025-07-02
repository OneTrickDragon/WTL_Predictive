import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

def diagnose_risk_scores(risk_scorer, features_df: pd.DataFrame, risk_df: pd.DataFrame):
    """Diagnose why risk scores are low"""
    
    print("=" * 60)
    print("RISK SCORING DIAGNOSTIC REPORT")
    print("=" * 60)
    
    # 1. Overall risk distribution
    print("\n1. OVERALL RISK DISTRIBUTION:")
    print(f"   Min Risk Score: {risk_df['TotalRiskScore'].min():.2f}")
    print(f"   Max Risk Score: {risk_df['TotalRiskScore'].max():.2f}")
    print(f"   Mean Risk Score: {risk_df['TotalRiskScore'].mean():.2f}")
    print(f"   Std Risk Score: {risk_df['TotalRiskScore'].std():.2f}")
    print(f"\n   Risk Level Distribution:")
    print(risk_df['RiskLevel'].value_counts())
    
    # 2. Component analysis
    print("\n2. RISK COMPONENT ANALYSIS:")
    components = ['FinancialRisk', 'EfficiencyRisk', 'ComplexityRisk', 'TimelineRisk', 'AnomalyRisk']
    for comp in components:
        print(f"\n   {comp}:")
        print(f"      Min: {risk_df[comp].min():.2f}")
        print(f"      Max: {risk_df[comp].max():.2f}")
        print(f"      Mean: {risk_df[comp].mean():.2f}")
        print(f"      % > 50: {(risk_df[comp] > 50).sum() / len(risk_df) * 100:.1f}%")
    
    # 3. Feature analysis
    print("\n3. KEY FEATURE DISTRIBUTIONS:")
    
    # Financial features
    if 'CostRevenueRatio' in features_df.columns:
        print(f"\n   Cost-Revenue Ratio:")
        print(f"      Mean: {features_df['CostRevenueRatio'].mean():.3f}")
        print(f"      > 0.7: {(features_df['CostRevenueRatio'] > 0.7).sum()} projects")
        print(f"      > 0.9: {(features_df['CostRevenueRatio'] > 0.9).sum()} projects")
    
    if 'ProfitMargin' in features_df.columns:
        print(f"\n   Profit Margin:")
        print(f"      < 0%: {(features_df['ProfitMargin'] < 0).sum()} projects")
        print(f"      < 10%: {(features_df['ProfitMargin'] < 10).sum()} projects")
        print(f"      Mean: {features_df['ProfitMargin'].mean():.1f}%")
    
    # Efficiency features
    if 'EfficiencyScore' in features_df.columns:
        print(f"\n   Efficiency Score:")
        eff_mean = features_df['EfficiencyScore'].mean()
        eff_std = features_df['EfficiencyScore'].std()
        print(f"      Mean: {eff_mean:.2f}")
        print(f"      Std: {eff_std:.2f}")
        print(f"      Z-score < -1: {((features_df['EfficiencyScore'] - eff_mean) / eff_std < -1).sum()} projects")
    
    # Complexity features
    if 'NumDepartments' in features_df.columns:
        print(f"\n   Number of Departments:")
        print(f"      > 5: {(features_df['NumDepartments'] > 5).sum()} projects")
        print(f"      > 7: {(features_df['NumDepartments'] > 7).sum()} projects")
        print(f"      > 10: {(features_df['NumDepartments'] > 10).sum()} projects")
    
    # 4. Calibration suggestions
    print("\n4. CALIBRATION SUGGESTIONS:")
    
    # Suggest new thresholds based on percentiles
    print("\n   Suggested threshold adjustments based on your data:")
    
    if 'CostRevenueRatio' in features_df.columns:
        p75 = features_df['CostRevenueRatio'].quantile(0.75)
        p90 = features_df['CostRevenueRatio'].quantile(0.90)
        print(f"   - Cost ratio threshold: current=0.7, suggested={p75:.2f} (75th percentile)")
        print(f"   - High cost ratio: current=0.9, suggested={p90:.2f} (90th percentile)")
    
    if 'NumDepartments' in features_df.columns:
        dept_p75 = features_df['NumDepartments'].quantile(0.75)
        dept_p90 = features_df['NumDepartments'].quantile(0.90)
        print(f"   - Many departments: current=10, suggested={int(dept_p75)} (75th percentile)")
        print(f"   - Very many departments: current=10+, suggested={int(dept_p90)} (90th percentile)")
    
    # 5. Create visualization
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Risk Score Distribution Analysis', fontsize=16)
        
        # Total risk distribution
        axes[0, 0].hist(risk_df['TotalRiskScore'], bins=20, edgecolor='black')
        axes[0, 0].set_title('Total Risk Score Distribution')
        axes[0, 0].set_xlabel('Risk Score')
        axes[0, 0].set_ylabel('Count')
        
        # Risk components
        risk_components = risk_df[components].mean().sort_values(ascending=False)
        axes[0, 1].bar(range(len(risk_components)), risk_components.values)
        axes[0, 1].set_xticks(range(len(risk_components)))
        axes[0, 1].set_xticklabels(risk_components.index, rotation=45)
        axes[0, 1].set_title('Average Risk by Component')
        axes[0, 1].set_ylabel('Average Risk Score')
        
        # Risk level pie chart
        risk_counts = risk_df['RiskLevel'].value_counts()
        axes[0, 2].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
        axes[0, 2].set_title('Risk Level Distribution')
        
        # Cost-Revenue Ratio vs Risk
        if 'CostRevenueRatio' in features_df.columns:
            axes[1, 0].scatter(features_df['CostRevenueRatio'], risk_df['FinancialRisk'], alpha=0.5)
            axes[1, 0].set_xlabel('Cost-Revenue Ratio')
            axes[1, 0].set_ylabel('Financial Risk Score')
            axes[1, 0].set_title('Cost Ratio vs Financial Risk')
        
        # Efficiency Score vs Risk
        if 'EfficiencyScore' in features_df.columns:
            axes[1, 1].scatter(features_df['EfficiencyScore'], risk_df['EfficiencyRisk'], alpha=0.5)
            axes[1, 1].set_xlabel('Efficiency Score')
            axes[1, 1].set_ylabel('Efficiency Risk Score')
            axes[1, 1].set_title('Efficiency vs Risk')
        
        # Department count vs Complexity Risk
        if 'NumDepartments' in features_df.columns:
            axes[1, 2].scatter(features_df['NumDepartments'], risk_df['ComplexityRisk'], alpha=0.5)
            axes[1, 2].set_xlabel('Number of Departments')
            axes[1, 2].set_ylabel('Complexity Risk Score')
            axes[1, 2].set_title('Departments vs Complexity Risk')
        
        plt.tight_layout()
        plt.savefig('risk_diagnostic_plots.png', dpi=300, bbox_inches='tight')
        print("\n   Diagnostic plots saved to 'risk_diagnostic_plots.png'")
        
    except Exception as e:
        print(f"\n   Could not create plots: {e}")
    
    return {
        'mean_total_risk': risk_df['TotalRiskScore'].mean(),
        'max_total_risk': risk_df['TotalRiskScore'].max(),
        'risk_distribution': risk_df['RiskLevel'].value_counts().to_dict(),
        'component_means': {comp: risk_df[comp].mean() for comp in components}
    }


def create_calibrated_risk_scorer(features_df: pd.DataFrame) -> Dict:
    """Create calibrated thresholds based on actual data distribution"""
    
    calibrated_thresholds = {}
    
    # Financial thresholds
    if 'CostRevenueRatio' in features_df.columns:
        calibrated_thresholds['cost_ratio_warning'] = features_df['CostRevenueRatio'].quantile(0.70)
        calibrated_thresholds['cost_ratio_critical'] = features_df['CostRevenueRatio'].quantile(0.85)
    
    if 'ProfitMargin' in features_df.columns:
        calibrated_thresholds['low_margin_threshold'] = features_df['ProfitMargin'].quantile(0.25)
        calibrated_thresholds['very_low_margin_threshold'] = features_df['ProfitMargin'].quantile(0.10)
    
    # Efficiency thresholds
    if 'EfficiencyScore' in features_df.columns:
        calibrated_thresholds['low_efficiency_threshold'] = features_df['EfficiencyScore'].quantile(0.25)
        calibrated_thresholds['very_low_efficiency_threshold'] = features_df['EfficiencyScore'].quantile(0.10)
    
    # Complexity thresholds
    if 'NumDepartments' in features_df.columns:
        calibrated_thresholds['high_dept_count'] = int(features_df['NumDepartments'].quantile(0.75))
        calibrated_thresholds['very_high_dept_count'] = int(features_df['NumDepartments'].quantile(0.90))
    
    if 'NumTasks' in features_df.columns:
        calibrated_thresholds['high_task_count'] = features_df['NumTasks'].quantile(0.80)
    
    # Timeline thresholds
    if 'WorkIntensityEarly' in features_df.columns:
        calibrated_thresholds['low_early_intensity'] = features_df['WorkIntensityEarly'].quantile(0.20)
        calibrated_thresholds['very_low_early_intensity'] = features_df['WorkIntensityEarly'].quantile(0.10)
    
    print("\nCALIBRATED THRESHOLDS:")
    for key, value in calibrated_thresholds.items():
        print(f"   {key}: {value:.3f}")
    
    return calibrated_thresholds


if __name__ == "__main__":
    print("Risk diagnostic tool loaded. Use diagnose_risk_scores() to analyze your risk distribution.")