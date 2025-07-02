import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from config import DEPARTMENT_SALARIES, WORK_HOURS_PER_YEAR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and calculate financial metrics"""
    
    def __init__(self, work_hours_df: pd.DataFrame, projects_df: pd.DataFrame):
        self.work_hours_df = work_hours_df
        self.projects_df = projects_df
        self.financial_summary_df = None
        self.department_summary_df = None
        
    def calculate_all_metrics(self) -> pd.DataFrame:
        """Calculate all financial metrics"""
        logger.info("Calculating financial metrics...")
        
        # Impute missing values before calculations
        self._impute_missing_values()
        
        # Calculate project-level metrics
        self.financial_summary_df = self._calculate_project_metrics()
        
        # Calculate department-level metrics
        self.department_summary_df = self._calculate_department_metrics()
        
        logger.info("All metrics calculated successfully")
        return self.financial_summary_df
    
    def _impute_missing_values(self):
        """Impute missing contract prices and purchase costs using domain-specific approach"""
        logger.info("Imputing missing values...")
        
        # First, get project features from work hours data
        project_features = self.work_hours_df.groupby('ProjectCode').agg({
            'Hours': 'sum',
            'Department': 'nunique',
            'Task': 'count'
        }).rename(columns={
            'Hours': 'TotalHours',
            'Department': 'NumDepartments',
            'Task': 'NumTasks'
        })
        
        # Merge with projects data
        projects_with_features = self.projects_df.merge(
            project_features,
            left_on='ProjectCode',
            right_index=True,
            how='left'
        )
        
        # Fill missing features with 0
        projects_with_features['TotalHours'] = projects_with_features['TotalHours'].fillna(0)
        projects_with_features['NumDepartments'] = projects_with_features['NumDepartments'].fillna(0)
        projects_with_features['NumTasks'] = projects_with_features['NumTasks'].fillna(0)
        
        # Add project type as numeric feature
        projects_with_features['IsGS'] = (projects_with_features['ProjectType'] == 'GS').astype(int)
        
        # Separate projects with and without missing values
        complete_mask = (projects_with_features['ContractPrice'] > 0) & (projects_with_features['PurchaseCost'] > 0)
        complete_projects = projects_with_features[complete_mask].copy()
        incomplete_projects = projects_with_features[~complete_mask].copy()
        
        if len(complete_projects) > 10 and len(incomplete_projects) > 0:
            # Use machine learning to predict missing values
            features = ['TotalHours', 'NumDepartments', 'NumTasks', 'IsGS']
            
            # Predict Contract Price
            contract_model = RandomForestRegressor(n_estimators=100, random_state=42)
            X_train = complete_projects[features]
            y_train = complete_projects['ContractPrice']
            
            contract_model.fit(X_train, y_train)
            
            # Predict missing contract prices
            missing_contract_mask = incomplete_projects['ContractPrice'] <= 0
            if missing_contract_mask.any():
                X_predict = incomplete_projects.loc[missing_contract_mask, features]
                predicted_contracts = contract_model.predict(X_predict)
                incomplete_projects.loc[missing_contract_mask, 'ContractPrice'] = predicted_contracts
            
            # Predict Purchase Cost using both features and contract price
            # First, ensure all projects have contract prices
            all_projects_with_contract = pd.concat([complete_projects, incomplete_projects])
            
            # Calculate typical cost ratio for each project type
            cost_ratios = complete_projects.groupby('ProjectType', group_keys=False).apply(
                lambda x: (x['PurchaseCost'] / x['ContractPrice']).mean()
            )
            
            # Default ratio if not enough data
            default_ratio = 0.8  # 80% of contract price
            
            # Predict purchase costs
            for idx, row in incomplete_projects.iterrows():
                if row['PurchaseCost'] <= 0:
                    project_type = row['ProjectType']
                    ratio = cost_ratios.get(project_type, default_ratio)
                    
                    # Adjust ratio based on project complexity (more departments = higher costs)
                    complexity_factor = 1 + (row['NumDepartments'] - complete_projects['NumDepartments'].mean()) * 0.02
                    adjusted_ratio = ratio * complexity_factor
                    adjusted_ratio = np.clip(adjusted_ratio, 0.6, 0.95)  # Keep ratio reasonable
                    
                    predicted_cost = row['ContractPrice'] * adjusted_ratio
                    incomplete_projects.at[idx, 'PurchaseCost'] = predicted_cost
            
            # Update the original projects dataframe
            for idx, row in incomplete_projects.iterrows():
                mask = self.projects_df['ProjectCode'] == row['ProjectCode']
                self.projects_df.loc[mask, 'ContractPrice'] = row['ContractPrice']
                self.projects_df.loc[mask, 'PurchaseCost'] = row['PurchaseCost']
            
            logger.info(f"Imputed {len(incomplete_projects)} projects with missing values")
        
        elif len(incomplete_projects) > 0:
            # Fallback: use simple median imputation if not enough data for ML
            logger.warning("Not enough complete projects for ML imputation, using median values")
            
            median_contract = complete_projects['ContractPrice'].median() if len(complete_projects) > 0 else 1000000
            median_cost = complete_projects['PurchaseCost'].median() if len(complete_projects) > 0 else 800000
            
            mask = self.projects_df['ContractPrice'] <= 0
            self.projects_df.loc[mask, 'ContractPrice'] = median_contract
            
            mask = self.projects_df['PurchaseCost'] <= 0
            self.projects_df.loc[mask, 'PurchaseCost'] = median_cost
    
    def _calculate_project_metrics(self) -> pd.DataFrame:
        """Calculate metrics for each project"""
        # Aggregate work hours by project
        project_hours = self.work_hours_df.groupby('ProjectCode').agg({
            'Hours': 'sum',
            'Department': 'nunique',
            'Task': 'count'
        }).rename(columns={
            'Hours': 'TotalHours',
            'Department': 'NumDepartments',
            'Task': 'NumTasks'
        })
        
        # Calculate labor costs by project
        labor_costs = self._calculate_labor_costs()
        
        # Merge with project data
        summary = self.projects_df.merge(
            project_hours, 
            left_on='ProjectCode', 
            right_index=True, 
            how='left'
        )
        
        summary = summary.merge(
            labor_costs,
            left_on='ProjectCode',
            right_index=True,
            how='left'
        )
        
        # Fill missing values
        summary['TotalHours'] = summary['TotalHours'].fillna(0)
        summary['NumDepartments'] = summary['NumDepartments'].fillna(0)
        summary['NumTasks'] = summary['NumTasks'].fillna(0)
        summary['LaborCost'] = summary['LaborCost'].fillna(0)
        
        # Calculate financial metrics
        summary['TotalCost'] = summary['PurchaseCost'] + summary['LaborCost']
        summary['Profit'] = summary['ContractPrice'] - summary['TotalCost']
        
        # Fix profit margin calculation - handle edge cases properly
        summary['ProfitMargin'] = np.where(
            summary['ContractPrice'] > 0,
            (summary['Profit'] / summary['ContractPrice']) * 100,
            0
        )
        
        # Cap extreme profit margins to reasonable bounds (-200% to +200%)
        summary['ProfitMargin'] = np.clip(summary['ProfitMargin'], -200, 200)
        
        summary['EfficiencyScore'] = np.where(
            summary['TotalHours'] > 0,
            summary['Profit'] / summary['TotalHours'],
            0
        )
        
        # Add derived metrics
        summary['RevenuePerHour'] = np.where(
            summary['TotalHours'] > 0,
            summary['ContractPrice'] / summary['TotalHours'],
            0
        )
        summary['CostPerHour'] = np.where(
            summary['TotalHours'] > 0,
            summary['TotalCost'] / summary['TotalHours'],
            0
        )
        
        return summary
    
    def _calculate_labor_costs(self) -> pd.Series:
        """Calculate labor costs for each project"""
        # Add hourly rates to work hours data
        hourly_rates = {dept: salary / WORK_HOURS_PER_YEAR 
                       for dept, salary in DEPARTMENT_SALARIES.items()}
        
        # Create a copy to avoid modifying original data
        work_hours_with_cost = self.work_hours_df.copy()
        work_hours_with_cost['HourlyRate'] = work_hours_with_cost['Department'].map(hourly_rates)
        
        # Handle unknown departments with average rate
        avg_rate = np.mean(list(hourly_rates.values()))
        work_hours_with_cost['HourlyRate'] = work_hours_with_cost['HourlyRate'].fillna(avg_rate)
        
        # Calculate labor cost
        work_hours_with_cost['LaborCost'] = (
            work_hours_with_cost['Hours'] * work_hours_with_cost['HourlyRate']
        )
        
        # Aggregate by project
        labor_costs = work_hours_with_cost.groupby('ProjectCode')['LaborCost'].sum()
        
        return labor_costs
    
    def _calculate_department_metrics(self) -> pd.DataFrame:
        """Calculate metrics for each department"""
        # Get hourly rates
        hourly_rates = {dept: salary / WORK_HOURS_PER_YEAR 
                       for dept, salary in DEPARTMENT_SALARIES.items()}
        
        # Department aggregation
        dept_summary = self.work_hours_df.groupby('Department').agg({
            'Hours': 'sum',
            'ProjectCode': 'nunique',
            'Task': 'count'
        }).rename(columns={
            'Hours': 'TotalHours',
            'ProjectCode': 'NumProjects',
            'Task': 'NumTasks'
        })
        
        # Add salary information
        dept_summary['AverageSalary'] = dept_summary.index.map(DEPARTMENT_SALARIES)
        dept_summary['HourlyRate'] = dept_summary.index.map(hourly_rates)
        
        # Handle unknown departments
        avg_salary = np.mean(list(DEPARTMENT_SALARIES.values()))
        avg_rate = np.mean(list(hourly_rates.values()))
        dept_summary['AverageSalary'] = dept_summary['AverageSalary'].fillna(avg_salary)
        dept_summary['HourlyRate'] = dept_summary['HourlyRate'].fillna(avg_rate)
        
        # Calculate total labor cost
        dept_summary['TotalLaborCost'] = (
            dept_summary['TotalHours'] * dept_summary['HourlyRate']
        )
        
        # Calculate productivity metrics
        dept_summary['HoursPerProject'] = (
            dept_summary['TotalHours'] / dept_summary['NumProjects']
        )
        dept_summary['HoursPerTask'] = (
            dept_summary['TotalHours'] / dept_summary['NumTasks']
        )
        
        return dept_summary
    
    def get_project_department_matrix(self) -> pd.DataFrame:
        """Create matrix of hours by project and department"""
        matrix = self.work_hours_df.pivot_table(
            values='Hours',
            index='ProjectCode',
            columns='Department',
            aggfunc='sum',
            fill_value=0
        )
        return matrix
    
    def get_efficiency_analysis(self) -> Dict:
        """Perform efficiency analysis"""
        if self.financial_summary_df is None:
            self.calculate_all_metrics()
        
        analysis = {
            'top_efficient_projects': self._get_top_projects('EfficiencyScore', 10),
            'bottom_efficient_projects': self._get_bottom_projects('EfficiencyScore', 10),
            'efficiency_by_type': self._get_metrics_by_type('EfficiencyScore'),
            'efficiency_by_status': self._get_metrics_by_status('EfficiencyScore'),
            'efficiency_distribution': self._get_efficiency_distribution()
        }
        
        return analysis
    
    def _get_top_projects(self, metric: str, n: int) -> pd.DataFrame:
        """Get top N projects by specified metric"""
        return (self.financial_summary_df
                .nlargest(n, metric)
                [['ProjectCode', 'ProjectName', metric, 'Profit', 'TotalHours']])
    
    def _get_bottom_projects(self, metric: str, n: int) -> pd.DataFrame:
        """Get bottom N projects by specified metric"""
        return (self.financial_summary_df
                .nsmallest(n, metric)
                [['ProjectCode', 'ProjectName', metric, 'Profit', 'TotalHours']])
    
    def _get_metrics_by_type(self, metric: str) -> pd.DataFrame:
        """Get metrics grouped by project type"""
        return self.financial_summary_df.groupby('ProjectType').agg({
            metric: ['mean', 'std', 'min', 'max', 'count']
        }).round(2)
    
    def _get_metrics_by_status(self, metric: str) -> pd.DataFrame:
        """Get metrics grouped by project status"""
        return (self.financial_summary_df
                [self.financial_summary_df['Status'] != 'Unknown']
                .groupby('Status')
                .agg({metric: ['mean', 'std', 'min', 'max', 'count']})
                .round(2))
    
    def _get_efficiency_distribution(self) -> Dict:
        """Get efficiency score distribution statistics"""
        efficiency_scores = self.financial_summary_df['EfficiencyScore']
        
        return {
            'mean': efficiency_scores.mean(),
            'median': efficiency_scores.median(),
            'std': efficiency_scores.std(),
            'percentiles': {
                '25%': efficiency_scores.quantile(0.25),
                '50%': efficiency_scores.quantile(0.50),
                '75%': efficiency_scores.quantile(0.75),
                '90%': efficiency_scores.quantile(0.90)
            }
        }
    
    def get_profitability_analysis(self) -> Dict:
        """Perform profitability analysis"""
        if self.financial_summary_df is None:
            self.calculate_all_metrics()
        
        analysis = {
            'overall_metrics': self._get_overall_profitability(),
            'profitable_projects': self._get_profitable_projects(),
            'loss_making_projects': self._get_loss_making_projects(),
            'profit_by_type': self._get_metrics_by_type('Profit'),
            'profit_margin_by_status': self._get_metrics_by_status('ProfitMargin')
        }
        
        return analysis
    
    def _get_overall_profitability(self) -> Dict:
        """Get overall profitability metrics with corrected margin calculation"""
        summary = self.financial_summary_df
        
        total_revenue = summary['ContractPrice'].sum()
        total_cost = summary['TotalCost'].sum()
        total_profit = summary['Profit'].sum()
        
        # Calculate weighted average profit margin (correct method)
        overall_profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        return {
            'total_revenue': total_revenue,
            'total_cost': total_cost,
            'total_profit': total_profit,
            'average_profit_margin': overall_profit_margin,  # Now correctly calculated
            'profitable_projects_count': len(summary[summary['Profit'] > 0]),
            'loss_making_projects_count': len(summary[summary['Profit'] < 0]),
            'break_even_projects_count': len(summary[summary['Profit'] == 0])
        }
    
    def _get_profitable_projects(self) -> pd.DataFrame:
        """Get profitable projects summary"""
        profitable = self.financial_summary_df[self.financial_summary_df['Profit'] > 0]
        return (profitable
                .nlargest(20, 'Profit')
                [['ProjectCode', 'ProjectName', 'Profit', 'ProfitMargin', 'Status']])
    
    def _get_loss_making_projects(self) -> pd.DataFrame:
        """Get loss-making projects summary"""
        losses = self.financial_summary_df[self.financial_summary_df['Profit'] < 0]
        return (losses
                .nsmallest(20, 'Profit')
                [['ProjectCode', 'ProjectName', 'Profit', 'ProfitMargin', 'Status']])


def main():
    """Test data processing"""
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    work_hours, gs_projects, iss_projects = loader.load_all_data()
    combined_projects = loader.combine_projects()
    
    # Process data
    processor = DataProcessor(work_hours, combined_projects)
    financial_summary = processor.calculate_all_metrics()
    
    print("\nFinancial Summary Sample:")
    print(financial_summary[['ProjectCode', 'Profit', 'ProfitMargin', 'EfficiencyScore']].head(10))
    
    print("\nDepartment Summary:")
    print(processor.department_summary_df.head())
    
    print("\nEfficiency Analysis:")
    efficiency = processor.get_efficiency_analysis()
    print("Top Efficient Projects:")
    print(efficiency['top_efficient_projects'])
    
    print("\nProfitability Analysis:")
    profitability = processor.get_profitability_analysis()
    print("Overall Metrics:")
    for key, value in profitability['overall_metrics'].items():
        print(f"{key}: {value:,.2f}" if isinstance(value, (int, float)) else f"{key}: {value}")


if __name__ == "__main__":
    main()