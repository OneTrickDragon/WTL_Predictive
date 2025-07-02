import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Load and perform initial processing of Excel data"""
    
    def __init__(self, file_path: str = EXCEL_FILE_PATH):
        self.file_path = file_path
        self.work_hours_df = None
        self.gs_projects_df = None
        self.iss_projects_df = None
        
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all data from Excel file"""
        logger.info(f"Loading data from {self.file_path}")
        
        try:
            # Load work hours data
            self.work_hours_df = self._load_work_hours()
            
            # Load GS projects data
            self.gs_projects_df = self._load_gs_projects()
            
            # Load ISS projects data
            self.iss_projects_df = self._load_iss_projects()
            
            logger.info("All data loaded successfully")
            return self.work_hours_df, self.gs_projects_df, self.iss_projects_df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _load_work_hours(self) -> pd.DataFrame:
        """Load work hours data from Excel"""
        df = pd.read_excel(
            self.file_path,
            sheet_name=SHEETS['work_hours'],
            header=0
        )
        
        # Rename columns
        df.columns = WORK_HOURS_COLUMNS['renamed']
        
        # Extract project code from project name
        df['ProjectCode'] = df['Project'].str.extract(r'(GS\d+[-\w]*|ISS\d+[-\w]*)')
        
        # Clean numeric data
        df['Hours'] = pd.to_numeric(df['Hours'], errors='coerce').fillna(0)
        
        logger.info(f"Loaded {len(df)} work hour records")
        return df
    
    def _load_gs_projects(self) -> pd.DataFrame:
        """Load GS projects data from Excel"""
        # Read Excel keeping the header row
        df = pd.read_excel(
            self.file_path,
            sheet_name=SHEETS['gs_projects'],
            header=0,  # First row contains headers
            nrows=GS_PROJECT_COLUMNS['max_rows']  # Read header + 120 data rows
        )
        
        # Keep only first 4 columns
        df = df.iloc[:, :4].copy()
        
        # Rename columns to standard names
        df.columns = ['ProjectCode', 'ProjectName', 'ContractPrice', 'PurchaseCost']
        
        # Add project type
        df['ProjectType'] = 'GS'
        
        # Create row index for color mapping (1-based, counting data rows only)
        df['RowIndex'] = range(1, len(df) + 1)
        
        # Map colors based on row index
        df['ColorCode'] = 'Unknown'  # Default
        df['Status'] = 'Unknown'  # Default
        
        for color, rows in GS_COLOR_CODING.items():
            mask = df['RowIndex'].isin(rows)
            df.loc[mask, 'ColorCode'] = color
            df.loc[mask, 'Status'] = COLOR_STATUS_MAP.get(color, 'Unknown')
        
        # Drop the temporary row index column
        df = df.drop('RowIndex', axis=1)
        
        # Filter out rows without project code
        df = df[df['ProjectCode'].notna()].copy()
        
        # Clean numeric columns
        df['ContractPrice'] = pd.to_numeric(df['ContractPrice'], errors='coerce').fillna(0)
        df['PurchaseCost'] = pd.to_numeric(df['PurchaseCost'], errors='coerce').fillna(0)
        
        logger.info(f"Loaded {len(df)} GS projects with color coding applied")
        return df
    
    def _load_iss_projects(self) -> pd.DataFrame:
        """Load ISS projects data from Excel"""
        # Read Excel keeping the header row
        df = pd.read_excel(
            self.file_path,
            sheet_name=SHEETS['iss_projects'],
            header=0,  # First row contains headers
            nrows=ISS_PROJECT_COLUMNS['max_rows']  # Read header + 148 data rows
        )
        
        # Keep only first 4 columns
        df = df.iloc[:, :4].copy()
        
        # Rename columns to standard names
        df.columns = ['ProjectCode', 'ProjectName', 'ContractPrice', 'PurchaseCost']
        
        # Add project type and default status
        df['ProjectType'] = 'ISS'
        df['ColorCode'] = None
        df['Status'] = 'Unknown'
        
        # Filter out rows without project code
        df = df[df['ProjectCode'].notna()].copy()
        
        # Clean numeric columns
        df['ContractPrice'] = pd.to_numeric(df['ContractPrice'], errors='coerce').fillna(0)
        df['PurchaseCost'] = pd.to_numeric(df['PurchaseCost'], errors='coerce').fillna(0)
        
        logger.info(f"Loaded {len(df)} ISS projects")
        return df
    
    def combine_projects(self) -> pd.DataFrame:
        """Combine GS and ISS projects into single dataframe"""
        if self.gs_projects_df is None or self.iss_projects_df is None:
            raise ValueError("Must load data first using load_all_data()")
        
        # Select common columns
        common_columns = ['ProjectCode', 'ProjectName', 'ProjectType', 
                         'ContractPrice', 'PurchaseCost', 'Status', 'ColorCode']
        
        combined_df = pd.concat([
            self.gs_projects_df[common_columns],
            self.iss_projects_df[common_columns]
        ], ignore_index=True)
        
        logger.info(f"Combined {len(combined_df)} total projects")
        return combined_df


def main():
    """Test data loading"""
    loader = DataLoader()
    work_hours, gs_projects, iss_projects = loader.load_all_data()
    combined_projects = loader.combine_projects()
    
    print("\nData Loading Summary:")
    print(f"Work Hours Records: {len(work_hours)}")
    print(f"GS Projects: {len(gs_projects)}")
    print(f"ISS Projects: {len(iss_projects)}")
    print(f"Total Projects: {len(combined_projects)}")
    
    print("\nSample Work Hours Data:")
    print(work_hours.head())
    
    print("\nSample Combined Projects:")
    print(combined_projects.head())


if __name__ == "__main__":
    main()