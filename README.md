# Overview
This ML system provides advanced analytics for WTL's project data, focusing on:

Project Success Prediction - Predicts whether GS projects will succeed or fail
Anomaly Detection - Identifies unusual projects that require attention

# System Architecture
WTL_Predictive/  
├── ml_config.py           # Configuration and parameters       
├── feature_engineering.py # Feature creation and selection       
├── success_predictor.py   # Success prediction models      
├── anomaly_detector.py    # Anomaly detection algorithms  
├── ml_pipeline.py        # Pipeline orchestrator      
├── ml_main.py           # Main execution script     
├── config.py            # Configuration and constants    
├── data_processor.py    # Data processing and calculations  
├── data_loader.py       # Excel data loading   
└── ml_requirements.txt  # Python dependencies    


# Usage
Run complete ML analysis:
python ml_main.py

Use custom Excel file:
python ml_main.py --excel "path/to/file.xlsx"

Generate detailed report:
python ml_main.py --report

Analyze specific project:
python ml_main.py --analyze-project "GS240001-BJ01"

# Output files
Located in ml_reports/ directory:
predictive analysis.csv 
