# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'password',
    'database': 'wtl_financial_db'
}

# Department salary information (synthetic data in CNY)
DEPARTMENT_SALARIES = {
    'IT部': 85000,
    '市场部': 75000,
    '设计部': 90000,
    '施工部': 70000,
    '项目管理部': 95000,
    '财务部': 80000,
    '人力资源部': 72000,
    '采购部': 78000,
    '质量控制部': 82000,
    '研发部': 92000,
    '客户服务部': 68000,
    '法务部': 88000,
    '行政部': 65000,
    '业务发展部': 87000
}

# Project status color mapping
COLOR_STATUS_MAP = {
    'Green': 'Success',
    'Dark Green': 'Negotiation',
    'White': 'In Progress',
    'Yellow': 'Fail'
}

#Change file path accordingly
EXCEL_FILE_PATH = r'C:\Users\arjun\Downloads\WTL Design Jr. Analyst Task.xlsx'

# Sheet names
SHEETS = {
    'work_hours': '20X2Q3 Work Hour',
    'gs_projects': 'GS Project',
    'iss_projects': 'ISS Project'
}

# Column mappings
WORK_HOURS_COLUMNS = {
    'original': ['Date', 'Project', 'Stage', 'Department', 'Detailed Task', 'Work Hour'],
    'renamed': ['Date', 'Project', 'Stage', 'Department', 'Task', 'Hours']
}

GS_PROJECT_COLUMNS = {
    'original': ['Project Code', 'Project Name', 'Contract Price', 'Purchase Cost'],
    'max_rows': 120  # Read 120 data rows (excluding header)
}

ISS_PROJECT_COLUMNS = {
    'original': ['Project Code', 'Project Name', 'Contract Price', 'Purchase Cost'],
    'max_rows': 148  # Read 148 data rows (excluding header)
}

# Color coding for GS projects (row numbers after header)
GS_COLOR_CODING = {
    'Green': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 33],
    'White': [24, 30, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 51, 52, 53, 54, 55, 57, 59, 61, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
    'Dark Green': [27, 28, 29, 32, 34, 36, 45, 47, 56, 58, 60, 62, 71, 76, 81],
    'Yellow': [31]
}

# Report configuration
REPORT_CONFIG = {
    'output_dir': 'reports',
    'visualizations_dir': 'visualizations',
    'auto_report_template': 'report_template.json'
}

# Calculation parameters
WORK_HOURS_PER_YEAR = 2080