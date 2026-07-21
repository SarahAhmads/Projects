# Ford GoBike Interactive Dashboard

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Graphs-indigo?logo=plotly&logoColor=white)](https://plotly.com/)
[![Dash](https://img.shields.io/badge/Dash-Dashboard-008DB8?logo=dash)](https://dash.plotly.com/)
[![Data Analysis](https://img.shields.io/badge/Data-Analysis-green)](#)

## Project Overview
This project is an end-to-end data analysis and visualization solution based on the **Ford GoBike System Data**. 
The project is divided into three main phases:
1. **Exploratory Data Analysis (EDA):** In-depth univariate, bivariate, and multivariate analysis inside a polished Jupyter Notebook.
2. **Data Preprocessing & Feature Engineering:** Cleaning data, handling missing values, and engineering new features (e.g., trip duration in minutes, age groups).
3. **Interactive Dashboard:** A dynamic web application featuring KPIs, time analysis, user behavior analysis, and station analysis with dynamic filters.

---

## Project Structure
The repository is professionally structured as follows:

```text
Interactive Dashboard/
├── dashboard/               # Main Dashboard application files
│   ├── app.py               # Application entry point
│   ├── config.py            # Centralized settings & color themes
│   ├── assets/              # Custom CSS and images
│   └── components/          # Reusable UI components (charts, filters, KPIs)
├── data/                    
│   ├── raw/                 # Original dataset (CSV)
│   └── processed/           # Cleaned and processed data files
├── notebooks/               # Jupyter notebooks for EDA and reporting
├── scripts/                 # Python scripts for data cleaning & visualization
├── CONTRIBUTING.md          # Rules for pushing code and GitHub workflows
├── TEAM_TASKS.md            # Team roles and responsibilities tracker
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## Team & Collaboration
This project is developed by a dedicated team of 6 members divided into two specialized squads:
- **Team 1 (Data & Notebook):** Responsible for data cleaning, feature engineering, and extracting insights.
- **Team 2 (Interactive Dashboard):** Responsible for UI/UX, interactivity (callbacks), and Plotly visualizations.

> **Important:** Please check [`TEAM_TASKS.md`](TEAM_TASKS.md) for current task assignments and progress.

### Contribution Rules
We follow a strict Git Workflow to maintain code quality:
1. **Never push directly to `main`**.
2. Work on separate branches (`feature/task-name` or `bugfix/issue-name`).
3. All PRs must be reviewed and approved by the Team Leader.
> Read the full rules in our [`CONTRIBUTING.md`](CONTRIBUTING.md) file.

---

## How to Run the Project Locally

**1. Clone the repository:**
```bash
git clone https://github.com/your-org/ford-gobike-dashboard.git
cd ford-gobike-dashboard
```

**2. Set up the virtual environment (Optional but Recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the Data Preprocessing Script (First time only):**
```bash
python scripts/preprocessing.py
```

**5. Launch the Dashboard:**
```bash
python dashboard/app.py
```
*Open `http://127.0.0.1:8050/` in your browser to view the application.*
