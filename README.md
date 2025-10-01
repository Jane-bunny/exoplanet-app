# Exoplanet App 🚀

A reproducible data science project exploring exoplanet catalogs from **Kepler (KOI cumulative)** and **TESS (TOI)**.  
The goal is to build a clean processing pipeline, align features, and prepare the data for visualisation and machine learning experiments.

--- 

## 🛠️ Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/Jane-bunny/exoplanet-app.git
cd exoplanet-app

python -m venv .venv
.venv\Scripts\Activate.ps1   # on Windows PowerShell
pip install --upgrade pip
```

## Dependencies
pip install -r requirements.txt

## Usage
python -m src.data_processing

## 📂 Project structure
exoplanet-app/
│
├── data/
│   ├── raw/           # raw input files (ignored by git)
│   ├── processed/     # cleaned, aligned datasets (tracked)
│   └── .gitkeep
│
├── docs/              # documentation (e.g., data dictionary)
├── figures/           # generated plots
├── notebooks/         # exploration, preview, EDA, modeling
│   ├── 01_data_processing.ipynb
│   ├── 02_preview_processed.ipynb
│   └── ...
│
├── src/               # source code
│   ├── __init__.py
│   ├── data_processing.py
│   └── plots.py       # (optional, for reproducible visualisations)
│
├── README.md
└── requirements.txt

## 📓 Notebooks

The notebooks document different stages of the workflow:

- **01_data_processing.ipynb**  
  Exploration of the raw Kepler and TESS catalogs in `data/raw/`.  
  Includes first inspections with `.info()` / `.head()` and notes on how to clean, align, and prepare the data.  
  Some outputs are kept to show exploratory reasoning.

- **02_preview_processed.ipynb**  
  Lightweight validation of the cleaned datasets created by `src/data_processing.py`.  
  Loads the processed CSVs from `data/processed/`, checks shapes, columns, and label counts.  
  Outputs are cleared for reproducibility.

More detailed analysis and visualisations will be added in later notebooks (`03_eda_visualisation.ipynb`, `04_modeling.ipynb`, …).

## 📊 Data Sources

This project uses publicly available exoplanet catalogs from the **NASA Exoplanet Archive**:

- **Kepler KOI (Cumulative) Catalog** — see [Kepler Objects of Interest documentation](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html)
- **TESS TOI Catalog / Column Definitions** — see [TOI Table Data Column Definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html) 
- More on the archive’s data holdings: [Data Resources in the Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/data.html)


Data was downloaded in CSV format and placed in `data/raw/`.  
Cleaning and alignment is handled by `src/data_processing.py`, with outputs in `data/processed/`.