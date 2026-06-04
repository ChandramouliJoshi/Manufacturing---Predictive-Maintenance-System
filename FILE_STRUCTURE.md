# File Structure

```text
Manufacturing - Predictive Maintenance System/
├── api/
│   └── API code will go here
├── dashboard/
│   └── Dashboard or UI code will go here
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv
│   └── processed/
│       └── Cleaned or transformed datasets will go here
├── models/
│   └── Trained model files will go here
├── notebooks/
│   └── Jupyter notebooks will go here
├── outputs/
│   ├── plots/
│   │   └── Generated charts and visualizations will go here
│   └── reports/
│       └── Generated reports will go here
├── src/
│   └── Source code for data processing and modeling will go here
├── .gitignore
├── README.md
└── requirements.txt
```

## Notes

- `data/` contains datasets and is usually ignored by Git.
- `outputs/` contains generated artifacts and is usually ignored by Git.
- `proj/` appears to be a local Python virtual environment, so it is not included in the main project structure.
