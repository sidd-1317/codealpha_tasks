# CodeAlpha Task 2 — Unemployment Analysis with Python

## Objective
Analyze unemployment data to understand trends, regional differences, monthly patterns and the effect of the COVID-19 period.

## Tech Stack
- Python
- Pandas
- Matplotlib
- Seaborn

## Dataset
The commonly used "Unemployment in India" dataset is available on Kaggle:

https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india

The dataset description identifies fields such as Region, Date, Frequency, Estimated Unemployment Rate (%), Estimated Employed and Estimated Labour Participation Rate (%).

Download the CSV and place `Unemployment in India.csv` in this project folder.

## How to Run
```bash
python -m pip install -r requirements.txt
python unemployment_analysis.py
```

## Analysis Performed
1. Data loading
2. Column-name and text cleaning
3. Date conversion
4. Numeric conversion
5. Missing-value handling
6. Overall unemployment trend
7. Top regions by average unemployment
8. Monthly pattern analysis
9. Before/during/after COVID-19 comparison
10. Correlation heatmap

## Generated Charts
- `unemployment_trend.png`
- `top_regions.png`
- `monthly_unemployment.png`
- `correlation_heatmap.png`

## Important
Run the script first and use the numbers printed by your own run in your final report. Do not invent results.

## GitHub Repository Name
`CodeAlpha_Unemployment_Analysis`
