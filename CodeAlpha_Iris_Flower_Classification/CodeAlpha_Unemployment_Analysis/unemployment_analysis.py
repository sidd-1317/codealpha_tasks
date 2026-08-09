# CodeAlpha Data Science Internship - Task 2
# Unemployment Analysis with Python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = Path("Unemployment in India.csv")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Dataset not found. Download 'Unemployment in India.csv' and place it "
        "in this project folder."
    )

# 1. Load dataset
df = pd.read_csv(DATA_FILE)

# 2. Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.replace("\ufeff", "", regex=False)
)

# Some versions of the dataset contain leading spaces in text values.
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

# 3. Convert date
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# 4. Convert numeric columns
numeric_cols = [
    "Estimated Unemployment Rate (%)",
    "Estimated Employed",
    "Estimated Labour Participation Rate (%)"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("Dataset shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())

# Remove rows missing key analysis fields
needed = [
    "Region",
    "Date",
    "Estimated Unemployment Rate (%)",
    "Estimated Employed",
    "Estimated Labour Participation Rate (%)"
]
available = [c for c in needed if c in df.columns]
df = df.dropna(subset=available).copy()

# 5. Add time features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.month_name()

# 6. Overall unemployment trend
monthly = (
    df.groupby("Date")["Estimated Unemployment Rate (%)"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(11, 5))
sns.lineplot(
    data=monthly,
    x="Date",
    y="Estimated Unemployment Rate (%)"
)
plt.title("Average Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("unemployment_trend.png", dpi=150)
plt.show()

# 7. State/region comparison
region_avg = (
    df.groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 7))
region_avg.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Regions by Average Unemployment Rate")
plt.xlabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.savefig("top_regions.png", dpi=150)
plt.show()

# 8. Monthly pattern
month_avg = (
    df.groupby("Month_Name")["Estimated Unemployment Rate (%)"]
    .mean()
    .reindex([
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ])
)

plt.figure(figsize=(10, 5))
month_avg.plot(kind="bar")
plt.title("Average Unemployment Rate by Month")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_unemployment.png", dpi=150)
plt.show()

# 9. COVID-19 comparison
covid_start = pd.Timestamp("2020-03-01")
covid_end = pd.Timestamp("2020-06-30")

pre_covid = df[df["Date"] < covid_start]["Estimated Unemployment Rate (%)"].mean()
during_covid = df[
    (df["Date"] >= covid_start) & (df["Date"] <= covid_end)
]["Estimated Unemployment Rate (%)"].mean()
post_covid = df[
    df["Date"] > covid_end
]["Estimated Unemployment Rate (%)"].mean()

print("\nCOVID-19 period comparison:")
print(f"Before March 2020: {pre_covid:.2f}%")
print(f"March-June 2020:    {during_covid:.2f}%")
print(f"After June 2020:    {post_covid:.2f}%")

# 10. Correlation heatmap
corr_cols = [
    "Estimated Unemployment Rate (%)",
    "Estimated Employed",
    "Estimated Labour Participation Rate (%)"
]
corr_cols = [c for c in corr_cols if c in df.columns]

plt.figure(figsize=(7, 5))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f")
plt.title("Correlation Between Employment Indicators")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()

print("\nAnalysis completed successfully.")
