import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.load_data import load_data

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train, test, test_rul = load_data()

# Basic information
print("TRAINING DATA")
print(train.info())

print("\nMissing values:")
print(train.isnull().sum().sum())

print("\nDuplicate rows:")
print(train.duplicated().sum())

print("\nNumber of training engines:", train["unit"].nunique())
print("Number of testing engines:", test["unit"].nunique())

print("\nCycles per engine:")
print(train.groupby("unit")["cycle"].max().describe())

# Plot 1: Engine lifetime distribution
engine_lifetimes = train.groupby("unit")["cycle"].max()

plt.figure(figsize=(10, 5))
sns.histplot(engine_lifetimes, bins=25, kde=True)
plt.title("Distribution of Engine Lifetimes")
plt.xlabel("Operating Cycles")
plt.ylabel("Number of Engines")
plt.savefig(OUTPUT_DIR / "engine_lifetimes.png")
plt.close()

# Plot 2: Sensor readings over time
sensors = ["sensor_2", "sensor_3", "sensor_7", "sensor_11"]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for sensor, ax in zip(sensors, axes.flatten()):
    sample = train[train["unit"] == 1]

    ax.plot(sample["cycle"], sample[sensor])
    ax.set_title(f"{sensor} over Engine Life")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Reading")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sensor_trends.png")
plt.close()

# Plot 3: Sensor correlation
plt.figure(figsize=(18, 14))

sensor_columns = [c for c in train.columns if c.startswith("sensor_")]

sns.heatmap(
    train[sensor_columns].corr(),
    cmap="coolwarm",
    center=0
)

plt.title("Sensor Correlation Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sensor_correlation.png")
plt.close()

print("\nEDA completed.")
print("Charts saved inside data/processed/")