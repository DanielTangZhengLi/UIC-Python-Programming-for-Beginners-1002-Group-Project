import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Dataset size
n = 1000

# Generate data
data = {
    "TrackerModel": np.random.choice(["Basic", "Advanced", "Pro"], size=n, p=[0.4, 0.4, 0.2]),
    "Age": np.random.randint(18, 65, size=n),
    "Gender": np.random.choice(["Male", "Female"], size=n),
    "EducationYears": np.random.choice([10, 12, 14, 16, 18, np.nan], size=n, p=[0.1, 0.3, 0.3, 0.2, 0.05, 0.05]),
    "MaritalStatus": np.random.choice(["Single", "Married", "Divorced"], size=n, p=[0.5, 0.4, 0.1]),
    "DailySteps": np.clip(np.random.normal(loc=7500, scale=3000, size=n), 1000, 20000),
    "FitnessRating": np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.2, 0.3, 0.3, 0.15, 0.05]),
    "AnnualIncome": np.random.randint(20000, 200000, size=n),
    "WorkoutHoursWeekly": np.clip(np.random.normal(loc=3.5, scale=2, size=n), 0, 12),
    "ResidenceArea": np.random.choice(["Urban", "Suburban", "Rural"], size=n, p=[0.5, 0.3, 0.2]),
    "DeviceUsageHours": np.clip(np.random.normal(loc=2, scale=0.5, size=n), 0.5, 5)
}

# Add missing values
data["WorkoutHoursWeekly"][np.random.choice(n, size=30, replace=False)] = np.nan

# Add outliers to AnnualIncome
data["DailySteps"][np.random.choice(n, size=10, replace=False)] *= 10

# Create DataFrame
df = pd.DataFrame(data)

# Save dataset
df.to_csv("WellnessTracker.csv", index=False)
print("Dataset created: WellnessTracker.csv")