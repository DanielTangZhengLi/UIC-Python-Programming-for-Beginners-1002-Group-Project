import pandas as pd
import numpy as np

# Read the original dataset
df = pd.read_csv("WellnessTracker.csv")

# Handle missing values and outliers

# 1. Fill missing values in EducationYears with the median
education_median = df["EducationYears"].median()
df["EducationYears"] = df["EducationYears"].fillna(education_median)

# 2. Fill missing values in WorkoutHoursWeekly with the mean
workout_mean = df["WorkoutHoursWeekly"].mean()
df["WorkoutHoursWeekly"] = df["WorkoutHoursWeekly"].fillna(workout_mean)

# 3. Process outliers in DailySteps using the IQR method
Q1 = df["DailySteps"].quantile(0.25)  # First quartile (25th percentile)
Q3 = df["DailySteps"].quantile(0.75)  # Third quartile (75th percentile)
IQR = Q3 - Q1                         # Interquartile range
lower_bound = Q1 - 1.5 * IQR          # Lower bound
upper_bound = Q3 + 1.5 * IQR          # Upper bound

# Replace outliers with the lower or upper bounds and convert to integers
df["DailySteps"] = np.where(
    df["DailySteps"] < lower_bound, lower_bound,
    np.where(df["DailySteps"] > upper_bound, upper_bound, df["DailySteps"])
).astype(int)  # Ensure the values are integers

# 4. Round WorkoutHoursWeekly and DeviceUsageHours to two decimal places
df["WorkoutHoursWeekly"] = df["WorkoutHoursWeekly"].round(2)
df["DeviceUsageHours"] = df["DeviceUsageHours"].round(2)

# Save the cleaned dataset
df.to_csv("WellnessTracker_Cleaned.csv", index=False)
print("Cleaned dataset saved: WellnessTracker_Cleaned.csv")