import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t

# Load the dataset
data = pd.read_csv("WellnessTracker_Cleaned.csv")

# PART 1: Advanced Data Analysis

# 1. (Complex Ratio) What is the ratio of customers who report walking fewer than 5,000 steps daily
# to those walking more than 12,000 steps daily (rounded to 2 decimal places)?
# Calculate the number of customers walking less than 5,000 steps and those walking more than 12,000 steps,
# then return the ratio rounded to two decimal places.
# (Round to two decimal places)

### BEGIN SOLUTION
less_5000 = len(data[data["DailySteps"] < 5000])  # Rows with DailySteps < 5000
more_12000 = len(data[data["DailySteps"] > 12000])  # Rows with DailySteps > 12000

try:
    a1 = round(less_5000 / more_12000, 2)  # Calculate ratio
except ZeroDivisionError:
    a1 = None  # Handle division by zero
### END SOLUTION

# 2. (Nested Grouping) For each ResidenceArea, calculate the median AnnualIncome.
# Then return the difference between the highest and lowest median incomes across the three areas.
# (No rounding needed)

### BEGIN SOLUTION
median_incomes = data.groupby("ResidenceArea", observed=True)["AnnualIncome"].median()
a2 = median_incomes.max() - median_incomes.min()  # Difference
### END SOLUTION

# 3. (Filtered Mean) Calculate the average WorkoutHoursWeekly of female customers who are Married
# and live in Suburban areas. Round to 2 decimal places.
# (Round to two decimal places)

### BEGIN SOLUTION
filtered_data = data[(data["Gender"] == "Female") &
                     (data["MaritalStatus"] == "Married") &
                     (data["ResidenceArea"] == "Suburban")]
a3 = round(filtered_data["WorkoutHoursWeekly"].mean(), 2)  # Average hours
### END SOLUTION

# 4. (Custom Metric) Define a new metric called ActivityEfficiency, calculated as
# ActivityEfficiency = DailySteps / DeviceUsageHours.
# What is the average ActivityEfficiency of customers younger than 40? Round to 2 decimal places.
# (Round to two decimal places)

### BEGIN SOLUTION
data["ActivityEfficiency"] = data["DailySteps"] / data["DeviceUsageHours"]
a4 = round(data[data["Age"] < 40]["ActivityEfficiency"].mean(), 2)
### END SOLUTION

# 5. (Advanced Count) How many customers under the age of 30 have a FitnessRating of 4 or 5 and live in Urban areas?
# (No rounding needed)

### BEGIN SOLUTION
a5 = len(data[(data["Age"] < 30) &
              (data["FitnessRating"] >= 4) &
              (data["ResidenceArea"] == "Urban")])
### END SOLUTION

# 6. (Weighted Sum) Compute a weighted sum of FitnessRating, weighted by WorkoutHoursWeekly,
# for all customers whose DailySteps exceed 10,000.
# (Round to two decimal places)

### BEGIN SOLUTION
high_steps = data[data["DailySteps"] > 10000]

# Compute the weighted sum normalized by the total WorkoutHoursWeekly.
numerator = (high_steps["FitnessRating"] * high_steps["WorkoutHoursWeekly"]).sum()
denominator = high_steps["WorkoutHoursWeekly"].sum()

# To avoid division by zero, check if the denominator is greater than zero.
a6 = round(numerator / denominator, 2) if denominator > 0 else 0
### END SOLUTION

# 7. (Proportional Analysis) What percentage of customers live in Urban areas and walk fewer than 6,000 steps daily?
# Provide the answer as a float (e.g., 0.12).
# (Round to four decimal places)

### BEGIN SOLUTION
urban_customers = data[data["ResidenceArea"] == "Urban"]  # Filter urban customers

try:
    a7 = round(len(urban_customers[urban_customers["DailySteps"] < 6000]) / len(urban_customers), 4)  # Calculate ratio
except ZeroDivisionError:
    a7 = None  # Handle zero division
### END SOLUTION

# 8. (Conditional Count) How many customers have an AnnualIncome in the top 5% but report FitnessRating below 3?
# (No rounding needed)

### BEGIN SOLUTION
income_threshold = data["AnnualIncome"].quantile(0.95)
a8 = len(data[(data["AnnualIncome"] > income_threshold) & (data["FitnessRating"] < 3)])
### END SOLUTION

# 9. (Time-to-Fit Prediction)
# Assuming a customer improves their fitness level by multiplying their current FitnessRating by 1.2 every week,
# calculate how many weeks it would take for a customer with a FitnessRating of 2 to reach or exceed a FitnessRating of 5.
# Round to the nearest integer.

### BEGIN SOLUTION
initial_value = 2  # Initial value
growth_rate = 1.2  # Growth rate (multiplier)
target_value = 5   # Target value
a9 = 0  # Step counter
current_value = initial_value  # Current value
# Loop until the current value reaches or exceeds the target value
while current_value < target_value:
    current_value *= growth_rate  # Increase current value by the growth rate
    a9 += 1  # Increment the step counter
### END SOLUTION

# 10. (Correlation Analysis) What is the correlation coefficient between DailySteps and WorkoutHoursWeekly
# for Rural customers?
# (Round to four decimal places)

### BEGIN SOLUTION
rural_data = data[data["ResidenceArea"] == "Rural"]
a10 = round(rural_data["DailySteps"].corr(rural_data["WorkoutHoursWeekly"]), 4)
### END SOLUTION

# Store all answers in a dictionary
answers = {"1": a1, "2": a2, "3": a3, "4": a4, "5": a5, "6": a6, "7": a7, "8": a8, "9": a9, "10": a10}

print(answers)

# PART 2: Advanced Visualizations

# 1. Pie chart: Percentage distribution of customers by ResidenceArea.

### BEGIN SOLUTION
data["ResidenceArea"].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title("Residence Area Distribution")
plt.ylabel("")
plt.show()
### END SOLUTION

# 2. Histogram: Distribution of ActivityEfficiency.

### BEGIN SOLUTION
plt.hist(data["ActivityEfficiency"].dropna(), bins=20, edgecolor='black')
plt.title("Activity Efficiency Distribution")
plt.xlabel("Activity Efficiency")
plt.ylabel("Frequency")
plt.show()
### END SOLUTION

# 3. Horizontal Bar Plot: Average AnnualIncome by MaritalStatus, differentiated by Gender.

### BEGIN SOLUTION
income_plot_data = data.groupby(["MaritalStatus", "Gender"], observed=True)["AnnualIncome"].mean().unstack()
income_plot_data.plot(kind="barh", stacked=True)
plt.title("Average Income by Marital Status and Gender")
plt.xlabel("Annual Income")
plt.ylabel("Marital Status")
plt.legend(title="Gender")
plt.show()
### END SOLUTION

# 4. Heatmap: Correlation matrix of numeric features.

### BEGIN SOLUTION
sns.set_theme(style="whitegrid", font_scale=1.2)
data = pd.read_csv("WellnessTracker_Cleaned.csv")
numeric_features = ["Age", "DailySteps", "FitnessRating", "AnnualIncome", "WorkoutHoursWeekly", "DeviceUsageHours"]

correlation_matrix = data[numeric_features].corr()

n = len(data)

r_values = correlation_matrix.values
np.fill_diagonal(r_values, np.nan)
t_values = (r_values * np.sqrt(n - 2)) / np.sqrt(1 - r_values**2)
p_values = 2 * (1 - t.cdf(np.abs(t_values), df=n - 2))

np.fill_diagonal(r_values, 1.0)
np.fill_diagonal(t_values, np.nan)
np.fill_diagonal(p_values, 0.0)

t_value_matrix = pd.DataFrame(t_values, index=numeric_features, columns=numeric_features)
p_value_matrix = pd.DataFrame(p_values, index=numeric_features, columns=numeric_features)

def plot_heatmap(data_matrix, title, fmt, cmap):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data_matrix.round(4),
        annot=True,
        fmt=fmt,
        cmap=cmap,
        cbar_kws={"shrink": 0.8},
        square=True,
        linewidths=0.5
    )
    plt.title(title, fontsize=16, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

plot_heatmap(correlation_matrix, "Correlation Matrix", ".4f", "RdBu_r")
plot_heatmap(t_value_matrix, "T-Value Matrix", ".4f", "inferno")
plot_heatmap(p_value_matrix, "P-Value Matrix (Significance of Correlations)", ".2e", "viridis")
### END SOLUTION

# 5. Jointplot: Relationship between AnnualIncome and DailySteps for Urban customers.

### BEGIN SOLUTION
urban_data = data[data["ResidenceArea"] == "Urban"]
jointplot = sns.jointplot(
    data=urban_data,
    x="AnnualIncome",
    y="DailySteps",
    kind="kde",
    fill=True,
    cmap="Blues"
)
plt.suptitle("Income vs. Steps (Urban Customers)", y=1.05, fontsize=14)
jointplot.ax_joint.set_xlabel("Annual Income", fontsize=12)
jointplot.ax_joint.set_ylabel("Daily Steps", fontsize=12)
jointplot.ax_joint.tick_params(axis="both", which="major", labelsize=10)
plt.show()
### END SOLUTION

# 6. Box Plot: DailySteps by TrackerModel, categorized by Gender.

### BEGIN SOLUTION
sns.boxplot(data=data, x="TrackerModel", y="DailySteps", hue="Gender")
plt.title("Daily Steps by Tracker Model and Gender")
plt.tight_layout()
plt.show()
### END SOLUTION