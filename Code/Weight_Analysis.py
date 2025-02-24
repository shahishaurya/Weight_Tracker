import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import zscore

# Load CSV data
input_path = r"D:\Workspace\weight_logger\source_file"
output_path = r"D:\Workspace\weight_logger\output_file"
input_file_path = f"{input_path}\Weight_Logger.csv"  # Update with actual file path
df = pd.read_csv(input_file_path)
df.info()

# Ensure required columns exist
required_columns = {"Date","Weight", "Calories(Activity)", "Calories(Resting)", "Daily Steps", "Daily Kilometers", "Sleep Score", "Sleep Duration"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Convert date column to datetime
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
df = df.dropna(subset=["Date"])  # Remove rows where date conversion failed

# Take height input from user
while True:
    try:
        height = float(input("Enter your height in meters: "))
        if height <= 0:
            raise ValueError("Height must be a positive number.")
        break
    except ValueError:
        print("Invalid input. Please enter a valid height in meters.")

print("Entered Height in meters:", height/100)

# Calculate BMI for each weight entry
df["BMI"] = df["Weight"] / ((height/100) ** 2)

# Categorize BMI
def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 24.9:
        return "Normal weight"
    elif 25.0 <= bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"

df["BMI Category"] = df["BMI"].apply(categorize_bmi)
# Calculate Net Calories

df["Net Calories"] = df["Calories(Activity)"].fillna(0) + df["Calories(Resting)"].fillna(0)

# Step and Distance Efficiency
df["Calories per Step"] = df["Calories(Activity)"] / df["Daily Steps"]
df["Calories per KM"] = df["Calories(Activity)"] / df["Daily Kilometers"]

# Activity Level Classification
def classify_activity(steps):
    if steps < 5000:
        return "Sedentary"
    elif steps < 7500:
        return "Lightly Active"
    elif steps < 10000:
        return "Moderately Active"
    else:
        return "Very Active"
df["Activity Level"] = df["Daily Steps"].apply(classify_activity)

# Weight Trends Over Time if date column exists
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")
    df["Weight Change Rate"] = df["Weight"].diff() / df["Date"].diff().dt.days
    df["7-Day Moving Avg"] = df["Weight"].rolling(window=7).mean()
    df["30-Day Moving Avg"] = df["Weight"].rolling(window=30).mean()
    
# Weekly and Monthly Averages
df["week"] = df["Date"].dt.isocalendar().week
df["month"] = df["Date"].dt.month

# Convert necessary columns to numeric, forcing errors='coerce' to handle non-numeric values
cols_to_convert = ["Weight", "Daily Steps", "Calories(Activity)", "Sleep Duration"]
df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

# Now apply groupby and mean
weekly_avg = df.groupby("week")[cols_to_convert].mean()
monthly_avg = df.groupby("month")[cols_to_convert].mean()


# Correlation Analysis for Sleep and Weight
df["Sleep-Weight Corr"] = df["Sleep Duration"].rolling(window=7).corr(df["Weight"])

# Predictive Analysis - Linear Regression
if "Date" in df.columns:
    df.dropna(subset=["Weight", "Date"], inplace=True)  # Drop NaN only from relevant columns
    if not df.empty:
        df["days_since_start"] = (df["Date"] - df["Date"].min()).dt.days
        X = df[["days_since_start"]]  # Use days since start as independent variable
        y = df["Weight"]
        if len(X) > 1:  # Ensure there are enough samples
            model = LinearRegression()
            model.fit(X, y)
            future_days = np.array(range(df["days_since_start"].max() + 1, df["days_since_start"].max() + 31)).reshape(-1, 1)
            future_weights = model.predict(future_days)
            df_future = pd.DataFrame({
                "Date": pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=30, freq='D'),
                "Predicted Weight": future_weights
            })
        else:
            print("Not enough data points for linear regression.")
    else:
        print("No data available after dropping NaN values.")

# Correlation matrix
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr(method='pearson')  # Using Pearson correlation method
print("\nCorrelation Matrix:\n", correlation_matrix["Weight"].sort_values(ascending=False))

# Visualization - Time Series Plots
fig, axes = plt.subplots(5, 1, figsize=(12, 36))

# Subplot 1: Weight Over Time
df.plot(x="Date", y="Weight", ax=axes[0], marker='o', color='blue')
axes[0].axhline(df["Weight"].mean(), color='red', linestyle='dashed', label='Average Weight')
axes[0].set_title("Weight Over Time")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Weight (kg)")
axes[0].legend()

# Subplot 2: BMI Over Time
df.plot(x="Date", y="BMI", ax=axes[1], marker='s', color='red')
axes[1].axhline(df["BMI"].mean(), color='blue', linestyle='dashed', label='Average BMI')
axes[1].set_title("BMI Over Time")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("BMI")
axes[1].legend()

# Subplot 3: Distance and Moving Averages
df.plot(x="Date", y=["Daily Kilometers", "7-Day Moving Avg", "30-Day Moving Avg"], ax=axes[2], marker='x')
axes[2].axhline(df["Daily Kilometers"].mean(), color='red', linestyle='dashed', label='Average Distance')
axes[2].set_title("Distance and Moving Averages Over Time")
axes[2].set_xlabel("Date")
axes[2].set_ylabel("Kilometers")
axes[2].legend()

# Subplot 4: Daily Steps
df.plot(x="Date", y="Daily Steps", ax=axes[3], marker='s', color='green')
axes[3].axhline(df["Daily Steps"].mean(), color='red', linestyle='dashed', label='Average Steps')
axes[3].set_title("Daily Steps Over Time")
axes[3].set_xlabel("Date")
axes[3].set_ylabel("Steps")
axes[3].legend()

# Subplot 5: Net Calories, Activity, and Resting Calories as Bar Plot
df.plot(x="Date", y=["Net Calories", "Calories(Activity)", "Calories(Resting)"], kind="bar", ax=axes[4], colormap='viridis')
axes[4].axhline(df["Net Calories"].mean(), color='black', linestyle='dashed', label='Average Net Calories')
axes[4].set_title("Caloric Breakdown Over Time")
axes[4].set_xlabel("Date")
axes[4].set_ylabel("Calories")
axes[4].legend()


plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Predictive Analysis - Linear Regression
X = np.arange(len(df)).reshape(-1, 1)
y = df["Weight"].values

if len(X) > 1:
    model = LinearRegression()
    model.fit(X, y)
    df["Predicted Weight"] = model.predict(X)

    plt.figure(figsize=(12, 6))
    plt.scatter(df["Date"], df["Weight"], color='blue', label='Actual Weight')
    plt.plot(df["Date"], df["Predicted Weight"], color='red', linestyle='dashed', label='Predicted Weight')
    plt.axhline(df["Weight"].mean(), color='green', linestyle='dashed', label='Average Weight')
    plt.title("Weight Prediction Over Time")
    plt.xlabel("Date")
    plt.ylabel("Weight (kg)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

#Weight Trend Analysis using Polynomial Regression
X = np.arange(len(df)).reshape(-1, 1)
y = df["Weight"].values
poly_model = LinearRegression()
poly_model.fit(X, y)
df["Weight Trend"] = poly_model.predict(X)

#Sleep vs. Caloric Burn Analysis
sleep_cal_corr = df["Sleep Duration"].corr(df["Net Calories"])
print(f"Correlation between sleep duration and calorie burn: {sleep_cal_corr}")

#Step Count Clustering
scaler = StandardScaler()
df["scaled_steps"] = scaler.fit_transform(df[["Daily Steps"]])
kmeans = KMeans(n_clusters=3, random_state=42)
df["Activity Cluster"] = kmeans.fit_predict(df[["scaled_steps"]])

#Ideal Weight Comparison
ideal_weight = 22 * (height ** 2)
df["Weight Deviation"] = df["Weight"] - ideal_weight

#Activity Impact on Weight
activity_model = LinearRegression()
X_activity = df[["Daily Steps", "Net Calories"]]
y_weight = df["Weight"]
activity_model.fit(X_activity, y_weight)
print(f"Activity impact coefficients: {activity_model.coef_}")

#Sleep Quality Impact (assuming sleep efficiency available)
if "Sleep Efficiency" in df.columns:
    sleep_eff_corr = df["Sleep Efficiency"].corr(df["weight"])
    print(f"Correlation between sleep efficiency and weight: {sleep_eff_corr}")
 
#Metabolic Rate Estimation (BMR using Mifflin-St Jeor)
age = 30  # Assuming an age input
bmr = 10 * df["Weight"] + 6.25 * (height * 100) - 5 * age + 5  # Assuming male, subtract 161 for female
print(f"Estimated BMR: {bmr.mean()}")

#Anomaly Detection in Weight Changes
df["Weight Z-Score"] = zscore(df["Weight"])
df["Anomaly"] = df["Weight Z-Score"].apply(lambda x: "Yes" if abs(x) > 2 else "No")

# Display basic statistics with better formatting
print("\n📊 **Statistics** 📊\n")
print(df.describe().round(2).to_string())  # Rounds values to 2 decimal places and prints in table format

stats_df = df.describe().round(2)  # Round to 2 decimal places

# Define the output file path
output_file = "statistical_output.xlsx"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Define the full output file path
output_file = os.path.join(output_path, "statistics_output.xlsx")

# Save to Excel
stats_df.to_excel(output_file, sheet_name="Statistics")

print(f"✅ Statistics exported successfully to: {output_file}")
df.info()

# Write BMI, BMI Category, and Net Calories back to CSV
if not os.path.exists(output_path):
    os.makedirs(output_path)
# Define the full file path
output_file_path = os.path.join(output_path, "updated_Weight_logger.csv")
df.to_csv(output_file_path, index=False, columns=["Date", "Weight", "Calories(Resting)", "Calories(Activity)", "Daily Steps", "Daily Kilometers",
                                                  "Sleep Duration", "Sleep Score", "BMI", "BMI Category", "Net Calories", "Calories per Step",
                                                  "Calories per KM", "Activity Level", "Weight Change Rate", "7-Day Moving Avg", "30-Day Moving Avg",
                                                  "week", "month", "Sleep-Weight Corr", "days_since_start", "Predicted Weight", "Weight Trend",
                                                  "scaled_steps", "Activity Cluster", "Weight Deviation", "Weight Z-Score", "Anomaly"])
print(f"File saved successfully at: {output_file_path}")