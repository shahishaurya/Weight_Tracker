# Weight_Tracker
 Notebook performs a comprehensive analysis of weight and related health metrics from data generated from Amazefit/Zepp App
 It includes the following key functionalities:

Data Loading & Preprocessing: Reads weight log data from a CSV file, handles missing values, and converts the date column for time-based analysis.
BMI Calculation & Classification: Computes Body Mass Index (BMI) and categorizes it into standard weight classes (Underweight, Normal, Overweight, Obese).
Caloric & Activity Analysis: Calculates net calories, efficiency metrics (calories per step/km), and classifies activity levels based on step count.
Weight Trends & Predictions: Uses rolling averages, correlation analysis, and predictive modeling (linear regression & polynomial regression) to analyze weight changes.
Sleep & Weight Correlation: Evaluates the impact of sleep duration and efficiency on weight fluctuations.
Clustering & Anomaly Detection: Applies K-Means clustering for activity segmentation and detects unusual weight changes using Z-score analysis.
Basal Metabolic Rate (BMR) Estimation: Uses the Mifflin-St Jeor equation to estimate metabolic rate.
Visualization & Reporting: Generates time-series plots for weight trends, correlations, and predictive insights.
Data Export: Saves processed data, including BMI and caloric metrics, into an updated CSV file.
This notebook is useful for tracking weight loss progress, understanding activity and caloric balance, and predicting future weight trends.
