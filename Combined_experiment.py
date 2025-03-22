import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dateutil.relativedelta import relativedelta
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel('CPI_monthly_2025_changes.xlsx', skiprows=11)
df = df.drop(columns=['Series ID'])

# Keep only monthly data
df = df[df['Period'].str.startswith('M')]

# Convert 'Year' & 'Period' into a proper Date format
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Period'].str[1:] + '-01')

# Keep only relevant columns
df = df[['Date', 'Value']]
df.columns = ['timestamp', 'CPI']

# Set date as index
df = df.set_index('timestamp')

# Outlier Treatment
q1 = np.percentile(df['CPI'], 25)
q3 = np.percentile(df['CPI'], 75)
boolean_condition = (df['CPI'] < q1 - (q3-q1)*1.3) | (df['CPI'] > q3 + (q3-q1)*1.3)
df.loc[boolean_condition, 'CPI'] = np.mean(df['CPI'])




train_size = int(0.9 * df.shape[0])
HW_train = df.iloc[:train_size]
HW_test = df.iloc[train_size:]



# Fit Holt-Winters model
hw_model = ExponentialSmoothing(HW_train['CPI'], seasonal_periods=12, trend='add', seasonal='add')
hw_fit = hw_model.fit()

# Forecast
hw_forecast = hw_fit.forecast(steps=len(HW_test))


hw_rmse = np.sqrt(mean_squared_error(HW_test['CPI'], hw_forecast))
hw_mae = mean_absolute_error(HW_test['CPI'], hw_forecast)

print("Holt-Winters Performance:")
print(f"RMSE: {hw_rmse:.4f}")
print(f"MAE: {hw_mae:.4f}")



from statsmodels.tsa.holtwinters import ExponentialSmoothing
from dateutil.relativedelta import relativedelta
from datetime import datetime

# Set rolling forecast start and end times
start_time = datetime(1913, 1, 1)
end_time = datetime(2014, 1, 1)

roll_end_time = end_time
HW_roll_train = df.loc[:end_time].copy()  # Copy data until end_time

# List to store rolling predictions
rolling_forecast = []

# Rolling forecast loop
while roll_end_time in df.index:  # Ensure roll_end_time exists in dataset
    # Select current rolling test point
    HW_roll_test = df.loc[roll_end_time:roll_end_time]

    # Train Holt-Winters model on rolling training set
    HW_roll_model = ExponentialSmoothing(HW_roll_train['CPI'], seasonal_periods=12, trend='add', seasonal='add')
    HW_roll_model_fit = HW_roll_model.fit()

    # Forecast next step
    HW_roll_predictions = HW_roll_model_fit.forecast(steps=1)

    # Store forecasted values
    rolling_forecast.append((roll_end_time, float(HW_roll_predictions)))

    # Print rolling forecast progress
    print(f"Rolling forecast for {roll_end_time}: {float(HW_roll_predictions)}")

    # Move to next month
    roll_end_time = roll_end_time + relativedelta(months=+1)

    # Append the prediction to the rolling training set
    new_entry = pd.DataFrame({'CPI': [float(HW_roll_predictions)]}, index=[roll_end_time])
    HW_roll_train = pd.concat([HW_roll_train, new_entry])

# Convert rolling forecast list to DataFrame
HW_rolling_results = pd.DataFrame(rolling_forecast, columns=['timestamp', 'CPI']).set_index('timestamp')