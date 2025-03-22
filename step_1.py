import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_excel('CPI_monthly_2025_changes.xlsx', skiprows=11)
print(df.shape)

df = df.drop(columns=['Series ID'])
print(df.shape)

df = df[df['Period'].str.startswith('M')]

df['Date'] = df['Year'].astype(str) + '-' + df['Period'].str[1:] + '-01'
df['Date'] = pd.to_datetime(df['Date'])

df = df[['Date', 'Value']]  # Reorder columns
df.columns = ['timestamp', 'CPI']  # Rename columns for consistency


df = df.set_index('timestamp')



q1 = np.percentile(df['CPI'],25)
q3 = np.percentile(df['CPI'],75)
boolean_condition = (df['CPI'] < q1 - (q3-q1)*1.3) | (df['CPI'] > q3 + (q3-q1)*1.3)
column_name = 'CPI'
new_value = np.mean(df['CPI'])

df.loc[boolean_condition, column_name] = new_value


from statsmodels.tsa.seasonal import seasonal_decompose
plt.rc("figure",figsize=(16,8))

def decompose(df, i):
  series = df.iloc[:,i]
  result = seasonal_decompose(series, model='additive',extrapolate_trend='freq',period=12)
  result.plot()
  plt.title('Decomposition of '+ df.columns.values.tolist()[i])
  plt.show()
  return result
result = decompose(df, 0)

from statsmodels.tsa.stattools import adfuller

def ADF_Test(df,df_adf,i):	
	series = df.iloc[:,i]
	X = series.values
	result = adfuller(X)
	df_adf_i = pd.DataFrame({'Output':df.columns.values.tolist()[i],'ADF Statistic':result[0],'p-value':result[1],'CriticalValues':result[4]})
	df_adf = df_adf._append(df_adf_i,ignore_index = True)
	return df_adf

df_adf = pd.DataFrame()
df_adf = ADF_Test(df,df_adf,0)
print(df_adf)

## Removing seasonality to make data stationary

df_nonseasonal = pd.DataFrame()

# Option 1: Use fillna
df_nonseasonal['CPI'] = df['CPI'] - df.shift(12)['CPI'].fillna(0)

df_adf = pd.DataFrame()
df_adf = ADF_Test(df_nonseasonal,df_adf,0)
print(df_adf)

## Subtracting Lags to make data stationary

# Option 1: Replace np.nan
df_nonseasonal['lag1'] = df_nonseasonal['CPI'] - df_nonseasonal.shift(1).replace(np.nan, 0)['CPI']

# Option 2: Using fillna to replace NaN values with 0
df_nonseasonal['lag1'] = df_nonseasonal['CPI'] - df_nonseasonal.shift(1)['CPI'].fillna(0)

df_adf = pd.DataFrame()
df_adf = ADF_Test(pd.DataFrame(df_nonseasonal['lag1']),df_adf,0)
print(df_adf)


#### SARIMA MODEL ###

from statsmodels.tsa.statespace.sarimax import SARIMAX

SARIMA_train = df.iloc[:int(0.9*df.shape[0]),:]
SARIMA_test = df.iloc[int(0.9*df.shape[0]):,:]

my_order = (0,1,1)
my_seasonal_order = (0, 1, 1, 12)

SARIMA_model = SARIMAX(SARIMA_train, order=my_order, seasonal_order=my_seasonal_order)

SARIMA_model_fit = SARIMA_model.fit()
print(SARIMA_model_fit.summary())


SARIMA_predictions = SARIMA_model_fit.predict(start = SARIMA_train.index[0], end = SARIMA_train.index[-1])
plt.figure(figsize=(10,4))
plt.plot(SARIMA_train)
plt.plot(SARIMA_predictions)
plt.legend(('Data', 'Predictions'), fontsize=16)
plt.title('Predictions for training data from SARIMA Model', fontsize=20)
plt.ylabel('CPI', fontsize=16)
plt.axhline(0, color='r', linestyle='--', alpha=0.2)



SARIMA_residuals = SARIMA_train['CPI'] - pd.DataFrame(SARIMA_predictions)['predicted_mean']
plt.figure(figsize=(10,4))
plt.plot(SARIMA_residuals)
plt.title('Residuals from SARIMA Model', fontsize=20)
plt.ylabel('Error', fontsize=16)
plt.axhline(0, color='r', linestyle='--', alpha=0.2)

print('Mean Absolute Error:', round(np.mean(abs(SARIMA_residuals)),4))
print('\nRoot Mean Squared Error:', np.sqrt(np.mean(SARIMA_residuals**2)))



#residuals analysis
residuals = [SARIMA_train.iloc[i,:]-SARIMA_predictions[i] for i in
range(len(SARIMA_predictions))]
residuals = pd.DataFrame(residuals)
residuals.head()



SARIMA_predictions_test = SARIMA_model_fit.predict(start = SARIMA_test.index[0], end = SARIMA_test.index[-1])
SARIMA_residuals_test = SARIMA_test['CPI'] - pd.DataFrame(SARIMA_predictions_test)['predicted_mean']
     


plt.figure(figsize=(10,4))

plt.plot(SARIMA_test)
plt.plot(SARIMA_predictions_test)

plt.legend(('Data', 'Predictions'), fontsize=16)

plt.title('CPI Predictions on test data', fontsize=20)
plt.ylabel('CPI', fontsize=16)


print('Mean Absolute Error:', round(np.mean(abs(SARIMA_residuals_test)),4))
print('\nRoot Mean Squared Error:', np.sqrt(np.mean(SARIMA_residuals_test**2)))


#### ROLLING FORECAST ####


from pandas.core.groupby.groupby import Timestamp
from dateutil.relativedelta import relativedelta
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

start_time = datetime(1913,1,1)
end_time = datetime(2025,1,1)



my_order = (0,1,1)
my_seasonal_order = (0, 1, 1, 12)

roll_end_time = end_time
SARIMA_roll_train = df.loc[start_time:end_time]

while roll_end_time+relativedelta(months=+1) in df.index:
  
  SARIMA_roll_test = df.loc[roll_end_time:roll_end_time+relativedelta(months=+0)]
  SARIMA_roll_model = SARIMAX(SARIMA_roll_train, order=my_order, seasonal_order=my_seasonal_order)
  SARIMA_roll_model_fit = SARIMA_roll_model.fit()
  SARIMA_roll_predictions = SARIMA_roll_model_fit.predict(start = SARIMA_roll_test.index[0], end = SARIMA_roll_test.index[-1])

  print(roll_end_time)
  roll_end_time = roll_end_time + relativedelta(months=+1)
  SARIMA_roll_train.loc[roll_end_time] = float(SARIMA_roll_predictions)



  
plt.figure(figsize=(10,4))

plt.plot(df.loc[end_time:])
plt.plot(SARIMA_roll_train.loc[end_time:])

plt.legend(('Data', 'Predictions'), fontsize=16)

plt.title('CPI Predictions on Test Data', fontsize=20)
plt.ylabel('CPI', fontsize=16)




SARIMA_roll_residuals = df.loc[end_time:] - SARIMA_roll_train.loc[end_time:]
plt.figure(figsize=(10,4))
plt.plot(SARIMA_roll_residuals)
plt.title('Residuals from SARIMA Model', fontsize=20)
plt.ylabel('Error', fontsize=16)
plt.axhline(0, color='r', linestyle='--', alpha=0.2)


print('Mean Absolute Error:', round(np.mean(abs(SARIMA_roll_residuals['CPI'])),4))
print('\nRoot Mean Squared Error:', np.sqrt(np.mean(SARIMA_roll_residuals**2)))

