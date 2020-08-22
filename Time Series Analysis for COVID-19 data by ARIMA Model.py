import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
from sklearn.metrics import mean_squared_error
from pylab import rcParams


matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

df_confirmed_cases = pd.read_csv("time_series-ncov-Confirmed.csv",
                 parse_dates=["Date"],
                 index_col="Date")
df_death_cases = pd.read_csv("time_series-ncov-Deaths.csv",
                 parse_dates=["Date"],
                 index_col="Date")
df_recovery_cases = pd.read_csv("time_series-ncov-Recovered.csv",
                 parse_dates=["Date"],
                 index_col="Date")
print("Confirmed Cases Dataset Head : ")
print(df_confirmed_cases.head())
print("Confirmed Cases Dataset Index : ")
print(df_confirmed_cases.index)
print("Death Cases Dataset Head : ")
print(df_death_cases.head())
print("Death Cases Dataset Index : ")
print(df_death_cases.index)
print("Recovery Cases Dataset Head : ")
print(df_recovery_cases.head())
print("Recovery Cases Dataset Index : ")
print(df_recovery_cases.index)

y_confirmed_cases = df_confirmed_cases['Value'].resample('D').mean()
y_confirmed_cases.plot(figsize=(15, 6))
plt.show()
y_death_cases = df_death_cases['Value'].resample('D').mean()
y_death_cases.plot(figsize=(15, 6))
plt.show()
y_recovery_cases = df_recovery_cases['Value'].resample('D').mean()
y_recovery_cases.plot(figsize=(15, 6))
plt.show()

rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y_confirmed_cases, model='additive')
fig = decomposition.plot()
plt.show()

rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y_death_cases, model='additive')
fig = decomposition.plot()
plt.show()

rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y_recovery_cases, model='additive')
fig = decomposition.plot()
plt.show()

#Time series forecasting with ARIMA
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y_confirmed_cases,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

#Fitting the ARIMA model
mod = sm.tsa.statespace.SARIMAX(y_confirmed_cases,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))
plt.show()

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y_death_cases,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

#Fitting the ARIMA model
mod = sm.tsa.statespace.SARIMAX(y_death_cases,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))
plt.show()

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y_recovery_cases,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

#Fitting the ARIMA model
mod = sm.tsa.statespace.SARIMAX(y_recovery_cases,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))
plt.show()

#Validating forecasts
pred = results.get_prediction(start=pd.to_datetime('2020-03-22'), dynamic=False)
pred_ci = pred.conf_int()
ax = y_confirmed_cases['2020':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Confirmed case of COVID-19')
plt.legend()
plt.show()

#Validating forecasts
pred = results.get_prediction(start=pd.to_datetime('2020-03-22'), dynamic=False)
pred_ci = pred.conf_int()
ax = y_death_cases['2020':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Confirmed case of COVID-19')
plt.legend()
plt.show()

#Validating forecasts
pred = results.get_prediction(start=pd.to_datetime('2020-03-22'), dynamic=False)
pred_ci = pred.conf_int()
ax = y_recovery_cases['2020':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Confirmed case of COVID-19')
plt.legend()
plt.show()

y_confirmed_forecasted = pred.predicted_mean
y_confirmed_truth = y_confirmed_cases['2020-03-22':]
mse = mean_squared_error(y_confirmed_truth, y_confirmed_forecasted)
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

y_death_forecasted = pred.predicted_mean
y_death_truth = y_death_cases['2020-03-22':]
mse = mean_squared_error(y_death_truth, y_death_forecasted)
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

y_recovery_forecasted = pred.predicted_mean
y_recovery_truth = y_recovery_cases['2020-03-22':]
mse = mean_squared_error(y_recovery_truth, y_recovery_forecasted)
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

#Producing and visualizing forecasts
pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y_confirmed_cases.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Death case of COVID-19')
plt.legend()
plt.show()

#Producing and visualizing forecasts
pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y_death_cases.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Confirmed case of COVID-19')
plt.legend()
plt.show()

#Producing and visualizing forecasts
pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y_recovery_cases.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Recovered case of COVID-19')
plt.legend()
plt.show()

df_recovery_cases = pd.DataFrame({'Date':y_recovery_cases.index, 'Value':y_recovery_cases.values})
df_death_cases = pd.DataFrame({'Date': y_death_cases.index, 'Value': y_death_cases.values})
forcast = df_recovery_cases.merge(df_death_cases, how='inner', on='Date')
forcast.rename(columns={'Value_x': 'df_recovery_cases_Value', 'Value_y': 'df_death_cases_Value'}, inplace=True)
forcast.head()

first_date_death_greater_then_recovery = forcast.ix[np.min(list(np.where(forcast['df_death_cases_Value'] > forcast['df_recovery_cases_Value'])[0])), 'Date']
print("Death Cases first time produced greater then recovered cases at {}.".format(first_date_death_greater_then_recovery.date()))

first_date_recovery_greater_then_death = forcast.ix[np.min(list(np.where(forcast['df_death_cases_Value'] < forcast['df_recovery_cases_Value'])[0])), 'Date']
print("Recovered Cases first time produced greater then death cases at {}.".format(first_date_death_greater_then_recovery.date()))
