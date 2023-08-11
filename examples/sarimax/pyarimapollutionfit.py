import os

import pandas as pd
import numpy as np
import pmdarima as pmda
import math
import matplotlib as mpl
from matplotlib import pyplot as plt
from pmdarima import model_selection
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from pmdarima.arima import auto_arima
from sklearn.metrics import mean_absolute_error
from pmdarima.metrics import smape

import mlflow.pmdarima
from mlflow.models import infer_signature, Model, ModelSignature


# Directory where mlflow artifacts will be stored.
ARTIFACT_PATH = "pmdarimafit"
polldata = pd.read_csv("/Users/stephenweller/Downloads/LSTM-Multivariate_pollution.csv")

# Split data into training and test datasets, with 80% used for training.
trainobs = int(len(polldata) * 0.8)
testobs = int(len(polldata) - trainobs)

# Make sure that consecutive observations are used for train and test datasets.
polltrain = polldata[:trainobs]
polltest = polldata[trainobs:len(polldata)]

polltrain['date'] = pd.to_datetime(polltrain.date)
polltest['date'] = pd.to_datetime(polltest.date)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(polltrain['date'], polltrain['pollution'])
plt.title('Plot of Beijing Pollution over time')
plt.xlabel('Month/Year')
plt.ylabel('Pollution')
plt.savefig("Pollutionplot.png")

# Now let's plot the partial autcorrelation function.
fig, ax = plt.subplots(figsize=(8,6))
plot_pacf(polltrain.pollution, ax=ax, lags=24)
plt.savefig("Pollutionpacfplot.png")

# Let's check for 'stationarity' in the time series.
def check_stationarity(series, signif=0.05, name='', verbose=False):
    result = adfuller(series.values)
    print(f'     Augmented Dickey-Fuller Test on "{name}"', "\n  "'-'*47)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s %.3f' % (key, value))
    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon_Stationary\x1b[0m")   

#ADF Test on each column
for name, column in polltrain.drop(columns=['date','wnd_dir']).iteritems():
    check_stationarity(column, name=column.name)
    print('\n')

# Now fit a multivariate sarimax model to the pollution data.
# We will try an initial sarimax model with p=1, q=1 and D=1 for the seasonal term.
# Use a stepwise method to find the best fitting model. The parameter 'P' here is the 
# seasonality parameter. The data has 'hourly' measurements, so we will use a frequency
# of 8,760 observations per year or 24 observations per day.

with mlflow.start_run():
   arima = pmda.auto_arima(polltrain['pollution'], X=polltrain.drop(columns=['date', 'wnd_dir', 'pollution']), d=2, start_P=1, start_q=1, max_p=3, max_q=3, m=24,
                        error_action='ignore', trace=True, suppress_warnings=True, maxiter=500, test='adf', stationary=True, seasonal=True, stepwise=True)

print("Model trained. \nExtracting parameters...")
parameters = arima.get_params(deep=True)
metrics = {x: getattr(arima, x)() for x in ["aicc", "aic", "bic", "hqic", "oob"]}

# Summary output for arima model.
print(arima.summary())

polltrainnew = polltrain.drop(columns=['date','wnd_dir'])

# Another test for time series stationarity
def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(dftest[0:4], index=["Test Statistic",
                                             "p-value",
                                             '#Lags Use',
                                             "Number of Observations Used",
                                             ],)
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)

adf_test(polltrainnew["pollution"])

model = arima

# Compute predictions on new data.
fc, conf_int = arima.predict(n_periods=len(polltest), X = polltest.drop(columns=['date','wnd_dir','pollution']), return_conf_int=True)
 
print(f"Mean squared error: {mean_squared_error(polltest['pollution'], fc)}")   
print(f"SMAPE: {smape(polltest['pollution'], fc)}")

signature = infer_signature(polltrain, fc)
mlflow.pmdarima.log_model(
        pmdarima_model=arima, artifact_path=ARTIFACT_PATH, signature=signature
)

mlflow.log_params(parameters)
mlflow.log_metrics(metrics)
model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)

print(f"Model artifact logged to: {model_uri}")

loaded_model = mlflow.pmdarima.load_model(model_uri)
forecast = loaded_model.predict(n_periods=len(polltest), exogenous = polltest.drop(columns=['date','wnd_dir','pollution']))

print(f"Forecast: \n{forecast}")

plt.figure(figsize=(15,5))
plt.grid()
plt.plot(polltrain['date'][:len(polltest)], polltest['pollution'], marker='o', label="Test")
plt.plot(polltrain['date'][:len(polltest)], fc, color='green', marker='v', label='Prediction')
plt.fill_between(polltest.index, conf_int[:, 0], conf_int[:, 1], alpha=0.9, color='orange', label="Confidence Intervals")
plt.legend()
plt.savefig("POLLPREDPLOT.PNG")


















