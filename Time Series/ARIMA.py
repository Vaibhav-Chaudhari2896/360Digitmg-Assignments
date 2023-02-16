import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot

Walmart = pd.read_csv("C:/Data/forecasting/Walmart Footfalls Raw.csv")
# Data Partition
Train = Walmart.head(147)
Test = Walmart.tail(12)

tsa_plots.plot_acf(Walmart.Footfalls, lags = 12)
tsa_plots.plot_pacf(Walmart.Footfalls,lags = 12)


# ARIMA with AR=4, MA = 6
model1 = ARIMA(Train.Footfalls, order = (12,1,6))
res1 = model1.fit()
print(res1.summary())

# Forecast for next 12 months
start_index = len(Train)
end_index = start_index + 11
forecast_test = res1.predict(start=start_index, end=end_index)

print(forecast_test)

# Evaluate forecasts
rmse_test = sqrt(mean_squared_error(Test.Footfalls, forecast_test))
print('Test RMSE: %.3f' % rmse_test)

# plot forecasts against actual outcomes
pyplot.plot(Test.Footfalls)
pyplot.plot(forecast_test, color='red')
pyplot.show()


# Auto-ARIMA - Automatically discover the optimal order for an ARIMA model.
# pip install pmdarima --user
import pmdarima as pm

ar_model = pm.auto_arima(Train.Footfalls, start_p=0, start_q=0,
                      max_p=12, max_q=12, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, trace=True,
                      error_action='warn', stepwise=True)


# Best Parameters ARIMA
# ARIMA with AR=3, I = 1, MA = 5
model = ARIMA(Train.Footfalls, order = (3,1,5))
res = model.fit()
print(res.summary())

# Forecast for next 12 months
start_index = len(Train)
end_index = start_index + 11
forecast_best = res1.predict(start=start_index, end=end_index)

print(forecast_best)

# Evaluate forecasts
rmse_best = sqrt(mean_squared_error(Test.Footfalls, forecast_best))
print('Test RMSE: %.3f' % rmse_best)

# plot forecasts against actual outcomes
pyplot.plot(Test.Footfalls)
pyplot.plot(forecast_best, color='red')
pyplot.show()


# Forecast for future 12 months
start_index = len(Walmart)
end_index = start_index + 11
forecast = res1.predict(start=start_index, end=end_index)

print(forecast)
