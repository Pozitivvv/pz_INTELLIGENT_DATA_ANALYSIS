import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Генерація штучного часового ряду з сезонністю
np.random.seed(42)
date_range = pd.date_range(start='2015-01-01', periods=120, freq='M')
seasonal = 10 * np.sin(2 * np.pi * date_range.month / 12)
trend = np.linspace(0, 20, 120)
noise = np.random.normal(scale=2, size=120)
data = seasonal + trend + noise
df = pd.DataFrame({'value': data}, index=date_range)

# Візуалізація ряду
plt.figure(figsize=(10, 4))
plt.plot(df, label='Time Series')
plt.title("Штучний часовий ряд")
plt.xlabel("Дата")
plt.ylabel("Значення")
plt.legend()
plt.show()

# Перевірка на стаціонарність
adf_result = adfuller(df['value'])
print(f"ADF-statistic: {adf_result[0]:.3f}")
print(f"p-value: {adf_result[1]:.3f}")
if adf_result[1] > 0.05:
    print("Ряд нестаціонарний, диференціюємо...")
    df_diff = df.diff().dropna()
else:
    print("Ряд стаціонарний")
    df_diff = df

# Поділ на train/test
train = df_diff.iloc[:-12]
test = df_diff.iloc[-12:]

# SARIMA модель: (p,d,q)(P,D,Q,s)
model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
model_fit = model.fit(disp=False)

# Прогноз
forecast = model_fit.forecast(steps=12)

# Повертаємо диференційований прогноз до початкового масштабу
last_real = df.iloc[-13]['value']
forecast_cumsum = forecast.cumsum() + last_real

# Побудова графіку
plt.figure(figsize=(10, 4))
plt.plot(df.index[-24:], df['value'].iloc[-24:], label='Факт')
plt.plot(forecast_cumsum.index, forecast_cumsum, label='Прогноз', linestyle='--')
plt.title("SARIMA: Прогноз vs Факт")
plt.xlabel("Дата")
plt.ylabel("Значення")
plt.legend()
plt.show()

# Метрики
actual = df['value'].iloc[-12:]
predicted = forecast_cumsum
print("MAE:", mean_absolute_error(actual, predicted))
print("MSE:", mean_squared_error(actual, predicted))
