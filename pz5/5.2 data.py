import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Завантажуємо CSV 
дані = pd.read_csv('data.csv', header=None, names=['X', 'y'])

# Розділяємо дані
X = дані[['X']]
y = дані['y']

# Створюємо та навчаємо модель
модель = LinearRegression()
модель.fit(X, y)

# Робимо прогноз
y_прогноз = модель.predict(X)

# Виводимо коефіцієнти
print(f'Коефіцієнт (нахил): {модель.coef_[0]}')
print(f'Вільний член (перетин): {модель.intercept_}')
print(f'Оцінка R^2: {модель.score(X, y)}')

# Візуалізація
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Вихідні дані')
plt.plot(X, y_прогноз, color='red', linewidth=2, label='Лінія регресії')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Лінійна регресія за data.csv')
plt.legend()
plt.grid(True)
plt.show()