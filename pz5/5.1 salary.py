import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Читаємо дані з CSV
дані = pd.read_csv('Salary_Data.csv')

# Розділяємо на X і y
X = дані[['YearsExperience']]
y = дані['Salary']

# Створюємо модель лінійної регресії
модель = LinearRegression()
модель.fit(X, y)

# Прогнозуємо значення
y_прогноз = модель.predict(X)

# Виводимо коефіцієнти
print(f'Коефіцієнт (нахил): {модель.coef_[0]}')
print(f'Вільний член (перетин): {модель.intercept_}')

# Візуалізація
plt.scatter(X, y, color='blue', label='Фактичні дані')
plt.plot(X, y_прогноз, color='red', label='Лінія регресії')
plt.xlabel('Досвід (YearsExperience)')
plt.ylabel('Зарплата (Salary)')
plt.legend()
plt.title('Залежність зарплати від досвіду')
plt.show()