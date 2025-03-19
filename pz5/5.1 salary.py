import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Читаємо дані з CSV
data = pd.read_csv('Salary_Data.csv')

# Розділяємо на X і y
X = data[['YearsExperience']]
y = data['Salary']

# Створюємо модель лінійної регресії
model = LinearRegression()
model.fit(X, y)

# Прогнозуємо значення
y_pred = model.predict(X)

# Виводимо коефіцієнти
print(f'Коефіцієнт (нахил): {model.coef_[0]}')
print(f'Вільний (перетин): {model.intercept_}')

# Візуалізація
plt.scatter(X, y, color='blue', label='Фактичні дані')
plt.plot(X, y_pred, color='red', label='Лінія регресії')
plt.xlabel('Досвід (YearsExperience)')
plt.ylabel('Зарплата (Salary)')
plt.legend()
plt.title('Залежність зарплати від досвіду')
plt.show()