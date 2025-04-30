# Акимов Павло 
# Білет № 7
# Завдання 3 

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Завантаження датасету з правильним роздільником
data = pd.read_csv('titanic3.csv', sep=';', quotechar='"', encoding='utf-8')

# Вибір ознак для кластеризації
features = ['pclass', 'age', 'fare', 'sex']
data = data[features].copy()

# Перевірка пропущених значень перед обробкою
print("Пропущені значення перед обробкою:")
print(data.isna().sum())
data['sex'] = data['sex'].map({'male': 0, 'female': 1})

# Перевірка, чи є пропущені значення в 'sex' після кодування
if data['sex'].isna().sum() > 0:
    print("Пропущені значення в 'sex' після кодування. Заповнюємо модою.")
    sex_imputer = SimpleImputer(strategy='most_frequent')
    data['sex'] = sex_imputer.fit_transform(data[['sex']])

# Перетворення значень age і fare, які містять кому, на числові
data['age'] = pd.to_numeric(data['age'].str.replace(',', '.'), errors='coerce')
data['fare'] = pd.to_numeric(data['fare'].str.replace(',', '.'), errors='coerce')

# Обробка пропущених значень для всіх числових стовпців
imputer = SimpleImputer(strategy='mean')
data[['pclass', 'age', 'fare']] = imputer.fit_transform(data[['pclass', 'age', 'fare']])

# Перевірка пропущених значень після імпутації
print("Пропущені значення після обробки:")
print(data.isna().sum())

# Стандартизація даних
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
if np.any(np.isnan(scaled_data)):
    print("У scaled_data є NaN. Перевірте вхідні дані.")
    raise ValueError("Знайдено NaN у scaled_data")

# Метод ліктя для визначення кількості кластерів
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Візуалізація методу ліктя
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Кількість кластерів')
plt.ylabel('Інерція')
plt.title('Метод ліктя для вибору кількості кластерів')
plt.savefig('elbow_plot.png')
plt.close()

# Кластеризація з оптимальною кількістю кластерів (k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled_data)

# Аналіз результатів
print("Середні значення ознак для кожного кластера:")
print(data.groupby('cluster').mean())

# Візуалізація кластерів (age vs fare)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='age', y='fare', hue='cluster', palette='viridis', style='sex', size='pclass')
plt.xlabel('Вік')
plt.ylabel('Вартість квитка')
plt.title('Кластери пасажирів Titanic (Вік vs Вартість квитка)')
plt.legend(title='Кластер')
plt.savefig('cluster_plot.png')
plt.close()