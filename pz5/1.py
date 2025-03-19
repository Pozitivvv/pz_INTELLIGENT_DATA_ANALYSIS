import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Вхідні дані
data = {
    'Марка': ['Acura', 'Audi', 'BMW', 'Buick', 'Corvette', 'Chrysler', 'Dodge', 'Eagle', 'Ford', 'Honda', 
              'Isuzu', 'Mazda', 'Mercedes', 'Minsub', 'Nissan', 'Olds', 'Pontiac', 'Porsche', 'Saab', 
              'Toyota', 'VW', 'Volvo'],
    'Вартість': [0.521, 0.866, 0.496, 0.614, 1.235, 0.614, 0.706, 0.614, 0.706, 0.429, 
                 0.798, 0.126, 1.051, 0.614, 0.429, 0.614, 0.614, 3.454, 0.588, 0.059, 0.706, 0.219],
    'Вік водія': [25, 24, 29, 50, 62, 43, 26, 20, 54, 38, 27, 51, 46, 23, 31, 45, 40, 41, 29, 36, 38, 42]
}

df = pd.DataFrame(data)

# Вибираємо параметри для кластеризації
X = df[['Вартість', 'Вік водія']].values

# Обчислюємо евклідову відстань і виконуємо ієрархічну кластеризацію
Z = linkage(X, method='single', metric='euclidean')

# Візуалізація дендрограми
plt.figure(figsize=(12, 6))
dendrogram(Z, labels=df['Марка'].values, leaf_rotation=90)
plt.title('Дендрограма кластеризації автомобілів')
plt.xlabel('Марка автомобіля')
plt.ylabel('Евклідова відстань')
plt.show()