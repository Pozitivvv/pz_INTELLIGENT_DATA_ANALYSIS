import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Завантаження даних
file_path = "churn-bigml-20.csv"
df = pd.read_csv(file_path)

# Вибір необхідних колонок для кластеризації (без категоріальних змінних)
numeric_columns = [
    "Account length", "Area code", "Number vmail messages",
    "Total day minutes", "Total day calls", "Total day charge",
    "Total eve minutes", "Total eve calls", "Total eve charge",
    "Total night minutes", "Total night calls", "Total night charge",
    "Total intl minutes", "Total intl calls", "Total intl charge",
    "Customer service calls"
]

df_numeric = df[numeric_columns]

# Масштабування даних
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Визначення оптимальної кількості кластерів методом "лікоть"
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=0, n_init=10)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# Побудова графіка методу "лікоть"
plt.plot(range(1, 10), wcss, marker='o', linestyle='-')
plt.xlabel('Кількість кластерів')
plt.ylabel('WCSS (Внутрішньокластерна сума квадратів)')
plt.title('Метод "лікоть" для визначення кількості кластерів')
plt.show()


# optimal_clusters = 4
# kmeans = KMeans(n_clusters=optimal_clusters, random_state=0, n_init=10)
# df['Cluster'] = kmeans.fit_predict(df_scaled)

# plt.figure(figsize=(8, 6))
# plt.scatter(df["Total day minutes"], df["Total eve minutes"], c=df["Cluster"], cmap="viridis", alpha=0.7)
# plt.xlabel("Total day minutes")
# plt.ylabel("Total eve minutes")
# plt.title("Розподіл кластерів за дзвінками вдень та ввечері")
# plt.colorbar(label="Кластер")
# plt.show()
