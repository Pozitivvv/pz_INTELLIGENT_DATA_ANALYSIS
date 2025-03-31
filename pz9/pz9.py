import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from mpl_toolkits.mplot3d import Axes3D

basket = pd.read_csv("Groceries_dataset.csv")

basket['itemDescription'] = basket['itemDescription'].apply(lambda x: [x])  
basket = basket.groupby(['Member_number', 'Date'])['itemDescription'].sum().reset_index(drop=True)

encoder = TransactionEncoder()
transactions = encoder.fit(basket).transform(basket)
transactions = pd.DataFrame(transactions, columns=encoder.columns_)

frequent_itemsets = apriori(transactions, min_support=6/len(basket), use_colnames=True, max_len=2)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)

print(rules.head())

sns.set(style="whitegrid")

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

x = rules['support']
y = rules['confidence']
z = rules['lift']

ax.set_xlabel("Support")
ax.set_ylabel("Confidence")
ax.set_zlabel("Lift")

ax.scatter(x, y, z, c='b', marker='o')

ax.set_title("3D Distribution of Association Rules")

plt.show()
