import matplotlib.pyplot as plt

# # Прикладні дані (продажі за період)
# time = range(1101, 1113)
# beer_sales = range(150, 1950, 150)
# diaper_sales = range(200, 2600, 200)

# plt.figure(figsize=(8, 5))
# plt.plot(time, beer_sales, label='Пиво', marker='o')
# plt.plot(time, diaper_sales, label='Підгузки', marker='s')
# plt.title('Продажі пива і підгузків з часом')
# plt.xlabel('Час')
# plt.ylabel('Продажі')
# plt.legend()
# plt.grid(True)
# plt.show()

fig, ax1 = plt.subplots(figsize=(8, 5))

time = range(1101, 1113)
iphone11_sales = range(200, 2600, 200)
iphone10_sales = range(1950, 150, -150)

# Перша вісь Y для iPhone 11
ax1.set_xlabel('Час')
ax1.set_ylabel('iPhone 11', color='blue')
ax1.plot(time, iphone11_sales, color='blue', marker='o', label='iPhone 11')
ax1.tick_params(axis='y', labelcolor='blue')

# Друга вісь Y для iPhone 10
ax2 = ax1.twinx()
ax2.set_ylabel('iPhone 10', color='red')
ax2.plot(time, iphone10_sales, color='red', linestyle='--', marker='s', label='iPhone 10')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Порівняння продажів: iPhone 11 vs iPhone 10')
fig.tight_layout()
plt.show()
