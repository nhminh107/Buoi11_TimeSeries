import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def create_recursive_data(data, window_size):
    i = 1
    while(i < window_size):
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i += 1
    data["target"] = data["co2"].shift(-i)
    data = data.dropna(axis=0)
    return data

def create_direct_data(data, window_size, target_size):
    i = 1
    while (i < window_size):
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i += 1

    i = 0
    while i < window_size:
        data["target_{}".format(i)] = data["co2"].shift(-i-window_size)
        i += 1

    data = data.dropna(axis = 0)
    return data


data = pd.read_csv('co2.csv')
data["time"] = pd.to_datetime(data["time"])
data["co2"] = data["co2"].interpolate()
"""fig, ax = plt.subplots()
ax.plot(data["time"], data["co2"])
ax.set_xlabel("Time")
ax.set_ylabel("CO2 Consumption")
plt.show()"""

target_size = 3
data = create_direct_data(data, 5, target_size)

x = data.drop(["time"] + ["target_{}".format(i) for i in range(target_size)], axis = 1)
y = data[["target_{}".format(i) for i in range(target_size)]]

train_size = 0.8
num_samples = len(x)

x_train = x[:int(num_samples*train_size)]
y_train = y[:int(num_samples*train_size)]
x_test = x[int(num_samples*train_size):]
y_test = y[int(num_samples*train_size):]

regs = [LinearRegression() for _ in range(target_size)]
for i, reg in enumerate(regs):
    reg.fit(x_train, y_train[f'target_{i}'])

r2 = []
mae = []
mse = []

for i, reg in enumerate(regs):
    y_predict = reg.predict(x_test)
    r2.append(r2_score(y_test[f"target_{i}"], y_predict))
    mae.append(mean_absolute_error(y_test[f"target_{i}"], y_predict))
    mse.append(mean_squared_error(y_test[f"target_{i}"], y_predict))

print("R2 score {}".format(r2))
print("Mean abs err {}".format(mae))
print("Mean sqr err {}".format(mse))

