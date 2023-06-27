import numpy as np
from sklearn.tree import DecisionTreeRegressor
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Generate sample data
# Assume you have input data with shape (num_samples, 3) and target variable with shape (num_samples, )
input_data = np.random.randn(100, 3)
target = 2 * input_data[:, 0] + 3 * input_data[:, 1] - 4 * input_data[:, 2] + np.random.randn(100)

# Create and fit the decision tree regression model
model = DecisionTreeRegressor()
model.fit(input_data, target)

# Generate new data for visualization
x = np.linspace(-1, 1, 20)
y = np.linspace(-1, 1, 20)
z = np.linspace(-1, 1, 20)
x, y, z = np.meshgrid(x, y, z)
data = np.column_stack((x.flatten(), y.flatten(), z.flatten()))

# Make predictions on new data
predictions = model.predict(data)

# Visualize the results
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(input_data[:, 0], input_data[:, 1], input_data[:, 2], c=target, cmap='viridis')
ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=predictions, cmap='Reds', alpha=0.2)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
print('end')
