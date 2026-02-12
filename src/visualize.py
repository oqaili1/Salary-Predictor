import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import load_data, gradient_descent, predict

# Load dataset
x, y = load_data('../data/Salary_Data.csv')

# Train model
w, b, j_hist = gradient_descent(x, y, 0, 0, 0.01, 1000)


# Plot 1: Cost vs Iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))

# First 100 iterations
ax1.plot(j_hist[:100])
ax1.set_title('Cost vs Iteration (start)')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Cost')

# Last iterations
ax2.plot(100 + np.arange(len(j_hist[100:])), j_hist[100:])
ax2.set_title('Cost vs Iteration (end)')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Cost')

plt.tight_layout()
plt.savefig('../images/cost_plot.png', dpi=150)
plt.show()
print('Saved: cost_plot.png')


# Plot 2: Regression
# Scatter plot of actual data
plt.scatter(x, y, marker='x', c='b', label='Actual data', alpha=0.7)

# Regression line
x_line = np.linspace(x.min(), x.max(), 100)
y_line = predict(x_line, w, b)
plt.plot(x_line, y_line, c='r', linewidth=2, label=f"Prediction: y = {w:.2f}x + {b:.2f}")

plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.title('Salary prediction using linear regression')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('../images/regression_line.png', dpi=150)
plt.show()
print('Saved: regression_line.png')

# Plot 3: Prediction example
plt.figure(figsize=(10, 6))

plt.scatter(x, y, c='b', label='Training data', alpha=0.7)
plt.plot(x_line, y_line, c='r', linewidth=2, label='Model')

# Highlight a prediction
test_years = 7
test_salary = predict(test_years, w, b)
plt.scatter([test_years], [test_salary], c='g', s=200, zorder=5, label=f"Prediction: {test_years} years -> ${test_salary:,.0f}")
plt.axvline(x=test_years, c='g', linestyle='--', alpha=0.5)
plt.axhline(y=test_salary, c='g', linestyle='--', alpha=0.5)

plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.title('Salary prediction example')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('../images/prediction_example.png', dpi=150)
plt.show()
print('Saved: prediction_example.png')