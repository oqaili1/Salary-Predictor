import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Load and prepare dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    x = df['YearsExperience'].values
    y = df['Salary'].values
    return x, y


# Cost function
def cost_function(x, y, w, b):
    m = x.shape[0]
    f_wb = w * x + b
    cost = np.sum((f_wb - y) ** 2) / (2 * m)
    return cost


# Gradient function
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    f_wb = w * x + b
    dj_dw = np.sum((f_wb - y) * x) / m
    dj_db = np.sum(f_wb - y) / m
    return dj_dw, dj_db


# Gradient descent
def gradient_descent(x, y, w_in, b_in, alpha, n_iters):
    w = w_in
    b = b_in
    j_hist = []

    for i in range(n_iters):
        # Calculate gradients
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # Update parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost
        cost = cost_function(x, y, w, b)
        j_hist.append(cost)

        if i % math.ceil(n_iters / 10) == 0:
            print(f"Iteration {i:5}: Cost {cost:.4f}, w: {w:.4f}, b: {b:.4f}")
    
    return w, b, j_hist



# Prediction function
def predict(x, w, b):
    result = w * x + b

    if isinstance(x, (int, float)):
        return float(result)
    
    return result



# Main execution
if __name__ == "__main__":
    # Load dataset
    print('Loading data...')

    x, y = load_data('../data/Salary_Data.csv')
    print(f"Loaded {len(x)} samples \n")

    # Initialize parameter
    w_init = 0
    b_init = 0
    alpha = 1e-2
    num_iters = 1000

    # Train model
    print('Training model...')
    print('-' * 50)
    w, b, j_history = gradient_descent(x, y, w_init, b_init, alpha, num_iters)
    print('-' * 50)
    print(f"Final parameters: w = {w:.4f}, b = {b:.4f}")

    # Make a prediction
    years = 5
    predicted_salary = predict(years, w, b)
    print(f"\n Prediction: {years} years experience -> ${float(predicted_salary):,.2f} salary")