# 💼 Salary Predictor using Linear Regression

A machine learning project that predicts salary based on years of experience using Linear Regression implemented from scratch.

![Regression Line](images/regression_line.png)

## 📌 Overview

This project demonstrates:
- Linear Regression from scratch (without sklearn)
- Gradient Descent optimization
- Cost function implementation
- Data visualization

## 📊 Dataset

- **Source**: [Kaggle - Salary Dataset](https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression)
- **Features**: Years of Experience
- **Target**: Salary
- **Samples**: 30

## 🧠 Model

The model uses the equation:
Salary = w × YearsExperience + b


Where:
- `w` = weight (slope)
- `b` = bias (intercept)

### Training Results

| Parameter | Value |
|-----------|-------|
| Weight (w) | 9449.96 |
| Bias (b) | 25792.20 |
| Iterations | 1000 |
| Learning Rate | 0.01 |

## 📈 Visualizations

### Cost vs. Iterations
![Cost Plot](images/cost_plot.png)

### Prediction Example
![Prediction](images/prediction_example.png)

## 🚀 How to Run

1. **Clone the repository**
   git clone https://github.com/oqaili1/salary-predictor.git
   cd salary-predictor
   
2. **Create virtual envirnoment**
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies**
    pip install -r requirements.txt
    
4. **Run the model**
    cd src
     python model.py


🛠️ Built With

    Python
    NumPy
    Pandas
    Matplotlib

👤 Author

    GitHub: @oqaili1

📄 License

This project is licensed under the MIT License.

