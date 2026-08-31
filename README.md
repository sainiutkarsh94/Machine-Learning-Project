# Forest Fire Weather Index Prediction

This project is a Machine Learning web application that predicts the Fire Weather Index (FWI) using meteorological and environmental parameters from the Algerian Forest Fire dataset. The application uses a Ridge Regression model trained on selected features after preprocessing and feature engineering.

The model takes inputs such as temperature, relative humidity, wind speed, rainfall, FFMC, DMC, ISI, classes, and region to estimate the Fire Weather Index. Feature scaling is performed using StandardScaler to improve model performance.

A user-friendly web interface built with Flask, HTML, and CSS allows users to enter the required parameters and obtain predictions in real time.

## Key Features

- Fire Weather Index prediction using Ridge Regression
- Data preprocessing and feature selection
- Feature scaling using StandardScaler
- Interactive web interface built with Flask
- Real-time prediction capability
- Responsive and modern frontend design

## Technologies Used

- Python
- Flask
- Scikit-Learn
- NumPy
- Pandas
- HTML
- CSS

## Machine Learning Model

- Algorithm: Ridge Regression
- Dataset: Algerian Forest Fire Dataset
- Target Variable: Fire Weather Index (FWI)
- Preprocessing: Feature selection and StandardScaler

## Project Structure

```
TestForestFire
│
├── application.py
├── models
│   ├── ridge.pkl
│   └── scaler.pkl
├── templates
│   └── home.html
├── notebook
│   └── MODEL_TRAINING.ipynb
└── README.md
```

## Author

Divyansh Kaushik
