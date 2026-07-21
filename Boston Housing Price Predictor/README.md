# Boston Housing Price Predictor (Dash Web App)

## Project Overview

This project is an end-to-end Machine Learning application that predicts housing prices using a trained regression model. The model is trained on the Boston Housing dataset and deployed through an interactive Dash web application that allows users to input housing features and receive an estimated house price.

The goal of this project is to demonstrate a complete ML workflow, including:

- Data preprocessing and model training
- Model serialization using `.pkl`
- Building an interactive UI using Dash
- Creating a simple deployment-ready ML application

---

## Features

- Predict house prices based on housing characteristics
- Interactive web interface
- Clean and responsive UI
- Real-time prediction using a trained ML model
- End-to-end machine learning pipeline

---

## Technologies Used

- Python
- Dash
- NumPy
- Scikit-learn
- Pickle (model serialization)

---

## Project Structure

```
BostonHousingApp/
│
├── boston_model.pkl         
├── boston_Dash.py           
├── boston_housing.ipynb     
└── README.md                
```

---

## Model Inputs

The model predicts housing prices using the following features:

| Feature  | Description                              |
|----------|------------------------------------------|
| RM       | Average number of rooms per dwelling     |
| LSTAT    | Percentage of lower status population    |
| PTRATIO  | Pupil–teacher ratio by town              |

---

## Running the Application

**Start the Dash server:**

```bash
python app.py
```

**Open your browser and go to:**

```
http://127.0.0.1:8050
```

---

## How It Works

1. A machine learning model is trained using the Boston Housing dataset.
2. The trained model is saved as a `.pkl` file.
3. The Dash web app loads the saved model.
4. Users enter housing parameters through the UI.
5. The app sends the inputs to the model and displays the predicted price.

---

## Example Workflow

1. Enter the following values:
   - **RM** — Average number of rooms
   - **LSTAT** — Lower status population percentage
   - **PTRATIO** — Pupil–teacher ratio
2. Click **Predict Price**
3. The application returns the estimated housing price.


