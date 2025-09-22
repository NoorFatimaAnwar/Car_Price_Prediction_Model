# 🚗 Car Price Prediction System  

A machine learning project that predicts the **selling price of used cars** based on various features such as brand, age, fuel type, transmission, and ownership details.  

This project demonstrates a **complete ML pipeline**:
- Exploration
- Data preprocessing (outlier removal, encoding, scaling)
- Model training & evaluation (multiple regressors, Gradient Boosting as final model)
- Interactive prediction (CLI & Jupyter Notebook with `ipywidgets`)
- Modular code structure for reusability  

---

## 📂 Project Structure  

├── notebooks/ # Jupyter notebooks for exploration & experiments  
│  └── car_price_prediction.ipynb # For Exporation and Training
  

├── src/ # Reusable Python scripts  
│ ├── preprocessing.py # Outlier removal, encoding, scaling  
│ ├── modeling.py # Train & evaluate ML models  
│ └── predict.py # Predict car prices with trained model  

├── data/ # Dataset  

└── README.md # Project documentation  

---
## 📊 Dataset

The dataset includes the following features:

- **Present Price**: The current price of the car (in lakhs).
- **Driven Kms**: The total kilometers driven by the car.
- **Car Age**: The age of the car in years.
- **Brand**: The brand or make of the car.
- **Fuel Type**: The type of fuel used by the car (e.g., Petrol, Diesel, CNG).
- **Transmission**: The transmission type of the car (e.g., Manual, Automatic).
- **Owner**: The number of previous owners of the car.
- **Selling Price**: The target variable, representing the price at which the car is being sold (in lakhs).
---

## 🚀 Usage
### 1. Data (Raw Dataset)
### 2. src (Store Functions)

Open the `src` folder and run all python file one by one using terminal command like this for all python files:
```bash
python src/predict.py
```
- **Preprocessing.py**: Data cleaning & preprocessing.
- **Modeling.py**: Train & evaluate models.
- **Predict.py**: Predict for user inputs.

### 3. Jupyter Notebook (Exploration & Training)
Open the `notebooks` folder and run each cell one by one in order

### 4. Interactive Prediction (Jupyter Widgets)
 Inside the notebook, you can use ipywidgets to input details with dropdowns and sliders and get real-time predictions.
 
---
## 📈 Models Implemented

- Linear Regression

- Decision Tree Regressor

- Random Forest Regressor

- Support Vector Regressor (SVR)

- **Gradient Boosting Regressor ✅ (final model used)**

### Evaluation Metrics

- Mean Absolute Error (MAE)

- Mean Squared Error (MSE)

- R² Score

---
## 🛠️ Requirements

- Python 3.8+

- pandas, numpy, scikit-learn, matplotlib, seaborn

- ipywidgets (for interactive UI)
---
## 📌 Future Improvements

- Deploy as a Flask/Django web app

- Add Streamlit app for interactive UI
---
## 👩‍💻 Author

- Developed by Noor Fatima ✨

- If you like this project, don’t forget to ⭐ the repo!
