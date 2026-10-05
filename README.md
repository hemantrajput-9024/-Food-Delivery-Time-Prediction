# Food Delivery Time Prediction

## Project Overview

This project is a Machine Learning project that predicts the **delivery time of a food order in minutes**. The model uses information such as delivery distance, weather, traffic level, time of day, vehicle type, food preparation time, and courier experience.

The project follows a complete Machine Learning workflow, starting from data collection and preprocessing and ending with model training, evaluation, and model saving.

The final model used in this project is a **Random Forest Regressor**.

---

## Dataset

The dataset used in this project is the **Food Delivery Time Prediction** dataset.

The dataset is downloaded using KaggleHub:

```python
path = kagglehub.dataset_download("denkuznetz/food-delivery-time-prediction")
```

The dataset file used is:

```text
Food_Delivery_Times.csv
```

### Dataset Information

- **Number of rows:** 1,000
- **Number of columns:** 9

### Features

| Column | Description |
|---|---|
| `Order_ID` | Unique ID of the food order |
| `Distance_km` | Distance between restaurant and delivery location |
| `Weather` | Weather condition during delivery |
| `Traffic_Level` | Traffic level during delivery |
| `Time_of_Day` | Time period of the delivery |
| `Vehicle_Type` | Type of vehicle used by the courier |
| `Preparation_Time_min` | Time required to prepare the food |
| `Courier_Experience_yrs` | Experience of the courier in years |
| `Delivery_Time_min` | Total delivery time in minutes and target variable |

---

## Technologies Used

The following Python libraries and technologies are used:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- KaggleHub
- Joblib

---

## Project Workflow

The project follows these steps:

1. Import required libraries
2. Download the dataset
3. Load the CSV file
4. Understand the dataset
5. Check missing values
6. Encode categorical features
7. Handle missing values
8. Perform data visualization
9. Detect and handle outliers
10. Separate features and target
11. Split the dataset into training and testing data
12. Train the Random Forest Regressor
13. Make predictions
14. Evaluate the model
15. Save the model and encoders
16. Load the saved model and encoders

---

## Data Preprocessing

### 1. Missing Value Checking

Missing values are checked using:

```python
df.isnull().sum()
```

Missing values were found in:

- `Weather`
- `Traffic_Level`
- `Time_of_Day`
- `Courier_Experience_yrs`

The missing values are handled before model training.

---

### 2. Categorical Encoding

The categorical columns are converted into numerical values using `LabelEncoder`.

The following columns are encoded:

```text
Weather
Traffic_Level
Time_of_Day
Vehicle_Type
```

Example:

```python
Weather_le = LabelEncoder()
Traffic_Level_le = LabelEncoder()
Time_of_Day_le = LabelEncoder()
Vehicle_Type_le = LabelEncoder()
```

The encoded values allow the Machine Learning model to work with categorical information.

---

### 3. Missing Value Handling

After encoding, missing values are filled using the mean of the respective columns.

```python
df["Weather"] = df["Weather"].fillna(df["Weather"].mean())
df["Traffic_Level"] = df["Traffic_Level"].fillna(df["Traffic_Level"].mean())
df["Time_of_Day"] = df["Time_of_Day"].fillna(df["Time_of_Day"].mean())
df["Courier_Experience_yrs"] = df["Courier_Experience_yrs"].fillna(
    df["Courier_Experience_yrs"].mean()
)
```

After this step, the dataset contains no missing values.

---

## Exploratory Data Analysis

The notebook performs different checks and visualizations to understand the dataset.

Examples include:

- Dataset information
- Missing value analysis
- Distribution of numerical features
- Weather frequency analysis
- Count plots
- Histograms
- Feature distributions

These visualizations help understand the data before building the Machine Learning model.

---

## Outlier Handling

Outliers are handled using the **Interquartile Range (IQR)** method.

The target column used for outlier handling is:

```text
Delivery_Time_min
```

The IQR is calculated as:

```text
IQR = Q3 - Q1
```

The notebook calculates the following fences:

```text
Lower Fence = -4.00
Upper Fence = 116.00
```

Values outside the calculated range are capped instead of being removed.

This helps reduce the influence of extreme values on the model.

---

## Feature and Target Selection

The target variable is:

```text
Delivery_Time_min
```

The features are created by removing the target column:

```python
X = df.drop(columns=["Delivery_Time_min"])
y = df["Delivery_Time_min"]
```

Here:

- `X` contains the input features.
- `y` contains the delivery time that the model needs to predict.

---

## Train-Test Split

The dataset is divided into training and testing data using `train_test_split`.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The dataset is divided into:

- **80% Training Data**
- **20% Testing Data**

`random_state=42` is used to make the results reproducible.

---

## Machine Learning Model

The project uses the **Random Forest Regressor**.

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

The model contains **100 decision trees**.

Random Forest is suitable for this problem because the target variable, `Delivery_Time_min`, is a continuous numerical value.

---

## Model Training

The model is trained using:

```python
rf_model.fit(X_train, y_train)
```

During training, the Random Forest model learns the relationship between the input features and food delivery time.

---

## Prediction

After training, predictions are generated using the testing data:

```python
y_pred = rf_model.predict(X_test)
```

The predicted values are then compared with the actual delivery times.

---

## Model Evaluation

Three regression metrics are used to evaluate the model:

### R² Score

```python
r2_score(y_test, y_pred)
```

The R² score measures how well the model explains the variation in the target variable.

**Result:**

```text
R² Score = 0.780873
```

The model explains approximately **78% of the variation** in delivery time according to the recorded test result.

---

### Mean Absolute Error

```python
mean_absolute_error(y_test, y_pred)
```

The MAE represents the average absolute difference between the actual and predicted delivery times.

**Result:**

```text
MAE = 6.9688 minutes
```

This means the predictions are approximately **6.97 minutes away from the actual values on average**.

---

### Root Mean Squared Error

```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

The RMSE gives more weight to larger prediction errors.

**Result:**

```text
RMSE = 9.8683 minutes
```

---

## Model Performance

| Metric | Score |
|---|---:|
| R² Score | 0.7809 |
| MAE | 6.9688 minutes |
| RMSE | 9.8683 minutes |

The recorded results show that the Random Forest model provides a reasonable prediction of food delivery time.

---

## Model Saving

The trained model and all categorical encoders are saved together using Joblib.

```python
model_data = {
    "model": rf_model,
    "Weather_le": Weather_le,
    "Traffic_Level_le": Traffic_Level_le,
    "Time_of_Day_le": Time_of_Day_le,
    "Vehicle_Type_le": Vehicle_Type_le
}

joblib.dump(model_data, "traffic_rf_model.joblib")
```

The saved file is:

```text
traffic_rf_model.joblib
```

Saving the encoders together with the model is important because the same categorical encoding must be used when making future predictions.

---

## Model Loading

The saved model can be loaded using:

```python
loaded_data = joblib.load("traffic_rf_model.joblib")
```

Then the model and encoders are restored:

```python
rf_model = loaded_data["model"]

Weather_le = loaded_data["Weather_le"]
Traffic_Level_le = loaded_data["Traffic_Level_le"]
Time_of_Day_le = loaded_data["Time_of_Day_le"]
Vehicle_Type_le = loaded_data["Vehicle_Type_le"]
```

The notebook successfully verifies both model dumping and loading.

---

## Project Structure

A possible project structure is:

```text
Food-Delivery-Time-Prediction/
│
├── Food_Delivery_Time_Prediction.ipynb
├── Food_Delivery_Times.csv
├── traffic_rf_model.joblib
└── README.md
```

---

## Installation

Install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub joblib
```

---

## How to Run the Project

### Step 1: Install Python

Make sure Python is installed on your system.

### Step 2: Install Dependencies

Run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub joblib
```

### Step 3: Open the Notebook

Open:

```text
Food_Delivery_Time_Prediction.ipynb
```

using Jupyter Notebook, JupyterLab, or Google Colab.

### Step 4: Run the Cells

Run the notebook cells from beginning to end.

The notebook will download the dataset, preprocess the data, train the model, evaluate its performance, and save the trained model.

---

## Key Learning Outcomes

This project demonstrates practical Machine Learning concepts including:

- Data collection
- Data loading using Pandas
- Exploratory Data Analysis
- Missing value handling
- Label Encoding
- Outlier detection
- IQR-based outlier capping
- Feature-target separation
- Train-test splitting
- Random Forest Regression
- Model prediction
- R² Score
- MAE
- RMSE
- Model serialization using Joblib
- Loading a trained model for future use

---

## Future Improvements

The project can be improved further by:

- Comparing multiple regression algorithms.
- Performing hyperparameter tuning using GridSearchCV.
- Testing XGBoost and Extra Trees Regressors.
- Performing feature-importance analysis.
- Using a complete preprocessing pipeline.
- Improving categorical missing-value handling.
- Building a Streamlit web application.
- Creating a real-time food delivery prediction system.
- Adding more data to improve model generalization.

---

## Conclusion

The Food Delivery Time Prediction project demonstrates a complete Machine Learning regression workflow.

The dataset is cleaned and preprocessed, categorical features are encoded, missing values are handled, and outliers are capped using the IQR method. A Random Forest Regressor is then trained to predict food delivery time.

The recorded model performance is:

```text
R² Score : 0.7809
MAE      : 6.9688 minutes
RMSE     : 9.8683 minutes
```

The trained model and categorical encoders are saved using Joblib in:

```text
traffic_rf_model.joblib
```

This allows the trained model to be reused later for prediction applications or deployment.
