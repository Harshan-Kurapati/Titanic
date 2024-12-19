

# Titanic Survival Prediction using Logistic Regression with Pipelines

This project focuses on predicting survival outcomes for passengers on the Titanic based on their personal and ticket information. The project implements a complete preprocessing and classification pipeline using **Logistic Regression**, with efficient handling of missing data, feature scaling, encoding, and evaluation.

## Project Overview

This project demonstrates:
1. **Data Preprocessing**:
   - Handling missing values.
   - Encoding categorical variables.
   - Scaling and standardizing numerical features.
2. **Pipeline Integration**:
   - Combining preprocessing and model training into a unified pipeline.
3. **Logistic Regression**:
   - Training a classification model to predict survival based on passenger attributes.
4. **Evaluation**:
   - Assessing the model's accuracy on unseen test data.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Performance](#performance)
- [Conclusion](#conclusion)
- [License](#license)

## Installation

To run this project, ensure you have the following libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Dataset

The **Titanic dataset** provides information on passengers aboard the Titanic, including:
- **Features**:
  - `Pclass`: Ticket class (1, 2, or 3).
  - `Sex`: Gender of the passenger.
  - `Age`: Passenger's age.
  - `SibSp`: Number of siblings or spouses aboard.
  - `Parch`: Number of parents or children aboard.
  - `Fare`: Ticket fare.
  - `Embarked`: Port of embarkation (C, Q, S).
  - `Deck`: Extracted from the `Cabin` feature.
- **Target Variable**:
  - `Survived`: Whether the passenger survived (1) or not (0).

## Project Workflow

### Step 1: Data Preprocessing
1. **Handling Missing Values**:
   - Imputed missing `Age` values with the median.
   - Imputed missing `Embarked` values with the mode.
   - Retained `Deck` as it showed significant correlation with survival, replacing missing values with "unknown."
   - Dropped `Cabin`, `Name`, and `Ticket` columns due to high missing rates or irrelevance.

2. **Feature Encoding**:
   - Converted `Sex` to numerical values (`male -> 1`, `female -> 0`).
   - One-hot encoded the `Embarked` column.

3. **Feature Scaling**:
   - Standardized numerical features using **StandardScaler**.
   - Compared distributions between **StandardScaler** and **MinMaxScaler**.

### Step 2: Building a Machine Learning Pipeline
- **Numerical Features**:
  - Imputed missing values using the median.
  - Standardized values using **StandardScaler**.
- **Categorical Features**:
  - Imputed missing values using the most frequent value.
  - One-hot encoded categorical features using **OneHotEncoder**.
- Integrated preprocessing steps with **Logistic Regression** in a single pipeline using **ColumnTransformer**.

### Step 3: Training and Evaluation
- Split the data into training (80%) and testing (20%) sets.
- Trained the model using logistic regression and evaluated it on the test set.

### Pipeline Code Snippet
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Define pipelines
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Train and evaluate
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")
```

## Performance

- **Model Accuracy**: **82.12%**

This result indicates that the logistic regression model performs well in predicting survival outcomes, given the available features.

## Conclusion

This project demonstrates:
- Effective preprocessing using **Pipelines** for a seamless and modular workflow.
- A straightforward implementation of logistic regression for binary classification.
- Handling of missing data and categorical variables to improve model accuracy.

### Future Improvements
- Explore other classification models, such as decision trees, random forests, or gradient boosting, to compare performance.
- Perform hyperparameter tuning on the logistic regression model to optimize performance.

## License

This project is licensed under the MIT License.

