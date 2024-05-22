# Credit Risk Classification

This repository contains a project for classifying credit risk using a Logistic Regression model. The project involves reading data, preprocessing it, training a model, and evaluating its performance using various metrics.

## Project Structure

The project has the following structure:
.
├── credit_risk_classification.ipynb # Jupyter Notebook with the code for the project
├── Resources
│ └── lending_data.csv # Source data file for the project
└── README.md # Project README file



## Getting Started

### Prerequisites

To run this project, you need the following:
- Google Colab or a local Jupyter Notebook environment
- A Google Drive account to store and access the dataset
- Python 3.6+
- Required Python libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`

### Installation

1. Clone this repository to your local machine:
   bash
   git clone https://github.com/your-username/credit-risk-classification.git
Upload the Resources folder containing lending_data.csv to your Google Drive.

Open the credit_risk_classification.ipynb file in Google Colab or your local Jupyter Notebook environment.

Ensure the following Python libraries are installed:

    bash
    pip install numpy pandas scikit-learn matplotlib

## Running the Project
Mount Google Drive:
In the Jupyter Notebook, mount your Google Drive to access the dataset:

    python
    from google.colab import drive
    drive.mount('/content/drive')

## Read the Dataset:
Load the dataset from the specified path in Google Drive:

    python
    from pathlib import Path
    import pandas as pd
    
    csv_path = Path('/content/drive/My Drive/credit_risk_classification/Resources/lending_data.csv')
    lending_df = pd.read_csv(csv_path)

## Data Preparation:
Separate the dataset into features (X) and labels (y):

    python
    y = lending_df["loan_status"]
    X = lending_df.drop(columns=["loan_status"])

## Split the Data:
Split the data into training and testing sets:
 
    python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

## Train the Model:
Instantiate and train the Logistic Regression model:

    python
    from sklearn.linear_model import LogisticRegression
    logistic_regression_model = LogisticRegression(random_state=1)
    logistic_regression_model.fit(X_train, y_train)

## Make Predictions:
Use the model to make predictions on the test set:

    python
    y_pred = logistic_regression_model.predict(X_test)

## Evaluate the Model:
Generate and print the confusion matrix and classification report:

    python
    from sklearn.metrics import confusion_matrix, classification_report
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    class_report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(class_report)

### Results
The Logistic Regression model shows high accuracy in predicting the credit risk:

Accuracy: 99%
Precision, Recall, F1-score: Detailed in the classification report for each class (0 and 1).
Confusion Matrix:
    lua
    Confusion Matrix:
    [[18616   149]
     [   55   564]]

## Classification Report:
    Classification Report:
                   precision    recall  f1-score   support
    
               0       1.00      0.99      1.00     18765
               1       0.85      0.91      0.88       619
    
        accuracy                           0.99     19384
       macro avg       0.92      0.95      0.94     19384
    weighted avg       0.99      0.99      0.99     19384

## Contributing
If you would like to contribute to this project, please fork the repository and create a pull request with your changes. We welcome contributions that improve the code or add new features.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
