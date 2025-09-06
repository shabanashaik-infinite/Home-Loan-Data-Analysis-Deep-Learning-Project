# Home-Loan-Data-Analysis-Deep-Learning-Project

House Loan Data Analysis

Project Overview

This project focuses on building a deep learning model to predict the likelihood of loan default based on historical loan data. The dataset provided is highly imbalanced and includes various features such as applicant demographics, income, credit scores, and loan details. The objective is to preprocess the data, balance the dataset, and develop a neural network model to predict whether an applicant will repay a loan (TARGET = 0) or default (TARGET = 1). The model is evaluated using sensitivity (recall) and the area under the ROC curve (ROC-AUC).

Domain: Finance
Objective: Predict loan repayment probability using historical data
Tools: Python, Google Colab, TensorFlow, Scikit-learn, SMOTE, Pandas, Matplotlib, Seaborn

Dataset

The dataset (loan_data.csv) contains historical loan data with the following key features:


TARGET: Binary target variable (0 = non-default, 1 = default)

Applicant details: Gender, income, family status, education, occupation, etc.

Loan details: Loan type, amount, credit scores, and other financial metrics

The dataset is imbalanced, with a higher proportion of non-default cases.

Project Steps

Load the Dataset: Upload and read loan_data.csv in Google Colab.

Check for Null Values: Identify and impute missing values (median for numerical, mode for categorical).

Analyze TARGET Distribution: Calculate and display the percentage of defaults and non-defaults.

Balance the Dataset: Use SMOTE to address class imbalance.

Visualize Data: Plot class distribution before and after balancing.

Encode Features: Apply LabelEncoder for categorical variables and StandardScaler for numerical features.

Build and Train Model: Create a deep learning model using TensorFlow/Keras with class weights for imbalance.

Evaluate Metrics:

Calculate sensitivity (recall) for the positive class (default).

Compute the ROC-AUC score.

Visualize Training: Plot training and validation loss.

Requirements

The script is designed to run in Google Colab with the following libraries (pre-installed in Colab):

Python 3
pandas
numpy
matplotlib
seaborn
scikit-learn
imblearn
tensorflow
No additional installations are required in Google Colab.

How to Run
Open Google Colab: Go to colab.research.google.com and create a new notebook.

Upload Dataset:

Ensure loan_data.csv is available.

The script includes files.upload() to prompt for uploading the CSV file.

Copy the Code

Execute the Script:

Run the code cell. The script will:

Prompt for uploading loan_data.csv.

Preprocess the data (handle missing values, encode features, scale numerical data).
Balance the dataset using SMOTE and display class distribution plots

Train a deep learning model and evaluate it.

Output sensitivity, ROC-AUC, a classification report, and a training loss plot.

Review Outputs:

Check console output for null value summaries, default percentages, and metrics.

View plots for class distribution and model training loss.

File Structure

loan_data.csv: Input dataset with loan data.



loan_default_prediction.py: Main Python script for data analysis and model building.

Notes


Imbalance Handling: The script uses SMOTE for balancing and class weights during training to handle the imbalanced dataset.

Model Architecture: A simple neural network with three hidden layers (64, 32, 16 neurons) and dropout (0.3) to prevent overfitting.

Metrics: Sensitivity (recall) and ROC-AUC are calculated to evaluate performance on the imbalanced dataset.

Customization: Modify the script for specific feature selection, hyperparameter tuning, or additional visualizations as needed.

Error Handling: The script includes a fix for compute_class_weight to use np.array([0, 1]) for the classes parameter.

Troubleshooting

Upload Issues: Ensure test_loan_data.csv is correctly formatted and not corrupted.

Memory Errors: If Colab runs out of memory, reduce the batch size or dataset size.

Model Performance: Adjust model architecture (layers, neurons) or hyperparameters (epochs, learning rate) for better results.

Contact: For issues, refer to the script comments or contact the project maintainer.

Maintainer

This project is maintained for educational purposes. For questions or enhancements, please provide feedback or additional dataset details.
