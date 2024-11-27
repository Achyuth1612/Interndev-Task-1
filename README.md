# Interndev-Task-1
HEART DISEASE PREDICTION

The code implements a machine learning pipeline to predict Heart Disease using a Random Forest Classifier. 

Here’s a summary of the steps and what the code does:

Importing Libraries: The code starts by importing essential Python libraries such as pandas, numpy, matplotlib, seaborn, LabelEncoder from scikit-learn, and various metrics and tools for machine learning like train_test_split, StandardScaler, RandomForestClassifier, classification_report, confusion_matrix, roc_auc_score, and roc_curve.

Data Loading and Initial Exploration: The dataset Heart_Disease_Prediction.csv is loaded using pandas. The first few rows are printed, and it’s checked for missing values using isnull().sum(). If missing values are found, appropriate imputation methods can be applied.

Data Preprocessing:

Encoding Categorical Data: A LabelEncoder is used to encode the Heart Disease column, transforming categorical labels into numeric format.
Correlation Heatmap and Pairplot:
A correlation heatmap is generated using seaborn to understand relationships between features.
A pairplot is created to visualize relationships between features and the target (Heart Disease).
Feature-Target Split:
Features (X) are extracted by dropping the Heart Disease column.
The target variable (y) is set to the Heart Disease column.
Data Scaling: The features (X) are scaled using StandardScaler to ensure all feature values are on a similar scale. This improves model performance.

Model Training and Evaluation:

The dataset is split into X_train (training set) and X_test (test set) using train_test_split.
A Random Forest Classifier is trained on the training set with n_estimators set to 100.
Model Prediction and Evaluation:
Confusion Matrix: Visualized using seaborn to understand how well the model predicted positive and negative cases.
Classification Report: Printed to get precision, recall, F1-score, and support metrics.
ROC Curve: Generated using roc_curve to understand the model's ability to discriminate between classes, and the AUC score is computed.
Feature Importance: Determined using feature_importances_ from the Random Forest model, and visualized using a bar plot.

Results:

The confusion matrix helps assess how well the model performs in terms of true positive, true negative, false positive, and false negative predictions.
The classification report gives detailed performance metrics.
The ROC curve helps visualize the model's discriminative power, with an AUC score indicating how well the model distinguishes between positive and negative cases.
The feature importance plot helps identify which features contribute most to predicting heart disease.
Overall, the code provides a comprehensive approach to building and evaluating a Random Forest model for heart disease prediction, including data preprocessing, feature scaling, model training, and various evaluation techniques to interpret the results.
