import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('./spotify_churn_dataset.csv')
print("# Raw data: \n",df.head())
print("\n")
print("# Shape of dataset (rows, columns): \n",df.shape)
print("\n")
print("# Dataset information: \n",df.info())
print("\n")
print("# Summary Statistics of Numeric Columns: \n",df.describe())
print("\n")
print("# Missing values in each column: \n",df.isnull().sum())
print("\n")
print("# Duplicate values in the dataset: \n",df.duplicated().sum())
print("\n")
print("# Unique values in the dataset: \n",df.nunique())

# convert categorical (object) data to numerical (int64 / float64)
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1, 'Other': 2})
df['country'] = df['country'].map({'CA':0, 'DE':1, 'AU':2, 'US':3, 'UK':4, 'IN':5, 'FR':6, 'PK':7})
df['subscription_type'] = df['subscription_type'].map({'Free':0, 'Family':1, 'Premium':2, 'Student':3})
df['device_type'] = df['device_type'].map({'Desktop':0, 'Web':1, 'Mobile':2})

# display first 5 record after conversion
print("\n")
print("# Converted categoric to numeric: \n",df.head())

# calculate correlation matrix
correlation_matrix = df.corr()

# display correlation with "is_churned" target
print("\n")
print("# Correlation of features with 'is_churned': \n",correlation_matrix['is_churned'].sort_values(ascending=False))

# From correlation heatmap, all features have very weak correlation with target "is_churned"
# means that single feature cannot strongly predict "churn", but combination of features still can be usefull in machine learning models

# --------------------

# drop fitur "user_id" and "is_churned", then split dataset into features (X) and target (y)
df = df.drop('user_id', axis=1)
X = df.drop('is_churned', axis=1)
y = df['is_churned']

# Display shapes of X and y
print("\n")
print("# Shape of Features (X):", X.shape)
print("\n")
print("# Shape of Target (y):", y.shape)

# Features (X) contains all columns except user_id and the target "is_churned", used as model to predict churn
# Target (Y) contains only the "is_churned" will predit like (1=churned) (0=active)

# --------------------

# split data set into Training and Testing sets
# Training set -> used to train ML models
# Testing set -> used to evaluate model performance
# with 75:25 ratio means 75% for data training and 25% for data testing
# also random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# display shapes after split
print("\n")
print("# Shape of X_train:", X_train.shape)
# print("# X_train data: \n", X_train.head())
# contains n samples and n features for training the model

print("# Shape of X_test:", X_test.shape)
# print("\n# X_test data: \n", X_test.head())
#  contains n samples and n features for testing/evaluating the model

print("# Shape of y_train:", y_train.shape)
# print("\n# y_train data: \n", y_train.head())
# target values corresponding to X_train

print("# Shape of y_test:", y_test.shape)
# print("\n# y_test data: \n", y_test.head())
# target values corresponding to X_test
# this ensure ML models have enough data to learn pattern and a seperate set to evaluate performance reliably

# --------------------

# Features scaling using standar scaler
# important step before training a ML model, standardized range of features so each of that can contribute equally to the model's preformance
# to produce value for each feature like Mean = 0, Standar Deviasion = 1
scaler = StandardScaler()

# Fit the scaler on training data and transform both training and testing sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display first 5 records after scaling
print("\n")
# print("# First five records after scaling:\n",X_train_scaled[:5])
print("# X_test_caled data:\n",X_test_scaled[:5])

# --------------------

# Create Logistic Regression model with class weights balanced
log_reg = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

# Train the model
log_reg.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = log_reg.predict(X_test_scaled)

# Evaluate performance
print("\n")
print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))
print("\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n")
print("Classification Report:\n",classification_report(y_test, y_pred))

# show correlation matrix as heatmap
# plt.figure(figsize=(12,8))
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
# plt.title("Correlation Heatmap of Spotify Churn Dataset")
# plt.show()