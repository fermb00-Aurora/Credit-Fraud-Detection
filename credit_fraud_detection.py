import timeit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

# Suppress warnings
warnings.filterwarnings("ignore")

# Streamlit App Title
st.title('Credit Card Fraud Detection!')

# Load the dataset with caching
@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

df = load_data()

# Display DataFrame details
if st.sidebar.checkbox('Show what the dataframe looks like'):
    st.write(df.head(100))
    st.write('Shape of the dataframe: ', df.shape)
    st.write('Data description:', df.describe())

# Fraud and Valid Transaction Analysis
fraud = df[df['Class'] == 1]
valid = df[df['Class'] == 0]
outlier_percentage = (len(fraud) / len(valid)) * 100

if st.sidebar.checkbox('Show fraud and valid transaction details'):
    st.write(f'Fraudulent transactions are: {outlier_percentage:.3f}%')
    st.write('Fraud Cases:', len(fraud))
    st.write('Valid Cases:', len(valid))

# Splitting the features and labels
X = df.drop(columns=['Class'])
y = df['Class']

# Train-test split
size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)

# Initialize classifiers
logreg = LogisticRegression()
svm = SVC()
knn = KNeighborsClassifier()
etree = ExtraTreesClassifier(random_state=42)
rforest = RandomForestClassifier(random_state=42)

# Feature selection through feature importance
@st.cache_data
def feature_sort(_model, X_train, y_train):
    _model.fit(X_train, y_train)
    importance = _model.feature_importances_
    return importance

# Feature Importance Selection
clf = ['Extra Trees', 'Random Forest']
mod_feature = st.sidebar.selectbox('Which model for feature importance?', clf)

if mod_feature == 'Extra Trees':
    model = etree
    importance = feature_sort(model, X_train, y_train)
elif mod_feature == 'Random Forest':
    model = rforest
    importance = feature_sort(model, X_train, y_train)

# Plot of feature importance
if st.sidebar.checkbox('Show plot of feature importance'):
    plt.bar(range(len(importance)), importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature (Variable Number)')
    plt.ylabel('Importance')
    st.pyplot()

# Select top features based on importance
top_features = [X.columns[i] for i in np.argsort(importance)[-10:]]
X_train_sfs = X_train[top_features]
X_test_sfs = X_test[top_features]

# Handling class imbalance with SMOTE
smt = SMOTE(random_state=42)
X_train_bal, y_train_bal = smt.fit_resample(X_train_sfs, y_train)

# Model selection and training
classifier = st.sidebar.selectbox('Select the classifier', ['Logistic Regression', 'kNN', 'SVM', 'Random Forest', 'Extra Trees'])

if classifier == 'Logistic Regression':
    model = logreg
elif classifier == 'kNN':
    model = knn
elif classifier == 'SVM':
    model = svm
elif classifier == 'Random Forest':
    model = rforest
elif classifier == 'Extra Trees':
    model = etree

# Train the selected model
st.write(f"Training {classifier}...")
model.fit(X_train_bal, y_train_bal)

# Save the trained model as a .pkl file
model_filename = f"{classifier.lower().replace(' ', '_')}_model.pkl"
joblib.dump(model, model_filename)
st.write(f"Model saved as '{model_filename}'")

# Model evaluation
y_pred = model.predict(X_test_sfs)
cm = confusion_matrix(y_test, y_pred)
st.write('Confusion Matrix:', cm)
st.write('Classification Report:', classification_report(y_test, y_pred))
mcc = matthews_corrcoef(y_test, y_pred)
st.write(f'Matthews Correlation Coefficient: {mcc:.3f}')


