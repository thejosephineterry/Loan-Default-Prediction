import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
df = pd.read_csv("Financial Loan.csv")

# -------------------------------
# BASIC CLEANING
# -------------------------------

# fix interest rate column
df['int_rate'] = df['int_rate'].astype(str).str.replace('%','').astype(float)

# create default variable
df['default'] = df['loan_status'].apply(lambda x: 1 if x == "Charged Off" else 0)

# clean income
df['annual_income'] = df['annual_income'].fillna(df['annual_income'].median())

# -------------------------------
# DATE FIX
# -------------------------------

df['issue_date'] = pd.to_datetime(df['issue_date'], errors='coerce')
df = df.dropna(subset=['issue_date'])

df['issue_year'] = df['issue_date'].dt.year

# -------------------------------
# VISUAL 1: DEFAULT RATE BY CREDIT GRADE
# -------------------------------

plt.figure()

sns.barplot(
    x='grade',
    y='default',
    data=df
)

plt.title("Loan Default Rate by Credit Grade")
plt.xlabel("Credit Grade")
plt.ylabel("Default Rate")

plt.show()

# -------------------------------
# VISUAL 2: INTEREST RATE VS DEFAULT
# -------------------------------

plt.figure()

sns.boxplot(
    x='default',
    y='int_rate',
    data=df
)

plt.title("Interest Rate Distribution by Loan Default Status")
plt.xlabel("Default Status")
plt.ylabel("Interest Rate")

plt.show()

# -------------------------------
# VISUAL 3: DTI VS LOAN AMOUNT
# -------------------------------

plt.figure()

sns.scatterplot(
    x='dti',
    y='loan_amount',
    hue='default',
    data=df
)

plt.title("Debt-to-Income Ratio vs Loan Amount")
plt.xlabel("Debt-to-Income Ratio")
plt.ylabel("Loan Amount")

plt.show()

# -------------------------------
# VISUAL 4: LOANS ISSUED OVER TIME
# -------------------------------

loans_per_year = df.groupby('issue_year')['loan_amount'].sum().reset_index()

plt.figure()

sns.lineplot(
    x='issue_year',
    y='loan_amount',
    data=loans_per_year,
    marker='o'
)

plt.title("Total Loan Issuance Over Time")
plt.xlabel("Year")
plt.ylabel("Total Loan Amount")

plt.show()



# -----------------------------
# MACHINE LEARNING MODEL
# -----------------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Features (independent variables)
X = df[['loan_amount','annual_income','dti','int_rate']]

# Target variable
y = df['default']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, pred)

print("Model Accuracy:", accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, pred))



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = df[['loan_amount','annual_income','dti','int_rate']]
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

model = LogisticRegression(class_weight='balanced')

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))




plt.figure(figsize=(8,6))

corr = df[['loan_amount','annual_income','dti','int_rate','default']].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Correlation Heatmap of Loan Risk Variables")

plt.show()

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------

import pandas as pd

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.coef_[0]
})

importance = importance.sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(importance)



plt.figure()
sns.set_style("whitegrid")
sns.barplot(
    x='Importance',
    y='Feature',
    data=importance
)

plt.title("Feature Importance in Loan Default Prediction")

plt.show()
