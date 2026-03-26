import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
df = pd.read_csv("C:/Users/pcworld computers/Downloads/Financial Loan.csv")

# preview data
print(df.head())

# creating histogram with assigned columns
plt.figure()

sns.histplot(df["loan_amount"], bins=40)

plt.title("Distribution of Loan Amounts")

plt.show()
date_cols = [
    'issue_date',
    'last_credit_pull_date',
    'last_payment_date',
    'next_payment_date'
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

df['emp_length'] = df['emp_length'].fillna('Unknown')

# fill missing income
df['annual_income'] = df['annual_income'].fillna(df['annual_income'].median())

#dropping duplicates
df = df.drop_duplicates()

#
df['int_rate'] = df['int_rate'].astype(str).str.replace('%','').astype(float)
df['home_ownership'] = df['home_ownership'].str.upper()

print(df.info())
#assigning values to the "default" column

df['default'] = df['loan_status'].apply(lambda x: 1 if x == "Charged Off" else 0)

# creating a barplot

plt.figure()

sns.barplot(
    x="grade",
    y="default",
    data=df
)

plt.title("Loan Default Rate by Credit Grade")
plt.xlabel("Credit Grade")
plt.ylabel("Default Rate")

plt.show()

#creating a boxplot
plt.figure()

sns.boxplot(
    x="default",
    y="int_rate",
    data=df
)

plt.title("Interest Rate Distribution by Loan Default Status")
plt.xlabel("Default Status")
plt.ylabel("Interest Rate")

plt.show()

# creating a scatterplot
plt.figure()

sns.scatterplot(
    x="dti",
    y="loan_amount",
    hue="default",
    data=df
)

plt.title("Debt-to-Income Ratio and Loan Default")
plt.xlabel("Debt-to-Income Ratio")
plt.ylabel("Loan Amount")

plt.show()


8
df['issue_year'] = df['issue_date'].dt.year

loans_per_year = df.groupby('issue_year')['loan_amount'].sum()

plt.figure()

loans_per_year.plot()

plt.title("Total Loan Issuance Over Time")
plt.xlabel("Year")
plt.ylabel("Total Loan Amount")

plt.show()



df['issue_date'] = pd.to_datetime(
    df['issue_date'],
    errors='coerce',
    dayfirst=True
)

df = df.dropna(subset=['issue_date'])


df['issue_year'] = df['issue_date'].dt.year


print(df['issue_year'].value_counts())


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

print(df['issue_year'].unique())



print(df.shape)
print(df.head())


corr = df[['loan_amount','annual_income','dti','int_rate','default']].corr()

sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X = df[['loan_amount','annual_income','dti','int_rate']]
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)
from sklearn.metrics import accuracy_score

pred = model.predict(X_test)

accuracy_score(y_test,pred)
