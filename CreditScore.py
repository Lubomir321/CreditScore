#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries for data manipulation and visualization
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for enhanced data visualization

# Load training, test, and sample entry datasets
train_df = pd.read_csv("C:/Users/ACER/Desktop/ii project/cs-training.csv")
test_df = pd.read_csv("C:/Users/ACER/Desktop/ii project/cs-test.csv")
sample_entry = pd.read_csv("C:/Users/ACER/Desktop/ii project/sampleEntry.csv")

# Display the first 10 rows of the training dataset
train_df.head(10)

# Drop the unnecessary 'Unnamed: 0' column from the training dataset
train_df.drop(columns=['Unnamed: 0'], inplace=True)
train_df

# Display summary statistics of the training dataset
train_df.describe()

# Display the test dataset
test_df

# Drop the unnecessary 'Unnamed: 0' column from the test dataset
test_df.drop(columns=['Unnamed: 0'], inplace=True)
test_df

# Display summary statistics of the test dataset
test_df.describe()

# Display the sample entry dataset
sample_entry

# Display the value counts for the target variable 'SeriousDlqin2yrs' in the training dataset
train_df['SeriousDlqin2yrs'].value_counts()

# Check for missing values in the training dataset
train_df.isna().sum()

# Import SimpleImputer from sklearn to handle missing values
from sklearn.impute import SimpleImputer

# Create an imputer object with strategy to fill missing values with mean
si = SimpleImputer(strategy='mean').fit(train_df[['MonthlyIncome']])

# Identify rows with missing 'MonthlyIncome'
has_monthly_income_na = train_df['MonthlyIncome'].isna()

# Impute missing 'MonthlyIncome' values
train_df['MonthlyIncome'] = si.transform(train_df[['MonthlyIncome']])

# Adjust 'DebtRatio' for rows where 'MonthlyIncome' was missing and imputed
train_df.loc[has_monthly_income_na, 'DebtRatio'] = train_df.loc[has_monthly_income_na, 'DebtRatio'] / train_df.loc[has_monthly_income_na, 'MonthlyIncome']

# Plot a boxplot for 'DebtRatio' to check for outliers
sns.boxplot(train_df['DebtRatio'])
plt.title('DebtRatio')
plt.show()

# Function to remove outliers based on the Interquartile Range (IQR) method
def remove_outliers(df, column_name, iqr_multiplier=2):
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    return df

# Remove outliers from 'DebtRatio'
train_df = remove_outliers(train_df, 'DebtRatio')

# Plot a boxplot for 'DebtRatio' after outlier removal
sns.boxplot(train_df['DebtRatio'])
plt.title('DebtRatio after outliers removal')
plt.show()

# Plot a boxplot for 'MonthlyIncome' to check for outliers
sns.boxplot(train_df['MonthlyIncome'])
plt.title('MonthlyIncome')
plt.show()

# Remove outliers from 'MonthlyIncome'
train_df = remove_outliers(train_df, 'MonthlyIncome')

# Plot a boxplot for 'MonthlyIncome' after outlier removal
sns.boxplot(train_df['MonthlyIncome'])
plt.show()

# Define independent variables (features) and the dependent variable (target)
independent_cols = [
    'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents'
]
dependent_col = 'SeriousDlqin2yrs'

# Check for missing values in the selected independent variables
train_df[independent_cols].isna().sum()

# Compute and print the correlation between 'age' and 'DebtRatio'
correlation = train_df[['age', 'DebtRatio']].corr()
print(f"Correlation between Age and DebtRatio: \n{correlation}\n")

# Compute binned averages for 'DebtRatio' by age group
age_bins = pd.cut(train_df['age'], bins=5)
age_bin_means = train_df.groupby(age_bins)['DebtRatio'].mean()

# Plot binned averages for 'DebtRatio' by age group
plt.figure(figsize=(10, 6))
age_bin_means.plot(kind='bar', color='skyblue')

# Adding title and labels
plt.title('Average DebtRatio by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average DebtRatio')

# Show plot
plt.show()

# Display summary statistics for 'age'
age_stats = train_df['age'].describe()
print(f"Summary statistics for age:\n{age_stats}\n")

# Display summary statistics for 'NumberOfOpenCreditLinesAndLoans'
credit_lines_stats = train_df['NumberOfOpenCreditLinesAndLoans'].describe()
print(f"Summary statistics for NumberOfOpenCreditLinesAndLoans:\n{credit_lines_stats}\n")

# Compute and print the correlation between 'age' and 'NumberOfOpenCreditLinesAndLoans'
correlation = train_df[['age', 'NumberOfOpenCreditLinesAndLoans']].corr()
print(f"Correlation between Age and Number of Open Credit Lines and Loans: \n{correlation}\n")

# Scatter plot of 'age' vs 'NumberOfOpenCreditLinesAndLoans'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='NumberOfOpenCreditLinesAndLoans', data=train_df, alpha=0.5)

# Fit a regression line to the scatter plot
sns.regplot(x='age', y='NumberOfOpenCreditLinesAndLoans', data=train_df, scatter=False, color='red')

# Adding title and labels
plt.title('Relationship between Age and Number of Open Credit Lines and Loans')
plt.xlabel('Age')
plt.ylabel('Number of Open Credit Lines and Loans')

# Show plot
plt.show()

# Compute binned averages for 'NumberOfOpenCreditLinesAndLoans' by age group
age_bins = pd.cut(train_df['age'], bins=5)
age_bin_means = train_df.groupby(age_bins)['NumberOfOpenCreditLinesAndLoans'].mean()

# Plot binned averages for 'NumberOfOpenCreditLinesAndLoans' by age group
plt.figure(figsize=(10, 6))
age_bin_means.plot(kind='bar', color='skyblue')

# Adding title and labels
plt.title('Average Number of Open Credit Lines and Loans by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average Number of Open Credit Lines and Loans')

# Show plot
plt.show()

# Scatter plot of 'NumberOfTimes90DaysLate' vs 'MonthlyIncome'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='MonthlyIncome', y='NumberOfTimes90DaysLate', data=train_df, alpha=0.5)

# Fit a regression line to the scatter plot
sns.regplot(x='MonthlyIncome', y='NumberOfTimes90DaysLate', data=train_df, scatter=False, color='red')

# Adding title and labels
plt.title('Relationship between Monthly Income and Number of Times 90 Days Late')
plt.xlabel('Monthly Income')
plt.ylabel('Number of Times 90 Days Late')

# Show plot
plt.show()

# Bar plot of average 'MonthlyIncome' by 'SeriousDlqin2yrs'
plt.figure(figsize=(10, 6))
sns.barplot(x='SeriousDlqin2yrs', y='MonthlyIncome', data=train_df, ci=None, palette='viridis')

# Adding title and labels
plt.title('Average Monthly Income by SeriousDlqin2yrs')
plt.xlabel('SeriousDlqin2yrs')
plt.ylabel('Average Monthly Income')

# Show plot
plt.show()

# Box plot of 'MonthlyIncome' by 'SeriousDlqin2yrs'
plt.figure(figsize=(10, 6))
sns.boxplot(x='SeriousDlqin2yrs', y='MonthlyIncome', data=train_df, palette='viridis')

# Adding title and labels
plt.title('Distribution of Monthly Income by SeriousDlqin2yrs')
plt.xlabel('SeriousDlqin2yrs')
plt.ylabel('Monthly Income')

# Show plot
plt.show()

# Import additional libraries for modeling
import random
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Define a pipeline for preprocessing numerical features
numeric_preprocessor = Pipeline(
    steps=[
        ("imputation_mean", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

# Combine preprocessing for numerical features
preprocessor = ColumnTransformer(
    [
        ("numerical", numeric_preprocessor, independent_cols),
    ]
)
preprocessor

# Sample 5000 rows from the training dataset for model training
df_sample = train_df.sample(5000, random_state=4)
X = df_sample[independent_cols]
y = df_sample[dependent_col]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)

# Display the first few rows of the training set
X_train.head(), y_train.head()

# Initialize models to try: RandomForestClassifier and LogisticRegression
rf = RandomForestClassifier(random_state=4)
lr = LogisticRegression(random_state=4)
models_to_try = [rf, lr]

# Dictionary to store pipelines for each model
all_pipelines = {}
for m in models_to_try:
    # Create a scikit-learn pipeline
    all_pipelines[m] = pipeline = make_pipeline(preprocessor, m)
    # Perform cross-validation on the training data
    cross_val_scores = cross_val_score(pipeline, X_train, y_train, cv=4, scoring='roc_auc')
    print(m)
    print(f"Cross-Validation Scores: {cross_val_scores}")
    print(f"Mean Cross-Validation Score: {np.mean(cross_val_scores)}\n")

# Select the best model (RandomForestClassifier based on cross-validation score)
all_pipelines[rf].fit(X_train, y_train)

# Import roc_auc_score for evaluating the model
from sklearn.metrics import roc_auc_score

# Use the selected model to predict on the test data
y_pred = all_pipelines[rf].predict_proba(X_test)[:, 1]
print(f'Test AUC of best model = {roc_auc_score(y_test, y_pred)}')

# Fit the best model on the entire dataset
all_pipelines[rf].fit(X, y)

# Check the test dataset
test_df

# Impute missing 'MonthlyIncome' values in the test dataset
has_monthly_income_na = test_df['MonthlyIncome'].isna()
test_df['MonthlyIncome'] = si.transform(test_df[['MonthlyIncome']])

# Adjust 'DebtRatio' for rows where 'MonthlyIncome' was missing and imputed
test_df.loc[has_monthly_income_na, 'DebtRatio'] = test_df.loc[has_monthly_income_na, 'DebtRatio'] / test_df.loc[has_monthly_income_na, 'MonthlyIncome']

# Check for missing values in the test dataset
test_df.isna().sum()

# Predict 'SeriousDlqin2yrs' for the test dataset using the best model
test_df[dependent_col] = all_pipelines[rf].predict(test_df)

# Display the value counts for the predictions
test_df['SeriousDlqin2yrs'].value_counts(normalize=True)

# Display the sample entry dataset
sample_entry

# Predict probabilities and update the 'Probability' column in the sample entry dataset
sample_entry['Probability'] = all_pipelines[rf].predict_proba(test_df)[:, 1]

# Save the updated sample entry dataset to a CSV file
sample_entry.to_csv('sampleEntry.csv', index=False)

# Display the final sample entry dataset
sample_entry


# In[ ]:




