# EXNO2DS
## AIM:
To perform Exploratory Data Analysis on the given data set.
      
## EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
## ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING:
```
# ----------------------------------------
# Step 1: Import Required Packages
# ----------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------
# Step 2: Load the Dataset
# ----------------------------------------
data = pd.read_csv("Exp_2_dataset_titanic_dataset.csv")

print("\nDataset Loaded Successfully\n")
print(data.head())
print("\nDataset Info:\n")
print(data.info())
print(data.describe())
# ----------------------------------------
# Step 3: Data Cleansing - Handle Missing Values
# ----------------------------------------
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = data[column].fillna(data[column].mode()[0])   # Mode for categorical
    else:
        data[column] = data[column].fillna(data[column].median())   # Median for numerical

print("\nMissing values handled successfully.\n")

# ----------------------------------------
# Step 4: Boxplot to Analyze Outliers (Age & Fare)
# ----------------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x=data["Age"])
plt.title("Boxplot - Age")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=data["Fare"])
plt.title("Boxplot - Fare")
plt.show()

# ----------------------------------------
# Step 5: Remove Outliers Using IQR Method
# ----------------------------------------
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

data = remove_outliers_iqr(data, "Age")
data = remove_outliers_iqr(data, "Fare")

print("Outliers removed using IQR method.\n")

# ----------------------------------------
# Step 6: Countplot for Categorical Data
# ----------------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=data)
plt.title("Countplot - Survival Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Sex", data=data)
plt.title("Countplot - Gender Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", data=data)
plt.title("Countplot - Passenger Class Distribution")
plt.show()

# ----------------------------------------
# Step 7: Displot for Univariate Distribution
# ----------------------------------------
sns.displot(data["Age"], kde=True, height=4, aspect=1.5)
plt.title("Displot - Age Distribution")
plt.show()

sns.displot(data["Fare"], kde=True, height=4, aspect=1.5)
plt.title("Displot - Fare Distribution")
plt.show()

# ----------------------------------------
# Step 8: Cross Tabulation
# ----------------------------------------
print("\nCross Tabulation: Sex vs Survived\n")
print(pd.crosstab(data["Sex"], data["Survived"]))

print("\nCross Tabulation: Pclass vs Survived\n")
print(pd.crosstab(data["Pclass"], data["Survived"]))

# ----------------------------------------
# Step 9: Heatmap for Correlation Analysis
# ----------------------------------------
plt.figure(figsize=(8,6))
correlation_matrix = data.select_dtypes(include=np.number).corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap - Titanic Dataset")
plt.show()

````
## OUTPUT:

<img width="1223" height="696" alt="Screenshot 2026-02-23 190305" src="https://github.com/user-attachments/assets/455a8260-65fc-4f17-bf7f-ddcb32b892d2" />
<img width="615" height="484" alt="Screenshot 2026-02-23 190318" src="https://github.com/user-attachments/assets/e6e394a2-386c-4ae1-a748-16affc0aa746" />
<img width="608" height="482" alt="Screenshot 2026-02-23 190329" src="https://github.com/user-attachments/assets/a88273bc-dc7c-42d1-ac48-03bb9d0af94a" />
<img width="672" height="479" alt="Screenshot 2026-02-23 190340" src="https://github.com/user-attachments/assets/1eaf7c74-acac-4527-aeeb-78008a9ce6f1" />
<img width="540" height="391" alt="image" src="https://github.com/user-attachments/assets/8751543d-2af6-456d-8d0f-915d97e0703c" />
<img width="540" height="391" alt="image" src="https://github.com/user-attachments/assets/4dc16920-540d-47df-a141-905aa0d0c5e5" />
<img width="589" height="410" alt="image" src="https://github.com/user-attachments/assets/eba977ab-63b7-40e0-a218-035619d32ca6" />
<img width="589" height="410" alt="image" src="https://github.com/user-attachments/assets/456330e4-8f58-4b49-aec0-b48d303d603c" />
<img width="547" height="343" alt="image" src="https://github.com/user-attachments/assets/769fdaa9-a4d3-43c5-a9fb-19c4ad08b93a" />
<img width="707" height="596" alt="image" src="https://github.com/user-attachments/assets/6a215f49-c9dd-46f9-ab1e-55ef63db2c4b" />

## RESULT:
To perform Exploratory Data Analysis on the given data set is succesfully completed.
       
