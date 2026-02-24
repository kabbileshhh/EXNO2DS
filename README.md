# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
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
Dataset Loaded Successfully

   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S  

Dataset Info:
...
max      6.000000  512.329200  

Missing values handled successfully.


Cross Tabulation: Sex vs Survived

Survived    0    1
Sex               
female     72  157
male      406   83

Cross Tabulation: Pclass vs Survived

Survived    0    1
Pclass            
1          41   53
2          86   76
3         351  111

<img width="707" height="596" alt="image" src="https://github.com/user-attachments/assets/addc7efe-7eb0-446b-b7c5-505a2a6966ea" />
<img width="707" height="596" alt="image" src="https://github.com/user-attachments/assets/d78ffd35-c6ea-4dc3-b11a-24b35c998943" />
<img width="707" height="596" alt="image" src="https://github.com/user-attachments/assets/b6f54661-3988-4eb8-ad0c-9255de0b7399" />
<img width="707" height="596" alt="image" src="https://github.com/user-attachments/assets/ce186208-ebb4-4913-ade1-63ec3c2fb4ab" />
<img width="707" height="596" alt="image" src="https://github.com/user-attachments/assets/6b73f2a6-adae-4d95-8a5e-9b0b08aa66ee" />
<img width="707" height="596" alt="image" src="https://github.com/user-attachments/assets/f4702f53-9313-47e4-ac95-ad43c49e1f58" />
<img width="707" height="596" alt="image" src="https://github.com/user-attachments/assets/2e3e713e-b4dc-42d5-b0ff-971de169e67a" />
<img width="707" height="596" alt="image" src="https://github.com/user-attachments/assets/47687672-6738-4a63-a1df-93fc732ef520" />
















```

# RESULT
        thus the result has been executed zuccessfully
