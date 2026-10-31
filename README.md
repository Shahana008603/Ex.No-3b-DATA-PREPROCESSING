# Ex.No-3b-DATA PREPROCESSING
## Aim
To perform data preprocessing on a dataset using Python and Scikit-learn by handling missing values, encoding categorical data, splitting the dataset, and applying feature scaling. 
## Procedure
1.	Import the required Python libraries. 
2.	Mount Google Drive and load the dataset using Pandas. 
3.	Display the first few records of the dataset. 
4.	Inspect the dataset using df.info() and df.shape. 
5.	Separate the independent variables (X) and dependent variable (Y). 
6.	Convert the independent variables into an array. 
7.	Identify and handle missing values using SimpleImputer with the mean strategy. 
8.	Encode the categorical Country column using LabelEncoder. 
9.	Apply One-Hot Encoding to convert categorical country values into dummy variables. 
10.	Encode the dependent variable Purchased using LabelEncoder. 
11.	Split the dataset into training and testing sets using train_test_split. 
12.	Apply StandardScaler for feature scaling. 
13.	Display the preprocessed training and testing datasets. 
### Steps
# Step 1: Import libraries and load dataset
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import numpy as np
df = pd.read_csv('/content/drive/MyDrive/Datasets/Data.csv')
# Display dataset
df.head()
# Step 2: Check dataset information
df.info()
print(df.shape)

# Step 3: Separate independent and dependent variables
x = df[['Country', 'Age', 'Salary']]
y = df[['Purchased']].values
# Convert X into array
x = df[['Country', 'Age', 'Salary']].values
# Step 4: Handle missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(
    missing_values=np.nan,
    strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)
# Step 5: Encode categorical data
from sklearn.preprocessing import LabelEncoder
label_encoder_x = LabelEncoder()
x[:, 0] = label_encoder_x.fit_transform(x[:, 0])
print(x)
# Step 6: One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
x_country = onehotencoder.fit_transform( df.Country.values.reshape(-1, 1)
).toarray()
print(x_country)
# Encode dependent variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(y)
# Step 7: Split dataset into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(  x, y, test_size=0.2, random_state=0)
print(x_train)
print(x_test)
print(y_train)
# Step 8: Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
print(x_train)
print(x_test)

## PROGRAM

import pandas as pd
import numpy as np

df = pd.read_csv('/content/drive/MyDrive/Datasets/Data.csv')
print(df.head())
print(df.info())
print(df.shape)
X = df[['Country', 'Age', 'Salary']]
y = df['Purchased']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, ['Age', 'Salary']),
    ('cat', categorical_transformer, ['Country'])
])
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
print("Training Data:")
print(X_train)

print("Testing Data:")
print(X_test)

print("Training Target:")
print(y_train)

print("Testing Target:")
print(y_test)
```

## OUTPUT

```text
Dataset Information:
First five records displayed successfully.

Dataset Shape:
(10, 4)

Training Data:
Preprocessed training dataset displayed successfully.

Testing Data:
Preprocessed testing dataset displayed successfully.

Training Target:
[1 1 1 0 1 0 0 1]

Testing Target:
[0 0]

Training Dataset Shape: (8, 5)
Testing Dataset Shape: (2, 5)

Missing values handled successfully.
Categorical data encoded successfully.
Feature scaling applied successfully.
```


## RESULT

Thus, the dataset was successfully preprocessed using Python and Scikit-learn by handling missing values, encoding categorical variables, splitting the dataset into training and testing sets, and applying feature scaling.

## Conclusion
Thus, the given dataset was successfully preprocessed by handling missing values, encoding categorical variables, splitting the data into training and testing sets, and performing feature scaling.

