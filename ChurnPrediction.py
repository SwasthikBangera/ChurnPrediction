'''# Churn Predition Model'''

# Import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

churn_data = pd.read_csv('https://github.com/SwasthikBangera/ChurnPrediction/blob/main/train.csv')

# Check the dataset
print(churn_data.head())

# Name of the features
print(churn_data.head(0))

# Alternatively use to name all columns in the list
print(f"{churn_data.columns.tolist()}\n")


# Number of empty/missing values
print(churn_data.isnull().sum())


# Total number of values in the dataset
print(f"\nThe number of rows are: {churn_data.shape[0]}")
print(f"\nThe number of columns are: {churn_data.shape[1]}\n")

# Convert categorical data to numerical 
cat_features = churn_data[['state','voice_mail_plan','international_plan','area_code','churn']]
num_features = churn_data.drop(['state','voice_mail_plan','international_plan','area_code','churn'],axis=1)

print(cat_features.head())
print(num_features.head())

le = preprocessing.LabelEncoder()
churn_cat = cat_features.apply(le.fit_transform)

churn_data = pd.merge(churn_cat, num_features,left_index=True, right_index=True)

''' Alternaitvely could have been done using Label Encoder

# Type of Target data column - churn
print(churn_data['churn'].value_counts())
print(churn_data['area_code'].value_counts())
print(churn_data['international_plan'].value_counts())
print(churn_data['voice_mail_plan'].value_counts())

# Convert categorical value to numerical value in target (churn) column
churn_data = churn_data.replace({'churn':{'yes':1,'no':0}})
churn_data = churn_data.replace({'voice_mail_plan':{'yes':1,'no':0}})
churn_data = churn_data.replace({'international_plan':{'yes':1,'no':0}})
churn_data = churn_data.replace({'area_code':{'area_code_415':0, 'area_code_408':1, 'area_code_510':2}})
 
'''


# Information of all the features in the dataset
print(churn_data.info())

# Checking the data types of each variable
for var in churn_data.head(0):
    print((churn_data[var]).value_counts())

# Statistical data of the parameters in data set
print(churn_data.describe())



''' Plot the data ''' 


# Plot counts for Area code and plans 
cols = ['area_code','international_plan',"voice_mail_plan"]
numerical = cols

plt.figure(figsize=(10,2))

for i, col in enumerate(numerical):
    ax = plt.subplot(1, len(numerical), i+1)
    sns.countplot(x=str(col), data=churn_data)
    ax.set_title(f"{col}")

# Plotting churn vs total charge per customer

plt.figure(figsize=(10,2))
churn_data['daily_charge'] = churn_data['total_eve_charge'] + churn_data['total_day_charge'] + churn_data['total_night_charge'] + churn_data['total_intl_charge']
sns.boxplot(x='churn', y='daily_charge', data=churn_data).set_title("Daily charges vs churn")



# Plot churn vs plans
cols_plot2 = ['international_plan',"voice_mail_plan"]

plt.figure(figsize=(14,4))

for i, col in enumerate(cols_plot2):
    ax = plt.subplot(1, len(cols_plot2), i+1)
    sns.countplot(x ="churn", hue = str(col), data = churn_data)
    ax.set_title(f"{col}")
    
''' Model Fitting '''

# Seperate features and target

X = churn_data.drop(['churn'],axis=1)
y = churn_data['churn']

# Seperate dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=42 )

print(X.shape, X_train.shape, X_test.shape)
print(y.shape, y_train.shape, y_test.shape)

'''

# Oversampling the dataset
from imblearn.over_sampling import SMOTE

oversample = SMOTE(k_neighbors=5)
X_smote, y_smote = oversample.fit_resample(X_train, y_train)
X_train, y_train = X_smote, y_smote

'''

# Building the Customer Churn Prediction Model
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=46)
rf.fit(X_train,y_train)

# Customer Churn Prediction Model Evaluation

from sklearn.metrics import accuracy_score

preds = rf.predict(X_test)
print(f" The accuracy score is {accuracy_score(preds,y_test)}")

''' Determine effect of daily_charge feature - Feature Engineering '''

# Delete variable daily charge
 
print(churn_data['daily_charge'])

# Seperate features and target

X_dc = X.drop(['daily_charge'],axis=1)
y = churn_data['churn']

# Seperate dataset into training and test data
X_dc_train, X_dc_test, y_train, y_test = train_test_split(X_dc, y, test_size=0.2,  random_state=42 )

print(X.shape, X_dc_train.shape, X_dc_test.shape)
print(y.shape, y_train.shape, y_test.shape)

'''

# Oversampling the dataset
from imblearn.over_sampling import SMOTE

oversample = SMOTE(k_neighbors=5)
X_smote, y_smote = oversample.fit_resample(X_train, y_train)
X_train, y_train = X_smote, y_smote

'''

# Building the Customer Churn Prediction Model
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=46)
rf.fit(X_dc_train,y_train)

# Customer Churn Prediction Model Evaluation

from sklearn.metrics import accuracy_score

preds_dc = rf.predict(X_dc_test)
print(f" The accuracy score with daily_charge is {accuracy_score(preds,y_test)} and without daily_charge is {accuracy_score(preds_dc,y_test)}")
print("'Applied machine learning' is basically feature engineering. — Prof. Andrew Ng")
