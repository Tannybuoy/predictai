# -*- coding: utf-8 -*-
"""TechCare

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10GbjK9ltqU-x-QRzhv9y44RsCyAnW8Eh
"""

import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

heart = pd.read_csv('heart.csv')

"""Data headings and description

*   **age** - age in years
*   **sex** - (1 = male; 0 = female)
*   **cp** - chest pain type
*   **trestbps** - resting blood pressure (in mm Hg on admission to the hospital)
*   **chol** - serum cholestoral in mg/dl
*   **fbs** - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
*   **restecg** - resting electrocardiographic results
*   **thalach** - maximum heart rate achieved
*   **exang** - exercise induced angina (1 = yes; 0 = no)
*   **oldpeak** - ST depression induced by exercise relative to rest
*   **slope** - the slope of the peak exercise ST segment
*   **ca** - number of major vessels (0-3) colored by flourosopy
*   **thal** - 3 = normal; 6 = fixed defect; 7 = reversable defect
*   **target** - have disease or not (1=yes, 0=no)
"""

df = heart[['age','sex', 'trestbps','chol', 'fbs', 'target']]
# Variables
x_data= df.drop(labels= 'target', axis= 1)
y= df['target']

#Data Normalisation
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

# Splitting the Dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)

# Instantiating LogisticRegression() Model
lor = LogisticRegression()

# Training/Fitting the Model
lor.fit(x_train,y_train)

pickle.dump(lor, open('iri.pkl', 'wb'))

# Making Predictions
#lor.predict(x_test)
#pred = lor.predict(x_test)

#Accuracy of the model
#acc = lor.score(x_test,y_test)*100

#print("Test Accuracy {:.2f}%".format(acc))

# Evaluating Model's Performance
#print('Mean Absolute Error:', mean_absolute_error(y_test, pred))
#print('Mean Squared Error:', mean_squared_error(y_test, pred))
#print('Mean Root Squared Error:', np.sqrt(mean_squared_error(y_test, pred)))