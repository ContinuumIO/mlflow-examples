# Fit a linear regression model using Keras to a randomly created dataset
# that resembles the 'adult' dataset available here:
# https://archive.ics.uci.edu/ml/datasets/Adult

import pandas as pd
import numpy as np
from scipy.stats import norm
import random
import array
import os
import sys
import math

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.losses import MeanSquaredLogarithmicError 

import tensorflow as tf
from keras import optimizers
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature


# This is the response variable in the model.
TARGET_NAME = 'salary'

# Create a dataframe of 50000 rows
nrows = 50000


# The next few rows create the columns in the dataset with random data.
age = norm.ppf(np.random.random(nrows), loc = 33, scale=10).astype(int)       
workclass = np.random.choice(['Private', 'Self-emp', 'Federal-gov', 'Local-gov', 'State-gov', 'W/o pay', 'Never-worked'], nrows, p = [0.4, 0.15, 0.15, 0.05, 0.1, 0.1, 0.05], replace = True)
education = np.random.choice(['Bachelors', 'Some-college', 'HS-grad', 'Prof-school', 'Masters', 'Doctorate'], nrows, 
                             p = [0.3, 0.2, 0.25, 0.1, 0.1, 0.05], replace = True)
occupation = np.random.choice(['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
                               'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Armed-Forces'], nrows,
                            p = [0.02, 0.1, 0.2, 0.05, 0.1, 0.05, 0.08, 0.18, 0.04, 0.08, 0.02, 0.08], replace = True)
hoursperweek = np.zeros(nrows)
salary = np.zeros(nrows)

dict = {'age':age, 'workclass': workclass, 'education': education, 'occupation': occupation, 'hoursperweek': hoursperweek, 'salary': salary}
salarydf = pd.DataFrame(dict)

# This the main function to populate the work hours and salary info based on the 'occupation' column.
def hours_and_salary_calc(df, numvalues, indices, loc1, scale1, loc2, scale2):
    for i in range(0, numvalues):
       hoursval = int(norm.ppf(np.random.random(1), loc=loc1, scale=scale1))
       salarydf.iloc[[indices[0][i]], 4] = hoursval
       salval = int(norm.ppf(np.random.random(1), loc=loc2, scale=scale2))
       salarydf.iloc[[indices[0][i]], 5] = salval
    return salarydf

salarydf = salarydf[age>=16]

privbachtech = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Tech-support')
numpbt = len(privbachtech[privbachtech==True])
indicespbt = np.where(privbachtech==True)
loc1 = 40
scale1 = 5
loc2 = 105000
scale2 = 8
hours_and_salary_calc(salarydf, numpbt, indicespbt, loc1, scale1, loc2, scale2)

privbachcr = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Craft-repair')
numpbcr = len(privbachcr[privbachcr==True])
indicespbcr = np.where(privbachcr==True)
loc1 = 50
scale1 = 5
loc2 = 130000
scale2 = 8
hours_and_salary_calc(salarydf, numpbcr, indicespbcr, loc1, scale1, loc2, scale2)

privbachos = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Other-service')
numpbos = len(privbachos[privbachos==True])
indicespbos = np.where(privbachos==True)
loc1 = 40
scale1 = 7
loc2 = 90000
scale2 = 10
hours_and_salary_calc(salarydf, numpbos, indicespbos, loc1, scale1, loc2, scale2)

privbachsales = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Sales')
numpbsales = len(privbachsales[privbachsales==True])
indicespbsales = np.where(privbachsales==True)
loc1 = 45
scale1 = 7
loc2 = 170000
scale2 = 20
hours_and_salary_calc(salarydf, numpbsales, indicespbsales, loc1, scale1, loc2, scale2)

privbachexec = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Exec-managerial')
numpbexec = len(privbachexec[privbachexec==True])
indicespbexec = np.where(privbachexec==True)
loc1 = 50
scale1 = 5
loc2 = 200000
scale2 = 30
hours_and_salary_calc(salarydf, numpbexec, indicespbexec, loc1, scale1, loc2, scale2)

privbachprof = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Prof-specialty')
numpbprof = len(privbachprof[privbachprof==True])
indicespbprof = np.where(privbachprof==True)
loc1 = 40
scale1 = 5
loc2 = 110000
scale2 = 10
hours_and_salary_calc(salarydf, numpbprof, indicespbprof, loc1, scale1, loc2, scale2)

privbachadm = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Adm-clerical')
numpbadm = len(privbachadm[privbachadm==True])
indicesadm = np.where(privbachadm==True)
loc1 = 45
scale1 = 3
loc2 = 90000
scale2 = 5
hours_and_salary_calc(salarydf, numpbadm, indicesadm, loc1, scale1, loc2, scale2)

privbachff = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Farming-fishing')
numpbff = len(privbachff[privbachff==True])
indicesff = np.where(privbachff==True)
loc1 = 40
scale1 = 5
loc2 = 95000
scale2 = 5
hours_and_salary_calc(salarydf, numpbff, indicesff, loc1, scale1, loc2, scale2)

privbachtm = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Transport-moving')
numpbtm = len(privbachtm[privbachtm==True])
indicestm = np.where(privbachtm==True)
loc1 = 50
scale1 = 3
loc2 = 95000
scale2 = 7
hours_and_salary_calc(salarydf, numpbtm, indicestm, loc1, scale1, loc2, scale2)

privbachhs = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Priv-house-serv')
numpbhs = len(privbachhs[privbachhs==True])
indiceshs = np.where(privbachhs==True)
loc1 = 40
scale1 = 3
loc2 = 45000
scale2 = 4
hours_and_salary_calc(salarydf, numpbhs, indiceshs, loc1, scale1, loc2, scale2)

privbachaf = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Armed-Forces')
numpbaf = len(privbachaf[privbachaf==True])
indicesaf = np.where(privbachaf==True)
loc1 = 40
scale1 = 5
loc2 = 65000
scale2 = 4
hours_and_salary_calc(salarydf, numpbaf, indicesaf, loc1, scale1, loc2, scale2)

privhscr = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'HS-grad') & (salarydf["occupation"] == 'Craft-repair')
numhscr = len(privhscr[privhscr==True])
indicescr = np.where(privhscr==True)
loc1 = 50
scale1 = 7
loc2 = 110000
scale2 = 10
hours_and_salary_calc(salarydf, numhscr, indicescr, loc1, scale1, loc2, scale2)

privhsos = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'HS-grad') & (salarydf["occupation"] == 'Other-service')
numhsos = len(privhsos[privhsos==True])
indiceshsos = np.where(privhsos==True)
loc1 = 45
scale1 = 5
loc2 = 70000
scale2 = 8
hours_and_salary_calc(salarydf, numhsos, indiceshsos, loc1, scale1, loc2, scale2)

privhsps = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'HS-grad') & (salarydf["occupation"] == 'Prof-specialty')
numhsps = len(privhsps[privhsps==True])
indiceshsps = np.where(privhsps==True)
loc1 = 50
scale1 = 5
loc2 = 90000
scale2 = 10
hours_and_salary_calc(salarydf, numhsps, indiceshsps, loc1, scale1, loc2, scale2)

privhsmo = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'HS-grad') & (salarydf["occupation"] == 'Machine-op-inspct')
numhsmo = len(privhsmo[privhsmo==True])
indiceshsmo = np.where(privhsmo==True)
loc1 = 50
scale1 = 3
loc2 = 100000
scale2 = 7
hours_and_salary_calc(salarydf, numhsmo, indiceshsmo, loc1, scale1, loc2, scale2)

privhstm = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'HS-grad') & (salarydf["occupation"] == 'Transport-moving')
numhstm = len(privhstm[privhstm==True])
indiceshstm = np.where(privhstm==True)
loc1 = 50
scale1 = 3
loc2 = 85000
scale2 = 5
hours_and_salary_calc(salarydf, numhstm, indiceshstm, loc1, scale1, loc2, scale2)

privhshs = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'HS-grad') & (salarydf["occupation"] == 'Priv-house-serv')
numhsphs = len(privhshs[privhshs==True])
indiceshsphs = np.where(privhshs==True)
loc1 = 40
scale1 = 3
loc2 = 38000
scale2 = 2
hours_and_salary_calc(salarydf, numhsphs, indiceshsphs, loc1, scale1, loc2, scale2)

privhsaf = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'HS-grad') & (salarydf["occupation"] == 'Armed Forces')
numhsaf = len(privhsaf[privhsaf==True])
indiceshsaf = np.where(privhsaf==True)
loc1 = 40
scale1 = 7
loc2 = 45000
scale2 = 6
hours_and_salary_calc(salarydf, numhsaf, indiceshsaf, loc1, scale1, loc2, scale2)

privdocos = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'Doctorate') & (salarydf["occupation"] == 'Other-service')
numdocos = len(privdocos[privdocos==True])
indicesdocos = np.where(privdocos==True)
loc1 = 40
scale1 = 7
loc2 = 110000
scale2 = 8
hours_and_salary_calc(salarydf, numdocos, indicesdocos, loc1, scale1, loc2, scale2)

privdocem = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'Doctorate') & (salarydf["occupation"] == 'Exec-managerial')
numdocem = len(privdocem[privdocem==True])
indicesdocem = np.where(privdocem==True)
loc1 = 60
scale1 = 5
loc2 = 410000
scale2 = 15
hours_and_salary_calc(salarydf, numdocem, indicesdocem, loc1, scale1, loc2, scale2)

privdocps = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'Doctorate') & (salarydf["occupation"] == 'Prof-specialty')
numdocps = len(privdocps[privdocps==True])
indicesdocps = np.where(privdocps==True)
loc1 = 50
scale1 = 5
loc2 = 165000
scale2 = 7
hours_and_salary_calc(salarydf, numdocps, indicesdocps, loc1, scale1, loc2, scale2)

privdocff = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'Doctorate') & (salarydf["occupation"] == 'Farming-fishing')
numdocff = len(privdocff[privdocff==True])
indicesdocff = np.where(privdocff==True)
loc1 = 45
scale1 = 3
loc2 = 125000
scale2 = 5
hours_and_salary_calc(salarydf, numdocff, indicesdocff, loc1, scale1, loc2, scale2)

privdocaf = (salarydf["workclass"] == 'Private') & (salarydf["education"] == 'Doctorate') & (salarydf["occupation"] == 'Armed Forces')
numdocaf = len(privdocaf[privdocaf==True])
indicesdocaf = np.where(privdocaf==True)
loc1 = 50
scale1 = 5
loc2 = 135000
scale2 = 10
hours_and_salary_calc(salarydf, numdocaf, indicesdocaf, loc1, scale1, loc2, scale2)

sempbachcr = (salarydf["workclass"] == 'Self-emp') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Craft-repair')
numbachcr = len(sempbachcr[sempbachcr==True])
indicesbachcr = np.where(sempbachcr==True)
loc1 = 55
scale1 = 3
loc2 = 165000
scale2 = 10
hours_and_salary_calc(salarydf, numbachcr, indicesbachcr, loc1, scale1, loc2, scale2)

sempbachos = (salarydf["workclass"] == 'Self-emp') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Other-service')
numbachos = len(sempbachos[sempbachos==True])
indicesbachos = np.where(sempbachos==True)
loc1 = 55
scale1 = 5
loc2 = 155000
scale2 = 10
hours_and_salary_calc(salarydf, numbachos, indicesbachos, loc1, scale1, loc2, scale2)

sempbachps = (salarydf["workclass"] == 'Self-emp') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Prof-specialty')
numbachps = len(sempbachps[sempbachps==True])
indicesbachps = np.where(sempbachps==True)
loc1 = 55
scale1 = 7
loc2 = 190000
scale2 = 15
hours_and_salary_calc(salarydf, numbachps, indicesbachps, loc1, scale1, loc2, scale2)

sempbachtm = (salarydf["workclass"] == 'Self-emp') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Transport-moving')
numbachtm = len(sempbachtm[sempbachtm==True])
indicesbachtm = np.where(sempbachtm==True)
loc1 = 50
scale1 = 5
loc2 = 170000
scale2 = 20
hours_and_salary_calc(salarydf, numbachtm, indicesbachtm, loc1, scale1, loc2, scale2)

sempbachphs = (salarydf["workclass"] == 'Self-emp') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Priv-house-serv')
numbachphs = len(sempbachphs[sempbachphs==True])
indicesbachphs = np.where(sempbachphs==True)
loc1 = 45
scale1 = 5
loc2 = 65000
scale2 = 5
hours_and_salary_calc(salarydf, numbachphs, indicesbachphs, loc1, scale1, loc2, scale2)

fgbachos = (salarydf["workclass"] == 'Federal-gov') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Other-service')
numfgbachos = len(fgbachos[fgbachos==True])
indicesfgbachos = np.where(fgbachos==True)
loc1 = 48
scale1 = 5
loc2 = 70000
scale2 = 3
hours_and_salary_calc(salarydf, numfgbachos, indicesfgbachos, loc1, scale1, loc2, scale2)

fgbachem = (salarydf["workclass"] == 'Federal-gov') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Exec-managerial')
numfgbachem = len(fgbachem[fgbachem==True])
indicesfgbachem = np.where(fgbachem==True)
loc1 = 50
scale1 = 2
loc2 = 140000
scale2 = 15
hours_and_salary_calc(salarydf, numfgbachem, indicesfgbachem, loc1, scale1, loc2, scale2)

fgbachadmc = (salarydf["workclass"] == 'Federal-gov') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Adm-clerical')
numfgbachadmc = len(fgbachadmc[fgbachadmc==True])
indicesfgbachadmc = np.where(fgbachadmc==True)
loc1 = 40
scale1 = 5
loc2 = 70000
scale2 = 8
hours_and_salary_calc(salarydf, numfgbachadmc, indicesfgbachadmc, loc1, scale1, loc2, scale2)

fgbachaf = (salarydf["workclass"] == 'Federal-gov') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Armed Forces')
numfgbachaf = len(fgbachaf[fgbachaf==True])
indicesfgbachaf = np.where(fgbachaf==True)
loc1 = 40
scale1 = 8
loc2 = 55000
scale2 = 5
hours_and_salary_calc(salarydf, numfgbachaf, indicesfgbachaf, loc1, scale1, loc2, scale2)

sgbachts = (salarydf["workclass"] == 'State-gov') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Tech-support')
numsgbachts = len(sgbachts[sgbachts==True])
indicessgbachts = np.where(sgbachts==True)
loc1 = 40
scale1 = 5
loc2 = 90000
scale2 = 8
hours_and_salary_calc(salarydf, numsgbachts, indicessgbachts, loc1, scale1, loc2, scale2)

sgbachos = (salarydf["workclass"] == 'State-gov') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Other-service')
numsgbachos = len(sgbachos[sgbachos==True])
indicessgbachos = np.where(sgbachos==True)
loc1 = 40
scale1 = 3
loc2 = 65000
scale2 = 8
hours_and_salary_calc(salarydf, numsgbachos, indicessgbachos, loc1, scale1, loc2, scale2)

sgbachem = (salarydf["workclass"] == 'State-gov') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Exec-managerial')
numsgbachem = len(sgbachem[sgbachem==True])
indicessgbachem = np.where(sgbachem==True)
loc1 = 50
scale1 = 3
loc2 = 120000
scale2 = 15
hours_and_salary_calc(salarydf, numsgbachem, indicessgbachem, loc1, scale1, loc2, scale2)

sgbachps = (salarydf["workclass"] == 'State-gov') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Prof-specialty')
numsgbachps = len(sgbachps[sgbachps==True])
indicessgbachps = np.where(sgbachps==True)
loc1 = 45
scale1 = 5
loc2 = 85000
scale2 = 10
hours_and_salary_calc(salarydf, numsgbachps, indicessgbachps, loc1, scale1, loc2, scale2)

sgbachmop = (salarydf["workclass"] == 'State-gov') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Machine-op-inspct')
numsgbachmop = len(sgbachmop[sgbachmop==True])
indicessgbachmop = np.where(sgbachmop==True)
loc1 = 50
scale1 = 3
loc2 = 85000
scale2 = 5
hours_and_salary_calc(salarydf, numsgbachmop, indicessgbachmop, loc1, scale1, loc2, scale2)

sgbachadm = (salarydf["workclass"] == 'State-gov') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Adm-clerical')
numsgbachadm = len(sgbachadm[sgbachadm==True])
indicessgbachadm = np.where(sgbachadm==True)
loc1 = 45
scale1 = 5
loc2 = 60000
scale2 = 8
hours_and_salary_calc(salarydf, numsgbachadm, indicessgbachadm, loc1, scale1, loc2, scale2)

sgbachff = (salarydf["workclass"] == 'State-gov') & (salarydf["education"] == 'Bachelors') & (salarydf["occupation"] == 'Farming-fishing')
numsgbachff = len(sgbachff[sgbachff==True])
indicessgbachff = np.where(sgbachff==True)
loc1 = 45
scale1 = 5
loc2 = 70000
scale1 = 5
hours_and_salary_calc(salarydf, numsgbachff, indicessgbachff, loc1, scale1, loc2, scale2)


# x_train = features, y_train = target
y = salarydf[TARGET_NAME]
# Create a trainings and test dataset.
x_train, x_test, y_train, y_test = train_test_split(salarydf.drop(TARGET_NAME, axis=1), y, test_size=0.33, random_state=42)

# One-Hot Encode categorical columns in salarydf
ct = ColumnTransformer([('one-hot-encoder', OneHotEncoder(), ['workclass', 'education', 'occupation'])], remainder='passthrough')

# Execute Fit_Transform
x_train_trans = ct.fit_transform(x_train)
x_test_trans = ct.fit_transform(x_test)

# Now let's create a keras Neural Networks model to mimic linear regression on this data.
hidden_units1 = 160
hidden_units2 = 480
hidden_units3 = 256
learning_rate = 256

# Creating model using the Sequential in tensorflow
def build_model_using_sequential():
  model = Sequential([
      Dense(hidden_units1, kernel_initializer='normal', activation='relu'), Dropout(0.2),
      Dense(hidden_units2, kernel_initializer='normal', activation='relu'), Dropout(0.2),
      Dense(hidden_units3, kernel_initializer='normal', activation='relu'),
      Dense(1, kernel_initializer='normal', activation='linear')
  ])
  return model

# build the model
mlflow.tensorflow.autolog()
model = build_model_using_sequential()

# loss function
msle = MeanSquaredLogarithmicError()
model.compile(loss=msle, optimizer=Adam(lr=learning_rate), metrics=["msle"])
              
# train the model
with mlflow.start_run() as run:
  history = model.fit(x_train_trans, y_train.values, epochs=10, batch_size=64)
  loss = history.history['loss']
  print("Active run_id: {}".format(run.info.run_id))
  signature = infer_signature(x_test_trans, model.predict(x_test_trans))
  mlflow.tensorflow.log_model(model, "tflinreg", signature=signature)

  logged_model = 'runs:/' + run.info.run_id + '/model'  
  loaded_model = mlflow.pyfunc.load_model(logged_model)
  # Compute model predictions
  preds = loaded_model.predict(x_test_trans)
  print("Computed predictions for fitted model:")
  print(preds)
    
  
    
    
                    









   





    