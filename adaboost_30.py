import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,LabelBinarizer,MinMaxScaler,OneHotEncoder
from sklearn.decomposition import TruncatedSVD,NMF,PCA,FactorAnalysis
from sklearn.feature_selection import SelectFromModel,SelectPercentile,f_classif
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# select classifier
## 0 for MLP
## 1 for Adaboost
classify_with = 1

# Load Data

data_path = "data/"
train = pd.read_csv(data_path+'act_train.csv', parse_dates=['date'])
test = pd.read_csv(data_path+'act_test.csv', parse_dates=['date'])
people = pd.read_csv(data_path+'people.csv', parse_dates=['date'])

# Preprocessing, feature selection

## No difference between char_1 and char_2
idx1 = pd.Index(people['char_1'])
idx2 = pd.Index(people['char_2'])
difference = idx1.difference(idx2)
if len(difference) == 0:
    print ("No difference.")
    del people['char_1']


## split date into day month year cols

def split_date(dsname):
    dataset = dsname
    
    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month
    dataset['day'] = dataset['date'].dt.day
    dataset['isweekend'] = (dataset['date'].dt.weekday >= 5).astype(int)
    dataset = dataset.drop('date', axis = 1)
    
    return dataset

## drop char_10 col because negative impact on accuracy
train = train.drop("char_10", axis=1)
test = test.drop("char_10", axis=1)

## split date
train = split_date(train)
test = split_date(test)

## merge people data with train and test
data = pd.concat([train,test])
data = pd.merge(data,people,how='left',on='people_id').fillna('NA') #fill empty cells with NA
train = data[:train.shape[0]]
test = data[train.shape[0]:]

## Encode features
columns = train.columns.tolist()
columns.remove('activity_id')
columns.remove('outcome')
data = pd.concat([train,test])
for c in columns:
    data[c] = LabelEncoder().fit_transform(data[c].values)
    train = data[:train.shape[0]]
    test = data[train.shape[0]:]

del data #save ram
train_labels = train['outcome'] #training labels
del train['outcome'] #remove labels from training set
del train['activity_id'] #dont need activity id in training set, all unique, no impact
train_target = train_labels.rename(None).to_frame() #labels

if classify_with == 0:
    print("Classifying with MLP")
    # Train
    no_hid_lyr = 30
    clf3 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(no_hid_lyr,), random_state=1)
    clf3.fit(train, list(train_target.values.ravel()))
    print("Training completed.\nGenerating predictions...")
    columns = train.columns.tolist()
    X_t10 = test[columns].values
    outcome = clf3.predict_proba(X_t10)
    submission = pd.DataFrame()
    submission['activity_id'] = test['activity_id']
    submission['outcome'] = [max(i) for i in outcome]
    print("Writing submission file.")
    submission.to_csv('submission_MLP_%d.csv'%(no_hid_lyr),index=False)
    print("Submission file written to submission_MLP_%d.csv"%(no_hid_lyr))

elif classify_with == 1:
    print("Classifying with Adaboost.")
    no_estimators = 30
    bdt = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1),
        algorithm="SAMME",
        n_estimators=no_estimators)
    bdt.fit(train, list(train_target.values.ravel()))
    print("Training completed.\nGenerating predictions...")
    columns = train.columns.tolist()
    X_t10 = test[columns].values
    outcome = bdt.predict(X_t10)
    submission = pd.DataFrame()
    submission['activity_id'] = test['activity_id']
    submission['outcome'] = outcome
    print("Writing submission file.")
    submission.to_csv('submission_adaboost_outcome_%d.csv'%(no_estimators),index=False)
    print("Submission file written to submission_adaboost_outcome_%d.csv"%(no_estimators))
