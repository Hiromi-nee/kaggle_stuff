import pandas as pd
import numpy as np
from scipy import sparse as ssp
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder,LabelBinarizer,MinMaxScaler,OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import TruncatedSVD,NMF,PCA,FactorAnalysis
from sklearn.feature_selection import SelectFromModel,SelectPercentile,f_classif
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline,make_pipeline

data_path = "data/" #folder where your csv files are stored in

#read csv files into pandas dataframe
train = pd.read_csv(data_path+'act_train.csv')
test = pd.read_csv(data_path+'act_test.csv')
people = pd.read_csv(data_path+'people.csv')

## START DATA PREPROCESSING
##
#char_1 and char_2 col of people data are the same so remove one
idx1 = pd.Index(people['char_1'])
idx2 = pd.Index(people['char_2'])
difference = idx1.difference(idx2) #no difference
del people['char_1'] #remove 1

#left join people data with test/train
data = pd.concat([train,test])
data = pd.merge(data,people,how='left',on='people_id').fillna('missing') #fill empty cells with missing
train = data[:train.shape[0]]
test = data[train.shape[0]:]

columns = train.columns.tolist()
columns.remove('activity_id') #dont need this to train
columns.remove('outcome') #dont need this to train, only verify
data = pd.concat([train,test])
#encode text to integers
for c in columns:
    data[c] = LabelEncoder().fit_transform(data[c].values)
train = data[:train.shape[0]]
test = data[train.shape[0]:]

train_labels = train['outcome'] #training labels
del train['outcome'] #remove labels from training set
del train['activity_id'] #dont need activity id in training set, all unique, no impact

#for some reason char_10 column reduces accuracy, so remove
train_no_char10 = train
del train_no_char10['char_10_x']

##END DATA PREPROCESSING
###

#--------------------------------
#training set is train_no_char10    //X
#target labels is list(train_target.values.ravel())  //Y
#-------------------------------

##Training time
##SAMPLE TRAINING WITH Multi Layer Perceptron
#init mlpclassifier
clf3 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64,), random_state=1)

#train
clf3.fit(train_no_char10, list(train_target.values.ravel()))

#output to submission format csv
columns = train_no_char10.columns.tolist()
X_t10 = test[columns].values
outcome = clf3.predict_proba(X_t10)
submission = pd.DataFrame()
submission['activity_id'] = activity_id
submission['outcome'] = [max(i) for i in outcome]
submission.to_csv('submission_nochar10_%d.csv'%(64),index=False)

##END TRAINING

#Save weights
joblib.dump(clf3, 'mlp_64_nochar10.pkl')

#load weights
def load(): #load weights
    clf = joblib.load('mlp_64_nochar10.pkl')
    return clf
