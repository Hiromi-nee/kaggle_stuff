# Solution for Predicting Red Hat Business Value

Competition URL: https://www.kaggle.com/c/predicting-red-hat-business-value

Contains feature engineering and 3 classifiers for the competition.

- Multi-Layer Perceptron
- Adaptive Boosting
- Gradient Boosting
- Random Forest 

## Requirements

- Scikit-learn
- numpy


## Usage

`python redhat_classifiers.py`

## Available Options

To choose a classifier, in `redhat_classifiers.py` change `classify_with = n` where n is one of the following:

- 0 for MLP
- 1 for Adaboost
- 2 for Gradientboost
- 3 for Random Forest

The following variables are available for the classifiers:


#Classifier Settings

## No. of estimators for Gradient Boost
`no_estimators_gb = 30`
## No. of estimators for Adaboost
`no_est_adaboost = 30`
## No. of MLP Hidden Layers
`no_hidden_layers = 30`
## No. of Random Forest estimators
`no_est_rf = 100`

## Program Flow

1. Preprocessing & Feature Engineering
2. Training of chosen classifier
3. Classifying
4. Generation of submission .csv file
