# Solution for Predicting Red Hat Business Value

Competition URL: https://www.kaggle.com/c/predicting-red-hat-business-value

Contains feature engineering and 3 classifiers for the competition.

- Multi-Layer Perceptron
- Adaptive Boosting
- Graident Boosting

## Usage

`python adaboost_30.py`

## Available Options

To choose a classifier, in `adaboost_30.py` change `classify_with = n` where n is one of the following:

- 0 for MLP
- 1 for Adaboost
- 2 for Gradientboost

The following variables are available for the classifiers:

`
#Classifier Settings
## No. of estimators for Gradient Boost
no_estimators_gb = 30
## No. of estimators for Adaboost
no_est_adaboost = 30
## No. of MLP Hidden Layers
no_hidden_layers = 30
`