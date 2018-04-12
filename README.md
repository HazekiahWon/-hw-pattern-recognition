# Features
this repo is an implementation of the Bayesian classifier, featuring
1. a base class template named `Bayes`
2. a sub class implementing `Naive Bayes` termed `NaiveBayes`
3. a sub class termed `multinomial_Bayes` estimates samples using the multinomial Gaussian distribution
4. a sub class named `Bayes4Test` adapted to the given dataset with custom `_read_data()`
5. a sub class named `Bayes4Test_PCA` further incorporates PCA dimensionality reduction while inheriting from the `Bayes4Test`

# Additional Files
- main.py  
you will find all codes here  
can be safely imported as a small library for Bayes  
- logger_*.txt
the logger file recording information during training, validation and testing phase  
Note that you can turn some off by switching the `verbose` param of `test()`
- *.xls
these are demanded by Prof. Zhang 
- validation.png
the plotting of validation accuracy corresponding different `n_components` for PCA

# Datasets
- adult.data & adult.test
[dataset link](http://archive.ics.uci.edu/ml/datasets/Adult)  
the original dataset used for experiments, found on UCI.  
this is a blend dataset of both discrete and continuous attributes, yet a binary classification.
- given datasets
    1. balance
    2. uspst
# Statement
this code is written by Hazekiah Wang.

all rights reserved.
