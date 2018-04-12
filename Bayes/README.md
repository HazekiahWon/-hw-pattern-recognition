# Features
this repo is an implementation of the Bayesian classifier, featuring
1. `Bayes`  
a base class template
2. `NaiveBayes`  
a sub class implementing `Naive Bayes`
3. `multinomial_Bayes`  
a sub class that estimates samples using the multinomial Gaussian distribution
4. `Bayes4Test`  
a sub class adapted to the given dataset with custom `_read_data()`
5. `Bayes4Test_PCA`  
a sub class that further incorporates PCA dimensionality reduction while inheriting from the `Bayes4Test`

# Main program
- main.py  
you will find all codes here  
can be safely imported as a small library for Bayes 

# Additional Files 
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
