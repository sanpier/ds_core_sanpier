# Data Science Core Functionalities

Modules included in this package:
 * 1) regressor_utils.py
 * 2) classifier_utils.py
 * 3) exploratory_data_analyzer.py
 
----

## 1) regressor_utils.py

Containing Regressor class which has certain type of functions that make life easier for regression problems.
These functions are quite various for different type of problems. The following functionalities are included in this module:
 - Data splitting 
 - Oversampling
 - Experimenting different regression algorithms
 - Training given model
 - Calculating residual difference between the target feature and predicted or calculated feature with visualization
 - Regression plots
 - Regression scoring metrics
 - Quantile regression

## 2) classifier_utils.py

Having Classifier class which contains a set of functions for modeling ML classification problems in the shortest time.
The functions included in the class are quite various, these can be seen as follows:
 - Data splitting 
 - Experimenting different regression algorithms
 - Training given model
 - Cross validation score of the given model
 - Confusion matrix visualization
 - Creating a stack model
 - Evaluating model in the test dataset with classification metrics

## 3) exploratory_data_analyzer.py

This module has EDA_Preprocessor class in it where the class functions serve as a baseline for all kinds of EDA.
The functions in this module are including the following analysis tasks:
 - filling missing values in the data
 - showing distributions / counts of the columns
 - dummification of the categorical data columns
 - PCA decomposition of the given data
 - standardization of the data
 - applying transformation function for handling data skewness
 - showing heatmap correlation of the features before modeling
 - checking the correlation of the categorical features compare to target feature
 - feature importances of a default model in the given problem domain
