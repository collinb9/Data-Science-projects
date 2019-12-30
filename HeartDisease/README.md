# Heart Disease Data

The two notebooks here are based on [this](https://www.kaggle.com/ronitf/heart-disease-uci) dataset of heart disease patients from kaggle. The [interpretation of data](https://github.com/collinb9/Data-Science-projects/blob/master/HeartDisease/Interpretation%20of%20data.ipynb) notebook aims to investigate the most important risk factors of heart disease by interpreting a logistic regression and random forest model. We use permutation importance, Shapley values, dependency plots and the naive importance measures for these models. This SHAP summary plot gives a good measure of the importances. 



![](https://github.com/collinb9/Data-Science-projects/blob/master/HeartDisease/images/summary_plot.png "SHAP summary plot")
 
 In the [model optimisation](https://github.com/collinb9/Data-Science-projects/blob/master/HeartDisease/ModelOptimisation.ipynb) notebook we go through the process of training, optimising and selecting a good classification model, where we focus on attaining good recall.
