# Project: Finding Donors for CharityML

### By: Mohamed Refaiy

##### Dec 2021

## Table of Contents
<ul>
<li><a href="#intro">Introduction</a></li>
<li><a href="#eda">Exploring the Data</a></li>
<li><a href="#wrangling">Data Wrangling</a></li>
<li><a href="#learning_model">Data Wrangling</a></li>
<li><a href="#conclusions">Conclusions</a></li>
</ul>


<a id='intro'></a>
## Introduction  

> This work is dedicated for **Finding Donors for CharityML**. In this project, I will employ several supervised algorithms of your choice to accurately model individuals' income using data collected from the 1994 U.S. Census.
You will then choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data. Your goal with this implementation is to construct a model that accurately predicts whether an individual makes more than $50,000. 
This sort of task can arise in a non-profit setting, where organizations survive on donations. Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with. While it can be difficult to determine an individual's general income bracket directly from public sources, we can (as we will see) infer this value from other publically available features. 
The dataset for this project originates from the UCI Machine Learning Repository. The datset was donated by Ron Kohavi and Barry Becker, after being published in the article "Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid". You can find the article by Ron Kohavi online. The data we investigate here consists of small changes to the original dataset, such as removing the 'fnlwgt' feature and records with missing or ill-formatted entries.


<a id='eda'></a>
## Exploring the Data

> This dataset composed of nearly 45k records. 

**Featureset Exploration**

* **age**: continuous. 
* **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
* **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
* **education-num**: continuous. 
* **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
* **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
* **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
* **race**: Black, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other. 
* **sex**: Female, Male. 
* **capital-gain**: continuous. 
* **capital-loss**: continuous. 
* **hours-per-week**: continuous. 
* **native-country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.




<a id='wrangling'></a>
## Data Wrangling

> In this section I will do some data wrangling to prepare data for deploying the machine learning models:
- Transforming Skewed Continuous Features.
- Normalizing Numerical Features.
- Implementation: Data Preprocessing.
- Shuffle and Split Data.



<a id='learning_model'></a>
##  Supervised Learning Models
> **The following are some of the supervised learning models that are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**
- Gaussian Naive Bayes (GaussianNB)
- Decision Trees
- Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
- K-Nearest Neighbors (KNeighbors)
- Stochastic Gradient Descent Classifier (SGDC)
- Support Vector Machines (SVM)
- Logistic Regression


> As the prediction target here is to find; if the person income is more than 50K or not so we will use **Classfication Model**


<li><a href="#KNeighbors">K-Nearest Neighbors (KNeighbors)</a></li>
<li><a href="#AdaBoost">AdaBoost </a></li>
<li><a href="#Random_Forest">Random Forest </a></li>

<a id='KNeighbors'></a>
### K-Nearest Neighbors (KNeighbors)
> **Real-world application**
>Economic forecasting, data compression and genetics. 


>**Model strengths**
>
>1. Simple to understand and impelment
>1. No assumption about data (for e.g. in case of linear regression we assume dependent variable and independent variables are linearly related, in Naïve Bayes we assume features are independent of each other etc., but k-NN makes no assumptions about data)
>1. Constantly evolving model: When it is exposed to new data, it changes to accommodate the new data points.
>1. Multi-class problems can also be solved.
>1. One Hyper Parameter: K-NN might take some time while selecting the first hyper parameter but after that rest of the parameters are aligned to it.

>**Model weaknesses**
>
>1. Slow for large datasets.
>1. Curse of dimensionality: Does not work very well on datasets with large number of features.
>1. Scaling of data absolute must.
>1. Does not work well on Imbalanced data. So before using k-NN either undersamplemajority class or oversample minority class and have a balanced dataset.
>1. Sensitive to outliers.
>1. Can’t deal well with missing values


>**Why to use this Model**
The KNN algorithm is a type of lazy learning, where the computation for the generation of the predictions is deferred until classification. Although this method increases the costs of computation compared to other algorithms, KNN is still the better choice for applications where predictions are not requested frequently but where accuracy is important.
<br>
><a href="https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor//" target="_blank">Ref 1</a>
><br>
><a href="https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/ " target="_blank">Ref 2</a>
><br>
><a href="https://www.ibm.com/docs/en/ias?topic=knn-usage " target="_blank">Ref 3</a>



<a id='AdaBoost'></a>
### AdaBoost

> **Real-world application**
>- predicting customer churn and classifying the types of topics customers are talking/calling about.

>**Model strengths**
>
>1. Easy to implement.
>1. Iteratively corrects the mistakes of the weak classifier and improves accuracy by combining weak learners.
>1. Use many base classifiers. 
>1. Not prone to overfitting.


>**Model weaknesses**
>
>1. Sensitive to noise data.
>1. Highly affected by outliers because it tries to fit each point perfectly.
>1. Slower compared to XGBoost.


>**Why to use this Model**
>
>1. Number of features is large.
<br>
><a href="https://hackernoon.com/under-the-hood-of-adaboost-8eb499d78eab" target="_blank">Ref 1</a>
><br>
><a href="https://www.datacamp.com/community/tutorials/adaboost-classifier-python " target="_blank">Ref 2</a>
><br>
><a href="http://users.cecs.anu.edu.au/~wanglei/SPR_course/boosting.pdf/" target="_blank">Ref 3</a>



<a id='Random_Forest'></a>
### Random Forest

> **Real-world application**
>- Credit card default.
>- Fraud customer/not.
>- Easy to identify patient’s disease or not.
>- Recommendation system for ecommerce sites.


>**Model strengths**
>
>1. Random forest algorithm is not biased.
>1. Reduced error.
>1. Good Performance on Imbalanced datasets.
>1. Handling of huge amount of data.
>1. Good handling of missing data. 
>1. Little impact of outliers.
>1. No problem of overfitting.
>1. Useful to extract feature importance (we can use it for feature selection)


>**Model weaknesses**
>
>1. Features need to have some predictive power else they won’t work.
>1. Predictions of the trees need to be uncorrelated.
>1. Appears as Black Box: It is tough to know what is happening. You can at best try different parameters and random seeds to change the outcomes and performance.
>1. They required much more computational resources.



>**Why to use this Model**
>
>1. Produces good predictions that can be understood easily. 
>1. Can handle large datasets efficiently. 
>1. Provides a higher level of accuracy in predicting outcomes over the decision tree algorithm.
<br>
><a href="https://builtin.com/data-science/random-forest-algorithm/" target="_blank">Ref 1</a>
><br>
><a href="https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/ " target="_blank">Ref 2</a>



<a id='conclusions'></a>
## Conclusions
>- KNeighborsClassifier has lowst time in training but took so much time in predicting, when using the whole testing set it took so long time. On training subset Random Forset classifier has shown an amazing accuracy plus to high f-score, but when it comes to testing set AdaBoostClassifier was the best in every way. Even through 3 sets of samples it scored highest f-score and accuracy. Based on previous comments and obsrvation. I believe that **AdaBoostClassifier**will be the most suitable model for this task.
>- **AdaBoost** is short for **Adaptive Boosting**. It is one of the most used algorithms in the machine learning community. <u> In our case to predict if the individual income more than 50K or not by reviewing many features such as: marital-status, occupation, relationship, capital-gain, capital-loss, etc.</u> 
Adaboost is an algorithm that combines classifiers with poor performance, aka weak classifiers, into a bigger classifier with much higher performance.
#### Results:
|     Metric     |  Naive predictor  | Unoptimized Model | Optimized Model |
| :------------: | :---------------: | :---------------: | :-------------: | 
| Accuracy Score |      0.2478       |      0.8576       |      0.8676     |
| F-score        |      0.2917       |      0.7246       |      0.7448     |
