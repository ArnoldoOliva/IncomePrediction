# IncomePrediction
Binary classification of level of income (&lt;50USD K / >50 USD K per year) with algorithms.

Using data from Kaggle, (https://www.kaggle.com/lodetomasi1995/income-classification), about levels of income across the world, and variables about age, workclass, education,
race, sex (better described in Kaggle), among others, ML algorithms are designed in order to classify either a person earns more than 50K USD per year or not. 

In this repository there are two scripts added in order to task this problem in different ways, the first one being a XGBoosting Classifier method  (in Python) 
with levels of accuracy around 85%. The process was conformed of data cleaning, data labeling  (most of the variables were characters), and oversampling.

The second one being a script of a logistic regression in RStudio, with the following steps: first, cleaning the database from Nulls, then, creating the factor/dummy variables 
(and also dropping dummy variables with low occurences); and after that, as a regression can be biased due to outliers, the outliers were treated (replacing outliers with quantiles), and finally, the logistic regression was made. Different measures were applied to evaluate the models, (such as VIF, sudo R-squared of McFadden and confusion matrix), to get a good final model. The final model was constituted with 19 variables, all significant, and with a sudo R-squared of 32%. The overall accuracy was 82.85%

Some of the variables in the model were the type of workclass (local government, private, self employed),occupation, marital status, sex, hours worked per week among others. Some of the insights from the model were that, in 1995, a divorced person had 72% more probability to earn less than 50K USD per year, being a US citizen gives you 19.8% more probabilities of earning more, males earn 2.4 times more than women, and also working 1 more hour per week gives you 4% more probability to earn more than 50k USD. It is worth noting that some of the variables that may have economic sense were not in the model due to the lack of significance, such as educational ones like "bachelor", "master"; and it could be because of the variable "education_some_college" may represent a better cutoff in the relationship of education-income than those variables.



