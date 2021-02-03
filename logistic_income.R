library(readr)
library(caret)
library(e1071)
library(dplyr)
library(car)
library(DMwR)
library(outliers)
library(aod)

income_csv = read_csv("C:/Users/HUGO/Desktop/income_evaluation.csv")
#View(income_csv)

####structure of the database:
str(income_csv)

#replace special characters to NaNs
income_csv[income_csv=="?"] =NA
income_csv[income_csv==99999] =NA
income_csv[income_csv==99] =NA

#count total rows with Nulls
sum(apply(income_csv, 1, anyNA))   #2625

#dispose of those rows
income_csv=income_csv[complete.cases(income_csv), ]  #32561 to 29936

###to factors the characters columns
income_fac=(income_csv %>% select_if(is.character))
income_num=(income_csv %>% select_if(is.numeric))

###work with the characters of income_fac
income_fac=sapply(income_fac, as.factor)
income_fac=as.data.frame(income_fac)
str(income_fac)

#####making the dummies
income_dummies=fastDummies::dummy_cols(income_fac)
str(income_dummies)

##delimiting the variables that we are going to use
#droping columns that are chrs
income_dummies=income_dummies[-c(1:9)]

#delimitation of low occurence categories:
signif=nrow(income_dummies)*0.05

#droping columns based on condition
income_dummies=income_dummies[,!sapply(income_dummies, function(x) (sum(x))<signif)]


#merge again dataframes and erasing 
income_csv=merge(income_dummies,income_num, by=0, all = TRUE)
income_csv$Row.names=NULL
income_fac=NULL
income_num=NULL
income_dummies=NULL

str(income_csv)

#to factor
income_csv[c(1:29)]=lapply(income_csv[c(1:29)], factor)

#imbalanced data: - this procedure is omitted as an imbalanced dataset is not necessary a problem
#income_bal= SMOTE('income_>50K' ~ ., income_csv, perc.over = 200)

str(income_csv)

#Here we have the data pre-prepared, so we dispose to preprocess it to the logit model

#avoid visible multicollinearity
income_csv$`income_<=50K`=NULL
income_csv$sex_Female=NULL

#demonstrating that we have the least amount of cases
sapply(income_csv[c(1:27)], table) 
#the least expected outcome is education masters, with (1603/29936)
#formula (n_predictors*10)/least amount:
n=(ncol(income_csv)*10)/(1603/nrow(income_csv))
n
#n < nrow, so we have the least amount of cases required


####working with numerical columns:

###############outliers
quantile(income_csv$`hours-per-week`)
boxplot.stats(income_csv$`hours-per-week`)$out

#tests of outliers:   Ho: not an outlier
test1=grubbs.test(income_csv$`hours-per-week`)#28
test1  #is an outlier

test2=grubbs.test(income_csv$`capital-loss`)#29
test2  #is an outlier

test3=grubbs.test(income_csv$`capital-gain`)#30
test3  ##is an outlier

test4=grubbs.test(income_csv$`education-num`)#31
test4  #not an outlier

test5=grubbs.test(income_csv$fnlwgt) #32
test5  #outlier

test6=grubbs.test(income_csv$age)  #33
test6  #not an outlier

##############those are NaNs
#length(income_csv[income_csv[,31] ==99999 ,31])
#length(income_csv[income_csv[,33] ==99 ,33])
###

boxplot.stats(income_csv$`capital-loss`)

#REPLACING OUTLIERS 
outreplac = function(x){
  quantiles <- quantile( x, c(.05, .95 ) )
  x[ x < quantiles[1] ] = quantiles[1]
  x[ x > quantiles[2] ] = quantiles[2]
  x
}

income_csv$`hours-per-week`=outreplac(income_csv$`hours-per-week`)
#income_csv$`capital-loss`=outreplac(income_csv$`capital-loss`) #this is omitted because its .95 and .05 quantiles are zero
income_csv$`capital-gain`=outreplac(income_csv$`capital-gain`)
income_csv$fnlwgt=outreplac(income_csv$fnlwgt)

###normalize the numeric data:
#we remove the variable fnlwgt as we don't know what it is:
income_csv$fnlwgt=NULL

sapply(income_csv[c(28:32)], summary)


####################Logit model

logitmdl=glm(income_csv$`income_>50K`~., data=income_csv, family = "binomial")
summary(logitmdl)

#multicollinearity
vif(logitmdl)
#marital-satusmarriedcivspouse, relationship-not-in-family, relationship-husband,
#and never-married, unmarried have vif values greater than 2.5

income_csv$`marital-status_Married-civ-spouse`=NULL
income_csv$`relationship_Not-in-family`=NULL
income_csv$relationship_Husband=NULL
income_csv$`marital-status_Never-married`=NULL
income_csv$relationship_Unmarried=NULL
income_csv$race_White=NULL


##2nd logit model:
logitmdl1=glm(income_csv$`income_>50K`~., data=income_csv, family = "binomial")
summary(logitmdl1)

####we dispose of the variables that were not significant
#3rd Model
logitmdl=glm(income_csv$`income_>50K`~.-education_Bachelors-`education_HS-grad`-education_Masters-`occupation_Adm-clerical`-`occupation_Machine-op-inspct`-`occupation_Transport-moving`, data=income_csv, family = "binomial")
summary(logitmdl)

#multicollinearity
vif(logitmdl)


predictprobs=logitmdl %>% predict(income_csv, type = "response")



wald.test(b = coef(logitmdl), Sigma = vcov(logitmdl), Terms = nrow(vcov(logitmdl))) #pv=0.0
#the model is useful to represent relationship in the data

#psudo R squared McFadden
psR2=function(model){
  ll.null=model$null.deviance/-2
  ll.proposed=model$deviance/-2
  psR2=(ll.null-ll.proposed)/ll.null
  return(psR2)
}

psR2(logitmdl)   #32.1%

##probabilities of coefficients:
logit2prob=function(logitmodel){
  odd_exponents=exp(logitmodel)
  #probs=odd_exponents/(1+odd_exponents)
  return(odd_exponents)
}

probs=round(logit2prob(logitmdl$coefficients), digits = 4)
probs


#confussion matrix
prds=as.factor(round(predictprobs, digits = 0))
cm=confusionMatrix(prds,income_csv$`income_>50K`)   #accuracy 82.85%
cm

#precision 71.16% 
#recall  50.46%

