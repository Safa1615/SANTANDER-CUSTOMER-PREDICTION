#Removed all the existing objects
rm(list = ls())
#Setting the working directory
setwd("D:/Edwisor_Project - Santander Prediction/")
getwd()

#Load the dataset
train_data = read.csv("train.csv",header=TRUE)
test_data = read.csv("test.csv",header=TRUE)


###################################### Exploratory Data Analysis ##################################################
# 1. Understanding the data values of every column of the dataset
str(train_data)
str(test_data)
# 2.Understanding the data distribution of the dataset
summary(train_data)
summary(test_data)
# 3. Checking the dimensions of the dataset
dim(train_data)
dim(test_data)
#####Insights from the above EDA--
#The independent variable 'ID_code' and the dependent variable 'target' happen to have NO relationship between them. Thus, we can drop the data column 'ID_code' from the dataset.

train_data=subset(train_data,select = -c(ID_code))
test_data=subset(test_data,select = -c(ID_code))
#############################Missing Value Analysis#########################################
as.data.frame(colSums(is.na(train_data)))
sum(is.na(train_data))
as.data.frame(colSums(is.na(test_data)))
sum(is.na(test_data))
##From the above result, it is clear that the dataset contains NO Missing Values.

##############################Outlier Analysis -- DETECTION###########################
numeric_col = (colnames(train_data)!='target')
train_independent = train_data[,numeric_col]
train_dependent = train_data$target
columns=colnames(test_data)

box_plot =function(begin ,end, columns,data)
{ 
  par(mar=c(3,3,3,3))
  par(mfrow=c(3,4))
  for (x in columns[begin:end])
  {
    boxplot(data[[x]] ,main=x )
  }
}

box_plot(0,40,columns,train_independent)
box_plot(40,80,columns,train_independent)
box_plot(80,120,columns,train_independent)
box_plot(120,160,columns,train_independent)
box_plot(160,200,columns,train_independent)

box_plot(0,40,columns,test_data)
box_plot(40,80,columns,test_data)
box_plot(80,120,columns,test_data)
box_plot(120,160,columns,test_data)
box_plot(160,200,columns,test_data)

#From the above boxplot visualization, it is clear that the training as well as testing dataset contains outliers.
#  Now, we will replace the outlier data values with NULL.
replace_outlier=function(data)
{
  for(x in columns)
  {
    
    value = data[,x][data[,x] %in% boxplot.stats(data[,x])$out]
    data[,x][data[,x] %in% value] = NA
  }
  return (data)
}

train_independent = replace_outlier(train_independent)
sum(is.na(train_independent))
test_data = replace_outlier(test_data)
sum(is.na(test_data))

## Imputing the outlier values with the MEAN

MEAN_impute=function(data)
{
  for(x in columns)
  {
    
    data[is.na(data[,x]), x] = mean(data[,x], na.rm = TRUE)
  }
  return (data)
}

train_independent = MEAN_impute(train_independent)
sum(is.na(train_independent))
test_data = MEAN_impute(test_data)
sum(is.na(test_data))

#########DATA VISUALIZATION -- Numeric variables of the dataset##################

dist_plot =function(begin ,end, columns,data)
{ 
  par(mar=c(2,2,2,2))
  par(mfrow=c(3,4))
  for (x in columns[begin:end])
  {
    plot(density(data[[x]]) ,main=x )
  }
}
dist_plot(0,40,columns,train_independent)
dist_plot(40,80,columns,train_independent)
dist_plot(80,120,columns,train_independent)
dist_plot(120,160,columns,train_independent)
dist_plot(160,200,columns,train_independent)

dist_plot(0,40,columns,test_data)
dist_plot(40,80,columns,test_data)
dist_plot(80,120,columns,test_data)
dist_plot(120,160,columns,test_data)
dist_plot(160,200,columns,test_data)

##From the above data visualization, the following insights can be drawn from the data-
# 1. The training as well as the testing data set is NORMALLY distributed.
# 2. The data distribution in the testing as well as training data set is similar.
# 3. The target variable contains data in an imbalanced form i.e. 90% of negative outcome (0) and 10% of positive outcome (1).
# 4. Thus, we can say that the problem statement contains imbalanced data for prediction.

####################################FEATURE SELECTION#############################################3

library(usdm)
vifcor(train_independent,th=0.9) # VIF

##we have performed VIF test to check for the presence of multi-collinearity amongst the independent variables. As a result, all the data columns were found to have a VIF = 1. 

##Thus, we can now assume that the data variables are not correlated and do not possess multi-collinearity.

#################################FEATURE SCALING##########################################################
##In our dataset, from the data visualization it is pretty clear that the training as well as testing dataset has normal distribution of data.
##Thus, we decide to apply the process of STANDARDIZATION to scale the data variables of testing as well as training dataset.

scale_data = function(data,columns)
{
  for(x in columns)
    {
      data[,x] = (data[,x]-mean(data[,x]))/sd(data[,x])
    }
}

scale_data(train_independent,columns)
scale_data(test_data,columns)

###################################SAMPLING OF DATA########################################
library(caret)
clean_data = cbind(train_independent,train_dependent)
split_index =createDataPartition(clean_data$train_dependent , p=.80 ,list=FALSE)
X = clean_data[split_index,]
Y  = clean_data[-split_index,]


#Defining error metrics to check the accuracy of the Classification ML algorithms
#error metrics -- Confusion Matrix

error_metric=function(CM)
{
  
  TN =CM[1,1]
  TP =CM[2,2]
  FP =CM[1,2]
  FN =CM[2,1]
  precision =(TP)/(TP+FP)
  recall_score =(FP)/(FP+TN)
  f1_score=2*((precision*recall_score)/(precision+recall_score))
  accuracy_model  =(TP+TN)/(TP+TN+FP+FN)
  False_positive_rate =(FP)/(FP+TN)
  False_negative_rate =(FN)/(FN+TP)
  print(paste("Precision value of the model: ",round(precision,2)))
  print(paste("Accuracy of the model: ",round(accuracy_model,2)))
  print(paste("Recall value of the model: ",round(recall_score,2)))
  print(paste("False Positive rate of the model: ",round(False_positive_rate,2)))
  print(paste("False Negative rate of the model: ",round(False_negative_rate,2)))
  print(paste("f1 score of the model: ",round(f1_score,2)))
  
  
}

# 1. Logistic regression

logit_model =glm(formula = train_dependent~. ,data =X ,family='binomial')
summary(logit_model)
logit_predict = predict(logit_model , Y[-201] ,type = 'response' )
logit_predict <- ifelse(logit_predict > 0.5,1,0) # Probability check
CM= table(Y[,201] , logit_predict)
error_metric(CM)
library(pROC)
roc_score=roc(Y[,201], logit_predict)
plot(roc_score ,main ="ROC curve for Logistic Regression ")

# 2. Naive Bayes
library(e1071)
X$train_dependent = factor(X$train_dependent ,levels = c(0,1))
# train model 
naive_model  =naiveBayes(train_dependent~.  , data =X )  
naive_predict = predict(naive_model , Y[-201])
CM_naive= table(Y[,201] , naive_predict)
error_metric(CM_naive)
naive_predict = as.numeric(naive_predict)
roc_scoren=roc(Y[,201], naive_predict) 
plot(roc_scoren ,main ="ROC curve for Naive Bayes ")


# 3. XGBOOST MODEL
library(xgboost)
X$train_dependent <- as.numeric(as.factor(X$train_dependent)) -1
Y$train_dependent <- as.numeric(as.factor(Y$train_dependent)) -1
xd =xgb.DMatrix(data =as.matrix(X[,-201]),label= X$train_dependent)
yd =xgb.DMatrix(data=as.matrix(Y[,-201]) ,label  =Y$train_dependent)

xgboost_model = xgb.train(data = xd,
                        max.depth = 2,
                        eta = 0.1,
                        nrounds = 500,
                        scale_pos_weight =11,
                        label = X$train_dependent,
                        objective = "binary:logistic")

summary(xgboost_model)
xgboost_predict = predict(xgboost_model,yd)
xgboost_predict <- ifelse(xgboost_predict > 0.5,1,0)
CM_xg= table(Y[,201] , xgboost_predict)
error_metric(CM_xg)
roc_scorexg=roc(Y[,201], xgboost_predict)
plot(roc_scorexg ,main ="ROC curve for XGB Tree ")

##We will select a model fulfilling the below conditions:
## 1. High f1 score
## 2. Low False positive rate
## 3. Low False negative rate
## 4. High AUC score
## 5. High level of Precision and Recall
## 6. High Accuracy


## Thus from the above models, we select XGBOOST model as the best fit for our problem statement.



Result = data.frame('Actual_target' = Y[,201], 'Predicted_target' = xgboost_predict )
write.csv(Result,"CUSTOMER_PREDICTION_R.csv",row.names=FALSE)