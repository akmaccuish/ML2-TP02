rm(list=ls())
setwd('~/3 Machine Learning/Final Project')
library(ggplot2) 
library(readr)
library(jsonlite)
library(dplyr)
library(ModelMetrics)

set.seed(5082)


json_file <- "train.json"     #Set file name

#opens and reads lines in from JSON file
json_str <- paste(readLines(json_file), collapse = "")  
json_dat <- fromJSON(json_str)

dat = as.data.frame(t(do.call(rbind, json_dat))) #JSON Comes in as list of lists converts to data frame
subset.rows <- sample(nrow(dat), 2000) #Samples subset of data since initially over 40,000 records
dat=dat[subset.rows,]
train.rows <- sample(nrow(dat), nrow(dat) * .7) #split to train and test
train <- dat[train.rows, ]
test <- dat[-train.rows, ]

#Unlisting data to flatten from JSON format
train$bathrooms <- unlist(train$bathrooms)
train$bedrooms <- unlist(train$bedrooms)
train$building_id <- unlist(train$building_id)
train$created <- unlist(train$created)
train$description <- unlist(train$description)
train$display_address <- unlist(train$display_address)
train$latitude <- unlist(train$latitude)
train$listing_id <- unlist(train$listing_id)
train$longitude <- unlist(train$longitude)
train$manager_id <- unlist(train$manager_id)
train$price <- unlist(train$price)
train$street_address <- unlist(train$street_address)
train$interest_level <- unlist(train$interest_level)

train$features <- as.character(train$features)
train$photos <- as.character(train$photos)
train$interest_level <- as.factor(train$interest_level)
glimpse(train)

train=train[,-c(3,4,5,6,7,9,11,12,14)] #remove undesired features
glimpse(train)

test$bathrooms <- unlist(test$bathrooms)
test$bedrooms <- unlist(test$bedrooms)
test$building_id <- unlist(test$building_id)
test$created <- unlist(test$created)
test$description <- unlist(test$description)
test$display_address <- unlist(test$display_address)
test$latitude <- unlist(test$latitude)
test$listing_id <- unlist(test$listing_id)
test$longitude <- unlist(test$longitude)
test$manager_id <- unlist(test$manager_id)
test$price <- unlist(test$price)
test$street_address <- unlist(test$street_address)
test$interest_level <- unlist(test$interest_level)
test$features <- as.character(test$features)
test$photos <- as.character(test$photos)
test$interest_level <- as.factor(test$interest_level)
test=test[,-c(3,4,5,6,7,9,11,12,14)]

#Make sure to omit any blanks
train=na.omit(train)
test=na.omit(test)

trainy <- train$interest_level
trainx <- train[1:5]

testy <- test$interest_level
testx <- test[1:5]
####SVM###

library(e1071)
attach(train)
set.seed(5082)
tune.out <- tune(svm,
                 interest_level~.,
                 data=train,
                 kernel ="radial",
                 probability = TRUE,
                 ranges = list(cost=c(.05,.5, 1, 2),
                               gamma=c(.01, .1, .25, .5)))
summary(tune.out) 
svm.yhat<-predict(tune.out$best.model, newdata=testx, probability = TRUE)
#attr(svm.yhat, "probabilities")
mlogLoss(testy,attr(svm.yhat, "probabilities"))
#LogLoss of 1.706324

svm.confusion<-table(svm.yhat, test$interest_level)
svm.confusion 
(svm.confusion[1,1]+svm.confusion[2,2]+svm.confusion[3,3])/(sum(svm.confusion))
#Accuracy Rate: 68.5


###RandomForest###
library(randomForest)
#install.packages("ModelMetrics")
library(ModelMetrics)
set.seed(5082)
rf.mod <- randomForest(interest_level~., 
                       data=train, 
                       mtry=2, 
                       importance = TRUE)
yhat.rf<-predict(rf.mod, newdata=testx, type='prob')
mlogLoss(testy, yhat.rf)
#LogLoss of 0.7849822

yhat.rf<-predict(rf.mod, newdata=testx)
rf.confusion<-table(yhat.rf, test$interest_level)
rf.confusion 
(rf.confusion[1,1]+rf.confusion[2,2]+rf.confusion[3,3])/(sum(rf.confusion))
#Accuracy Rate: 67.3%
mlogLoss(test$interest_level, yhat.rf)

###KNN###
set.seed(5082)

knnmod <- train(trainx, trainy, method='knn')
summary(knnmod)

predk <- predict(knnmod, newdata=testx, type="prob")
predk
mlogLoss(testy, predk)
conmat <- table(predk, testy)
conmat
accuracyknn <- (conmat[1,1]+conmat[2,2]+conmat[3,3])/sum(conmat)
accuracyknn