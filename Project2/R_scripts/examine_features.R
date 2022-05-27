library(data.table)
library(randomForest)
library(xgboost)

getPerformances <- function(Train,Test,ntree=500){
  logistic_model <- glm(y~.,data=Train,family = binomial)
  logistic_prob <- predict(object = logistic_model,newdata=Test)
  logistic_pred <- fifelse(logistic_prob>0.5,1,-1)
  
  random_forest_model <- randomForest(y~.,data=Train,ntree = ntree)
  random_forest_prob <- predict(object=random_forest_model,data=Test)
  random_forest_pred <- fifelse(random_forest_prob>0.5,1,-1)
  
  xgboost_model <- xgboost()
  
}
