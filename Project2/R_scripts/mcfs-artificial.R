library(dplyr)
library(data.table)
library(stringr)
library(HiClimR)
library(tidyr)
library(randomForest)
library(caret)
library(rmcfs)

### Loading the datasets
{
get_file <- function(dataset_name,train=TRUE,get_X=TRUE) {
  path <- str_c("../data",dataset_name,sep="/")
  if(train){
    if(get_X){
      path <- str_c(path,"/X_train.csv") 
    }
    else{
      path <- str_c(path,"/y_train.csv")
    }
  }
  else{
    if(get_X){
      path <- str_c(path,"/X_test.csv") 
    }
    else{
      path <- str_c(path,"/y_test.csv")
    }
  }
  if(get_X){
  return <- fread(path,header=TRUE,data.table=TRUE)
  }
  else{
  return <-fread(path,data.table=TRUE)
  }
}

X_train <- get_file("artificial")[,V1:=NULL]
Y_train <- get_file("artificial",get_X = FALSE)
#X_test <- get_file("artificial",train = FALSE)[,V1:=NULL]
#Y_test <- get_file("artificial",train = FALSE,get_X = FALSE)
colnames(Y_train) <- c("class")
#colnames(Y_test) <- c("class")
#Y_test$class <- as.factor( Y_test$class)
#Y_train$class <- as.factor( Y_train$class)
}
#### Standarization
{
  X_train_sc <- scale(X_train)
#  X_test_sc <- scale(X_test, center=attr(X_train_sc, "scaled:center"), 
#                     scale=attr(X_train_sc, "scaled:scale"))
}
###all into 1 dataframe (after standarization)
{
  Art_train <- data.frame(X_train_sc,Y_train,check.names=FALSE,fix.empty.names=FALSE)
#  Art_test <- data.frame(X_test_sc,Y_test,check.names=FALSE,fix.empty.names=FALSE)
}
Art_mcfs <- mcfs(formula=class~.,data=Art_train,mode = 2,featureFreq=75,threadsNumber=3)
{
fwrite(list(colnames(Art_mcfs$data)), file =  "../data/mcfs/artificial-features.csv",eol=",",append=TRUE)
fwrite(list(" "), file =  "../data/mcfs/artificial-features.csv",eol="\n",append=TRUE)
}
Art_mcfs$exec_time