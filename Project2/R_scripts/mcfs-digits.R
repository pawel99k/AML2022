library(dplyr)
library(data.table)
library(stringr)
library(HiClimR)
library(tidyr)
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
  
  X_train <- get_file("digits")[,V1:=NULL]
  Y_train <- get_file("digits",get_X = FALSE)
#  X_test <- get_file("digits",train = FALSE)[,V1:=NULL]
#  Y_test <- get_file("digits",train = FALSE,get_X = FALSE)
  colnames(Y_train) <- c("class")
#  colnames(Y_test) <- c("class")
  #Y_test$class <- as.factor( Y_test$class)
  #Y_train$class <- as.factor( Y_train$class)
}

{
  X_train_sc <- scale(X_train)
#  X_test_sc <- scale(X_test, center=attr(X_train_sc, "scaled:center"), 
#                     scale=attr(X_train_sc, "scaled:scale"))
}
###all into 1 dataframe (after standarization)
{
  Dig_train <- data.frame(X_train_sc,Y_train,check.names=FALSE,fix.empty.names=FALSE)
#  Dig_test <- data.frame(X_test_sc,Y_test,check.names=FALSE,fix.empty.names=FALSE)
}
Dig_mcfs <- mcfs(formula=class~.,data=Dig_train,mode = 2,featureFreq=75,threadsNumber=3)
{
  fwrite(list(head(colnames(Dig_mcfs$data), -1)), file="../data/mcfs/digits-features.csv",eol=",",append=TRUE)
  fwrite(list(" "), file ="../data/mcfs/digits-features.csv",eol="\n",append=TRUE)
}
Dig_mcfs$exec_time
