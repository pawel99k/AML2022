library(dplyr)
library(data.table)
library(stringr)
library(HiClimR)
library(tidyr)
library(randomForest)
library(mlr)
library(caret)
library(rmcfs)

### Loading the datasets
{
  path_to_data <- "../data"
  
  read_file <- function(dataset_name,data=TRUE,train=TRUE) {
    path <- str_c(path_to_data,dataset_name,sep="/")
    if(train){ 
      path <- str_c(path,"/",dataset_name,"_train")
      
      if(data) path <- str_c(path,".data")
      else path <- str_c(path,".labels")
    }
    else path <- str_c(path,"/",dataset_name,"_valid.data")
    
    fread(path)
  }
  
  X_dig=read_file("digits")
  Y_dig=read_file("digits",data=FALSE)
  colnames(Y_dig) <- c("class")
}

### removing constant variables (option to remove also almost constant)
{
  X_dig <- removeConstantFeatures(data.frame(X_dig),perc=0)
}
##### Train-test division
{
  trainIndex_dig <- createDataPartition(Y_dig$class,p=0.75,list = FALSE)
  Dig_train <- tibble(X_dig[trainIndex_dig,],Y_dig[trainIndex_dig,])
  Dig_test <- tibble(X_dig[-trainIndex_dig,],Y_dig[-trainIndex_dig,])
}
Dig_mcfs <- mcfs(formula=class~.,data=as.data.frame(Dig_train),mode = 2,featureFreq=75,threadsNumber=3)
{
  fwrite(list(colnames(Dig_mcfs$data)), file =  str_c(path_to_data,"/mcfs/digits-features.csv"),eol=",",append=TRUE)
  fwrite(list(" "), file =  str_c(path_to_data,"/mcfs/digits-features.csv"),eol="\n",append=TRUE)
}
