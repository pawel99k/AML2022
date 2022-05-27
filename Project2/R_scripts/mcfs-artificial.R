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

X_art=read_file("artificial")
Y_art=read_file("artificial",data = FALSE)
colnames(Y_art) <- c("class")
}
### removing constant variables (option to remove also almost constant)
{
  #change perc if you want to remove also almost constant
X_art <- removeConstantFeatures(data.frame(X_art),perc=0)
}
##### Train-test division
{
trainIndex_art <- createDataPartition(Y_art$class,p=0.75,list = FALSE)
  X_train <- X_art[trainIndex_art,]
  X_test <- X_art[-trainIndex_art,]
  Y_train <- Y_art[trainIndex_art]
  Y_test <- Y_art[-trainIndex_art]
}
#### Standarization
{
  X_train_sc <- scale(X_train)
  X_test_sc <- scale(X_test, center=attr(X_train_sc, "scaled:center"), 
                     scale=attr(X_train_sc, "scaled:scale"))
}
###all into 1 dataframe (after standarization)
{
  Art_train <- data.frame(X_train_sc,Y_train)
  Art_test <- data.frame(X_test_sc,Y_test)
}
Art_mcfs <- mcfs(formula=class~.,data=Art_train,mode = 2,featureFreq=50,threadsNumber=3)
{
fwrite(list(colnames(Art_mcfs$data)), file =  str_c(path_to_data,"/mcfs/artificial-features.csv"),eol=",",append=TRUE)
fwrite(list(" "), file =  str_c(path_to_data,"/mcfs/artificial-features.csv"),eol="\n",append=TRUE)
}
