library(dplyr)
library(data.table)
library(stringr)
library(HiClimR)
library(tidyr)
library(randomForest)
library(Boruta)
#install.packages("mlr")
library(mlr)
library(caret)
#install.packages("rmcfs")
library(rmcfs)

### Loading the datasets
{
path_to_data <- "../data"

read_file <- function(x,data=TRUE,train=TRUE) {
  path <- str_c(path_to_data,x,sep="/")
  if(train){ 
    path <- str_c(path,"/",x,"_train")
    
    if(data) path <- str_c(path,".DATA")
    else path <- str_c(path,".labels")
  }
  else path <- str_c(path,"/",x,"_valid.DATA")
  
  fread(path)
}

X_art=read_file("artificial")
X_dig=read_file("digits")
Y_art=read_file("artificial",data = FALSE)
Y_dig=read_file("digits",data=FALSE)
colnames(Y_art) <- c("class")
colnames(Y_dig) <- c("class")
#Y_art$y <- fifelse(Y_art$y==1,"Positive","Negative")
#Y_dig$y <- fifelse(Y_dig$y==1,"Positive","Negative")

}

### removing constatnt variables (option to remove also almost constant)
{
  #change perc if you want to remove also almost constant
X_dig <- removeConstantFeatures(data.frame(X_dig),perc=0)
X_art <- removeConstantFeatures(data.frame(X_art),perc=0)
}
##### Train-test division
{
trainIndex_art <- createDataPartition(Y_art$class,p=0.75,list = FALSE)
Art_train <- tibble(X_art[trainIndex_art,],Y_art[trainIndex_art,])
Art_test <- tibble(X_art[-trainIndex_art,],Y_art[-trainIndex_art,])

trainIndex_dig <- createDataPartition(Y_dig$class,p=0.75,list = FALSE)
Dig_train <- tibble(X_dig[trainIndex_dig,],Y_dig[trainIndex_dig,])
Dig_test <- tibble(X_dig[-trainIndex_dig,],Y_dig[-trainIndex_dig,])
}
Art_mcfs <- mcfs(formula=class~.,data=Dig_train,cutoffPermutations=4,splits=3,balance=4,featureFreq = 10)
Dig_train$class

#bez standaryzacji danych 
View(listLearners(obj="classif"))
selectFeatures()

X_art_sc_train <- scale(X_dig[trainIndex_dig,])

Boruta_model <- Boruta(class~.,data=Art_train,doTrace=1)
boruta_res <- which(Boruta_model$finalDecision=="Confirmed")

fwrite(list(boruta_res), file = str_c(path_to_data,"/boruta/plain.txt"),sep = ",",eol=" ")
