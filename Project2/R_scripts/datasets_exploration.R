library(dplyr)
library(data.table)
library(stringr)
#install.packages("HiClimR")
### package to calculate Correlation matrix fast
library(HiClimR)

### Loading the datasets
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

DATA_list <- list("artificial"=read_file("artificial"),"digits"=read_file("digits"))

#labels_list <- list("artificial"=read_file("artificial",FALSE),"digits"=read_file("digits",FALSE))

### checking if they have any missing values

sapply(DATA_list,FUN=function(x) sum(is.na(x)))

### no missing values in any of the datasets

### checking basic statistics of dataset

quantiles_to_check <- c(0,0.05,0.25,0.50,0.75,0.90,0.95,0.99,1)

basic_stat <- lapply(DATA_list,
                     FUN=function(x) do.call(cbind, lapply(x, quantile,probs = quantiles_to_check)))

# View(basic_stat$artificial)
# View(basic_stat$digits)

sum(basic_stat$artificial['0%',]==basic_stat$artificial["100%",])

### there are 0 completely useless columns in the dataset artificial

sum(basic_stat$digits['0%',]==basic_stat$digits["100%",])

### there are 45 completely useless columns in the dataset digits

sum(basic_stat$digits['0%',]==basic_stat$digits["75%",])
sum(basic_stat$digits['0%',]==basic_stat$digits["90%",])
sum(basic_stat$digits['0%',]==basic_stat$digits["95%",])
sum(basic_stat$digits['0%',]==basic_stat$digits["99%",])

### there are many features whose values are the same for significant number of observations


### Let us inpect correlation matrices, first we neeed to create them
### we will only use the values on the one side of the diagonal

## (uwaga liczy się dłużej niż cokolwiek wcześniej)
corr_matrix <- lapply(DATA_list,fastCor,upperTri=TRUE,optBLAS=TRUE)
corr_matrix <- lapply(corr_matrix,FUN=function(x) as.data.table(x))
#View(cor_matrix$artificial)
#View(cor_matrix$digits)

### let us check how many variables is "removeable" by in case we with to 
### remove some highly correlated ones (above value thresh)
thresh <- 0.99

sum(abs(corr_matrix$artificial)>thresh,na.rm = TRUE)

sum(abs(corr_matrix$digits)>thresh,na.rm = TRUE)
