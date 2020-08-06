#### PART TWO ####

# This script got the main tidying lines of part one to tidy the full 
# training dataset, nominated train.csv.

# Removes all existing objects and packages from the current workspace
rm(list = ls())
# Working directory 
# setwd("~/Documents/DSA/Big_Data_Analytics_com_R_e_Microsoft_Azure_Machine_Learning/Cap20_Projetos_com_feedback/project1")
# getwd()

# Packages
library(dplyr)
library(data.table)
library(caret)
library(randomForest)
library(DMwR)


# number of rows in the train dataset
# The train dataset named train.csv can be found on the web site
# https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data
n_rows <- fread(file = 'train.csv', header = T, select = 'is_attributed')
n_rows <- nrow(n_rows)
n_rows    # 184.903.890 rows
gc()

# Calculating the number of batches
for (i in c(15:100)) {
  if (n_rows%%i == 0) {
    print(c(i, n_rows/i))
  }
}             # 15 seems better for my computer capacity
rm(i)

# Batches
n = 15
train_set <- data.frame(is_attributed = c(),
                        app = c(),
                        channel = c(),
                        repetitions_fac = c(),
                        app_fac = c())
for (i in c(0:(n-1))) {
  if (i == 0) {
    # The train dataset named train.csv can be found on the web site
    # https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data
    train <- fread(file = 'train.csv', header = T, 
                 skip = n_rows/n*i, nrows = n_rows/n,
                  select = c('is_attributed', 'ip', 'app', 'channel'))
                 } else {
    train <- fread(file = 'train.csv', header = F, 
                   skip = n_rows/n*i, nrows = n_rows/n,
                   select = c(8,1,2,5))
    names(train) <- c('is_attributed', 'ip', 'app', 'channel')
  }
 
  # ip feature
  # Repeated ips in order
  n_dupl_ips <- train %>%
    count(ip, wt = n(), name = 'repetitions') %>%
    arrange(desc(repetitions))
  
  # Number of duplicate ips column
  train <- left_join(train, n_dupl_ips, by = 'ip')
  train$ip <- NULL
  
  # repetitions classes
  train$repetitions_fac <- cut(train$repetitions,
                               breaks = c(0,5,nrow(train)), 
                               labels = c(1, 2))
  train$repetitions <- NULL
  
  # app classes
  train$app_fac <- cut(train$app,
                       breaks = c(0, 3, 12, 18, nrow(train)),
                       right = F, labels = c(1, 2, 3, 4))
  
  # is_attributed classes
  train <- train %>%
    mutate(is_attributed = factor(is_attributed, levels = c(1,0)))
  head(train_set)
  
  # Balancing the target class
  train <- SMOTE(is_attributed ~ ., data  = train)
  
  # Binding the train dataset
  train_set <- rbind(train_set, train)
  
  rm(n_dupl_ips, train)
  gc()
  print(i)
}

dim(train_set)
table(train_set$is_attributed) 

# Saving the tidy train dataset 
write.csv(x = train_set, file = 'train_set.csv')

################################################################################

# Continue on part three, 
#             filename project_click_fraud_3_predictions_with_the_test_dataset.R