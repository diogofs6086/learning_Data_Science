#### PART FOUR ####

# In this script, the trained model was applied to the provided test dataset, 
# test.csv. Afterward, the predicted results were matched with the click_id 
# to produce the submission file.

# The test dataset is similar to the training dataset, with the following 
# differences:
#  click_id: reference for making predictions
# is_attributed: not included

# Removes all existing objects and packages from the current workspace
rm(list = ls())
# Working directory 
# setwd("~/Documents/DSA/Big_Data_Analytics_com_R_e_Microsoft_Azure_Machine_Learning/Cap20_Projetos_com_feedback/project1")
# getwd()

# Packages
library(dplyr)
library(data.table)
library(randomForest)

# Loading the model
model15s <- readRDS("model15.rds")

# Loading the test file
# The test dataset named test.csv can be found on the web site
# https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data
test_set <- fread(file = 'test.csv', header = T, 
                  select = c('click_id', 'ip', 'app', 'channel'))

# ip feature
# Repeated ips in order
n_dupl_ips <- test_set %>%
  count(ip, wt = n(), name = 'repetitions') %>%
  arrange(desc(repetitions))
  
# Number of duplicate ips column
test_set <- left_join(test_set, n_dupl_ips, by = 'ip')
test_set$ip <- NULL
rm(n_dupl_ips)

# repetitions classes
test_set$repetitions_fac <- cut(test_set$repetitions,
                                breaks = c(0,5,nrow(test_set)), 
                                labels = c(1, 2))
test_set$repetitions <- NULL

# app classes
test_set$app_fac <- cut(test_set$app,
                        breaks = c(0, 3, 12, 18, nrow(test_set)),
                        right = F, labels = c(1, 2, 3, 4))
gc()

# Predictions using the model 15s
predictions15 <- predict(model15s, test_set, type = "prob")
head(predictions15)

# The submission file with the calculated probabilities 
# for the is_attributed variable
test_set_results <- data.frame(click_id = test_set$click_id, 
                               is_attributed = predictions15[,1])
head(test_set_results)
dim(test_set_results)

# Saving the submission file
write.csv(x = test_set_results, file = 'submission_file.csv', row.names = F)

# Number yes (1) or no (0) is_attributed variable
table(round(test_set_results[,2]))

# Cleaning the house
rm(list = ls())
gc()

################################################################################

# THE END