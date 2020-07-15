# Removes all existing objects from the current workspace
rm(list = ls())
# Working directory 
setwd("~/Documents/learning_Data_Science/R_scripts")
getwd()

# Problem: Predict student final grades (G3)
# Data set site: https://archive.ics.uci.edu/ml/datasets/Student+Performance
# Method: Recursive Partitioning and Regression Trees

# Packages
library(rpart)
?rpart

# Read data set
data_train <- read.csv2('students_data_train.csv', stringsAsFactors = T)
data_train <- data_train[-1]
data_test <- read.csv2('students_data_test.csv', stringsAsFactors = T)
data_test <- data_test[-1]
str(data_train)
str(data_test)

### The exploratory data analysis and the divison on train and test data set 
### were done in PART 1 with the Linear Model


################################################################################
################################################################################
################################################################################

# Model 1 - RF model, where the target variable depends on all variables
model_1 <- rpart(G3 ~ .,
                 data = data_train,
                 control = rpart.control(cp = 0.01))

# Test data set predictions
tree_pred_1 <- predict(model_1, data_test, type = 'class')

# Correct prediction percentage
mean(tree_pred_1 == data_test$G3)

# Confusion Matrix
table(tree_pred_1, data_test$G3)


################################################################################
################################################################################
################################################################################
# Model 2 - RF model depends on the variables of the best simulation of PART1

data_train <- data_train %>%
  mutate(absences2 = ifelse (absences > 0, 1, 0))
data_test <- data_test %>%
  mutate(absences2 = ifelse (absences > 0, 1, 0))

data_train <- data_train %>%
  mutate(failures2 = ifelse (failures > 0, 1, 0))
data_test <- data_test %>%
  mutate(failures2 = ifelse (failures > 0, 1, 0))

data_train$famrel2 <- data_train$famrel^0.01
data_test$famrel2 <- data_test$famrel^0.01

data_train$G22 <- data_train$G2^1.2
data_test$G22 <- data_test$G2^1.2

model_2 <- rpart(G3 ~ absences2 + failures2 + famrel2 + Medu + studytime + G22,
                 data = data_train,
                 control = rpart.control(cp = 0.01))
model_2

# Test data set predictions
tree_pred_2 <- predict(model_2, data_test, type = 'class')

# Correct prediction percentage
mean(tree_pred_2 == data_test$G3)

# Confusion Matrix
table(tree_pred_2, data_test$G3)

################################################################################
################################################################################
################################################################################
# Model 3 - RF model depends only on G2 variable

model_3 <- rpart(G3 ~ G2,
                 data = data_train,
                 control = rpart.control(cp = 0.01))
model_3

# Test data set predictions
tree_pred_3 <- predict(model_3, data_test, type = 'class')

# Correct prediction percentage
mean(tree_pred_3 == data_test$G3)

# Confusion Matrix
table(tree_pred_3, data_test$G3)

################################################################################
################################################################################
################################################################################

### Conclusion: The Linear Model gives the results to predict G3.