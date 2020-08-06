# Removes all existing objects from the current workspace
rm(list = ls())
# Working directory 
setwd("~/Documents/learning_Data_Science/R_scripts")
getwd()

# Problem: Predict student final grades (G3)
# Data set site: https://archive.ics.uci.edu/ml/datasets/Student+Performance
# Method: Support Vector Machines

# Packages
library(e1071)
?svm

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

# Model 1 - SVM model, where the target variable depends on all variables
model_1 <- svm(G3 ~ .,
                data = data_train,
                type = 'C-classification',
                kernel = 'radial')
model_1

# Train data set predictions
pred_train <- predict(model_1, data_train)

# Correct prediction percentage
mean(pred_train == data_train$G3)

# Confusion Matrix
table(pred_train, data_train$G3)

# Test data set predictions
pred_test <- predict(model_1, data_test)

# Correct prediction percentage
mean(pred_test == data_test$G3)

# Confusion Matrix
table(pred_test, data_test$G3)


################################################################################
################################################################################
################################################################################
# Model 2 - SVM model dependent on the variables of the best simulation of PART1

model_2 <- svm(G3 ~ absences + failures + famrel + Medu + studytime + G2,
               data = data_train,
               type = 'C-classification',
               kernel = 'radial')
model_2

# Train data set predictions
pred_train <- predict(model_2, dta_train)

# Correct prediction percentage
mean(pred_train == data_train$G3)

# Confusion Matrix
table(pred_train, data_train$G3)

# Test data set predictions
pred_test <- predict(model_2, data_test)

# Correct prediction percentage
mean(pred_test == data_test$G3)

# Confusion Matrix
table(pred_test, data_test$G3)

################################################################################
################################################################################
################################################################################