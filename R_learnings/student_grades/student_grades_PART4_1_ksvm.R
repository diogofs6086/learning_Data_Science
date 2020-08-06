# Removes all existing objects from the current workspace
rm(list = ls())
# Working directory 
setwd("~/Documents/learning_Data_Science/R_scripts")
getwd()

# Problem: Predict student final grades (G3)
# Data set site: https://archive.ics.uci.edu/ml/datasets/Student+Performance
# Method: Kernel Support Vector Machines (kvsm)

# Packages
library(kernlab)

# Read data set
train_ds <- read.csv2('students_data_train.csv', stringsAsFactors = T)
test_ds <- read.csv2('students_data_test.csv', stringsAsFactors = T)
str(train_ds)
str(test_ds)

### The exploratory data analysis and the divison on train and test data set 
### were done in PART 1 with the Linear Model


################################################################################
################################################################################
################################################################################

# Model 1 - KSVM model, where the target variable depends on all variables

# Model
model_1 <- ksvm(G3 ~ ., data = train_ds, kernel = 'vanilladot')
model_1

# Predictions
predictions_1 <- predict(model_1, test_ds)

# Confusion Matrix
table(predictions_1, test_ds$G3)

# Correct prediction percentage
mean(predictions_1 == test_ds$G3)

################################################################################
################################################################################
################################################################################

# Model 1 - KSVM model, where the target variable depends on G2 only

# Model
model_2 <- ksvm(G3 ~ G2,
                data = train_ds, kernel = 'rbf')
model_2

# Predictions
predictions_2 <- predict(model_2, test_ds)

# Confusion Matrix
table(predictions_2, test_ds$G3)

# Correct prediction percentage
mean(predictions_2 == test_ds$G3)

################################################################################
################################################################################
################################################################################