setwd("~/Documents/learning_Data_Science/R_scripts")
getwd()

# Problem: Predict student final grades (G3)
# Data set site: https://archive.ics.uci.edu/ml/datasets/Student+Performance
# Method: k-Nearest Neighbour Classification

# Packages
library(class)
library(dplyr)
library(gmodels)
library(caret)
?knn

# Read data set
df <- read.csv2('students.csv', stringsAsFactors = F)


### The exploratory data analysis was done in PART 1 with the Linear Model

# Verifying if NA data exist
sum(is.na(df))


### Division into training and test data

# Relative proportion of schools
round(prop.table(table(df$school)), 2)

# Sort by school
df <- df %>%
  arrange(school)
View(df)

# ID column insertion
df$ID <- c(1:dim(df)[1])

# School IDs
GP_school <- df$ID[df$school == 'GP']
MS_school <- df$ID[df$school == 'MS']

# 70% of training observations
# 30% of test observations

# GP school
GP_school_test <- sample(GP_school, round(length(GP_school)*0.30))
GP_school_test <- df[df$ID %in% GP_school_test, ]
GP_school_test

GP_school_train <- GP_school[!(GP_school) %in% GP_school_test$ID]
GP_school_train <- df[df$ID %in% GP_school_train, ]
GP_school_train

# Checking if the number of rows of training and test data are equal to the total
(nrow(GP_school_test) + nrow(GP_school_train)) == length(GP_school)

# MS school
MS_school_test <- sample(MS_school, round(length(MS_school)*0.30))
MS_school_test <- df[df$ID %in% MS_school_test, ]
MS_school_test

MS_school_train <- MS_school[!(MS_school) %in% MS_school_test$ID]
MS_school_train <- df[df$ID %in% MS_school_train, ]
MS_school_train

# Checking if the number of rows of training and test data are equal to the total
(nrow(MS_school_test) + nrow(MS_school_train)) == length(MS_school)

# Concatenating the training data sets of the schools
data_train <- rbind(GP_school_train, MS_school_train)
data_train <- data_train[ , colnames(df) != 'ID']
dim(data_train)

# Concatenating the test data sets of the schools
data_test <- rbind(GP_school_test, MS_school_test)
data_test <- data_test[ , colnames(df) != 'ID']
dim(data_test)

### Making classes of grades, as in the reference of the data set site entitled 
### "Cortes, P., Silva, A. USING DATA MINING TO PREDICT SECONDARY SCHOOL STUDENT
###  PERFORMANC. University of Minho. Portugal"
grade_classes <- function(x) {
  if (x > 15) {x = "A"}
  else if (x < 10) {x = "F"}
  else if (x == 14 | x == 15) {x = "B"}
  else if (x == 12 | x == 13) {x = "C"}
  else {x = "D"}
}

data_train$G3 <- sapply(data_train$G3, grade_classes)
data_train$G3 <- as.factor(data_train$G3)
str(data_train$G3)
View(data_train)

data_test$G3 <- sapply(data_test$G3, grade_classes)
data_test$G3 <- as.factor(data_test$G3)
str(data_test$G3)
View(data_test)

data_train_labels <- as.factor(data_train$G3)
data_train_labels
data_test_labels <- as.factor(data_test$G3)
data_test_labels

# The KNN algorithm does not except string or factor type
# Selecting the numeric columns
str(data_train)
data_train_num <- select_if(data_train, is.numeric)
head(data_train_num)

data_test_num <- select_if(data_test, is.numeric)
head(data_test_num)

################################################################################
################################################################################
################################################################################

# Model 1 - KNN model, where the target variable depends on all the numeric 
#           variables
model_1 <- knn(train = data_train_num,
                     test = data_test_num,
                     cl = data_train_labels,
                     k = 3)
model_1

# Summary
summary(model_1)
# The summary shows the predictions

# Confusion Matrix
confusionMatrix(data_test$G3, model_1)

# Other Confusion Matrix
CrossTable(x = data_test_labels, y = model_1, prop.chisq = FALSE)


################################################################################
################################################################################
################################################################################
# Model 2 - KNN model dependent on the variables of the best simulation of PART1

model_2 <- knn(train = data_train_num[c('absences', 'failures', 'famrel', 
                                        'Medu', 'studytime', 'G2')],
               test = data_test_num[c('absences', 'failures', 'famrel', 
                                      'Medu', 'studytime', 'G2')],
               cl = data_train_labels,
               k = 10)
model_2

# Summary
summary(model_2)

# Confusion Matrix
confusionMatrix(data_test$G3, model_2)


################################################################################
################################################################################
################################################################################

## Salving the train and test data set to work on the next step
write.csv2(x = data_train, file = 'students_data_train.csv')
write.csv2(x = data_test, file = 'students_data_test.csv')
