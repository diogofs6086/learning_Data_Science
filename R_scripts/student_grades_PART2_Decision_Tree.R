setwd("~/Documents/learning_Data_Science/R_scripts")
getwd()

# Problem: Predict student final grades (G3)
# Data set site: https://archive.ics.uci.edu/ml/datasets/Student+Performance
# Method: Decision Tree

# Packages
library(C50)
library(caret)
library(pROC)
library(ggplot2)
library(dplyr)
library(ROSE)

# Read data set
df <- read.csv2('students.csv', stringsAsFactors = T)


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

################################################################################
################################################################################
################################################################################

# Model 1 - Decision Tree with G3 depending on all variables 
#           in the training data set with any modification
model_1 <- C5.0(G3 ~ ., data = data_train)
model_1

# Summary
summary(model_1)

# Predictions with the test data set
predictions_1 <- predict(model_1, data_test)

# Confision Matrix
?confusionMatrix
confusionMatrix(data_test$G3, predictions_1)

# Percentage of Correct Classifications (PCC) of the site reference article
PCC <- round(sum(predictions_1 == data_test$G3) / length(data_test$G3), 4) * 100
PCC
# The PCC is equal to the Accuracy given in the output of the Confusion Matrix

# Grade countings of the test and prediction data
grades_count <- data_test %>%
  count(Grade = G3) 
grades_count

pred_count <- as.data.frame(predictions_1) %>%
  count(Grade = predictions_1)
pred_count

grades <- rbind(grades_count, pred_count)
grades$Source <- c(rep('Observation', 5), rep('Prediction', 5))
grades

# Bar plot
ggplot(data = grades, aes(x = Grade, y = n, fill = Source)) +
  geom_bar(stat = "identity", position = "dodge", colour = "black") +
  ylab('Number of students')

################################################################################
################################################################################
################################################################################

# Model 2 - Decision Tree with G3 depending on the G2 variable

model_2 <- C5.0(G3 ~ G2, data = data_train, trials = 1)
model_2

# Summary
summary(model_2)

# Predictions with the test data set
predictions_2 <- predict(model_2, data_test)

# Confision Matrix
confusionMatrix(data_test$G3, predictions_2)

################################################################################
################################################################################
################################################################################

## Balancing classes

# Grade classes proportion
data_train %>%
  count(Grade = G3) %>%
  mutate(rel = round(n/sum(n),2))

data_test %>%
  count(Grade = G3) %>%
  mutate(rel = round(n/sum(n),2))


# "F" is the biggest class, so the balancing will be done by it
data_train_blc <- data.frame()
data_test_blc <- data.frame()
c = 0
for (i in c("A","B","C","D")) {
  train_blc <- ROSE(G3 ~ ., data = data_train[data_train$G3 == 'F' | data_train$G3 == i, ])$data
  test_blc <- ROSE(G3 ~ ., data = data_test[data_test$G3 == 'F' | data_test$G3 == i, ])$data
  
  ifelse (c == 0,
    data_train_blc <- rbind(data_train_blc, train_blc),
    data_train_blc <- rbind(data_train_blc, train_blc[train_blc$G3 == i, ]))
  
  ifelse (c == 0,
          data_test_blc <- rbind(data_test_blc, test_blc),
          data_test_blc <- rbind(data_test_blc, test_blc[test_blc$G3 == i, ]))
  
  c = c +1
}

# Comparison with the previous and balancing classes
round(prop.table(table(data_train$G3)), 2)
round(prop.table(table(data_train_blc$G3)), 2)
dim(data_train_blc)
dim(data_train)

round(prop.table(table(data_test$G3)), 2)
round(prop.table(table(data_test_blc$G3)), 2)
dim(data_test_blc)
dim(data_test)

# Sampling the data sets because they are ordered
View(data_train_blc)
View(data_test_blc)

sp_tr <- sample(c(1:nrow(data_train_blc)), size = nrow(data_train_blc), replace = F)
data_train_blc <- data_train_blc[sp_tr, ]
head(data_train_blc)

sp_te <- sample(c(1:nrow(data_test_blc)), size = nrow(data_test_blc), replace = F)
data_test_blc <- data_test_blc[sp_te, ]
head(data_test_blc)


################################################################################
################################################################################
################################################################################

# Model 3 - Decision Tree with the balancing classes

model_3 <- C5.0(G3 ~ ., data = data_train_blc, trials = 1)
model_3

# Summary
summary(model_3)

# Predictions with the test data set
predictions_3 <- predict(model_3, data_test_blc)
predictions_31 <- predict(model_3, data_test)

# Confision Matrix
confusionMatrix(data_test_blc$G3, predictions_3)
confusionMatrix(data_test$G3, predictions_31)

# Conclusion: The results with DT were worse than the LM.

################################################################################
################################################################################
################################################################################
