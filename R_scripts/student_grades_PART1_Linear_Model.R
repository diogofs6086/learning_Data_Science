setwd("~/Documents/learning_Data_Science/R_scripts")
getwd()

# Problem: Predict student final grades (G3)
# Data set site: https://archive.ics.uci.edu/ml/datasets/Student+Performance
# Method: Linear Model

# Packages that
library(ggplot2)
library(dplyr)
# library(gmodels)
library(psych)

# Read data set
df <- read.csv2('estudantes.csv')

### Exploratory data analysis
head(df)
View(df)
str(df)
any(is.na(df))

# Exploring the target variable G3
table(df$G3)
hist(df$G3, breaks = 20)
boxplot(df$G3)
mean(df$G3)
sd(df$G3)
# It does not exist outliers

# Exploring the family size variable
ggplot(data = df) +
  geom_boxplot(mapping = aes(x = famsize, y = G2, color = famsize))
ggplot(data = df) +
  geom_boxplot(mapping = aes(x = famsize, y = G3, color = famsize))
# Small families seems to have grades a little better and less disperse than big families


# Exploring the correlation between many variable
variable.names(df)
str(df)
pairs.panels(df[c("age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
                    "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3")])

# traveltime, failure, Dalc, abscences histograms seem to decrease like an exponential function
# Medu and Fedu have a moderate correlation equals to 0.62
# Dalc and Walc have a moderate correlation equals to 0.65
#
# target variable G3 correlations -> -0.36 failures, 0.22 Medu, -0.16 age, 0.15 Fedu, 
#                                    -0.13 goout, -0.12 traveltime, and 0.10 studytime.
#                                    strong correlations -> 0.80 G1 and 0.90 G2.
# Is G3 compound of G1 and G2? As shown in the data set site, it is not.


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

# Model 1 - Linear model with G3 depending on all variables 
#           in the training data set without modification
model1 <- lm(G3 ~ ., data = data_train)
model1

# Model prediction with the test data set
prediction1 <- predict(model1, data_test)

# Plot of prediction and G3 of test data set
plot(data_test$G3, pch = 20)
points(prediction1, pch = 20, col = 'blue')
for (i in 1:length(data_test$G3)) {
  lines(c(i, i), c(data_test$G3[i], prediction1[i]), col = 'red', lw = 2)
}

# Histogram of the residue per standard deviation
Residue = (data_test$G3 - prediction1) / sd(data_test$G3)
hist(Residue, breaks = 10)

# Residue value
(Residue_value1 = sum((data_test$G3 - prediction1) / sd(data_test$G3)))

# Summary of model 1
summary(model1)
# the correlation of G2 and G3 is very strong, and the significance is very low
# the abscence and quality of family relationships (famrel) have low significance, 
#               but the relative errors are very high, about 30%
# the romantic relationship, if the student wants to take higher education (higher),
#               and attended nursery school have low significance, but the 
#               the relative errors are very high, about 50%

################################################################################
################################################################################
################################################################################

# Model 2 - Linear model with G3 depending on the variables that reached the
#           smallest significance in the model 1
model2 <- lm(G3 ~ G2 + absences + famrel + romantic + higher + nursery, data = data_train)

# Summary of model 2
summary(model2)

# Model prediction with the test data set
prediction2 <- predict(model2, data_test)

# Residue value
(Residue_value2 = sum((data_test$G3 - prediction2) / sd(data_test$G3)))

# The R^2 of the training data set of model 2 is higher than the model 1, but
#       the residue of the test data set is higher.

################################################################################
################################################################################
################################################################################

# Model 3 - Linear model with G3 depending on the variables that reached the
#           smallest significance in the model 2
model3 <- lm(G3 ~ G2 + absences + famrel + romantic + higher, data = data_train)

# Summary of model 3
summary(model3)

# Model prediction with the test data set
prediction3 <- predict(model3, data_test)

# Residue value
(Residue_value3 = sum((data_test$G3 - prediction3) / sd(data_test$G3)))

# Root Mean Squared Error
(RMSE = sqrt(sum((data_test$G3 - prediction3)^2) / length(data_test$G3)))

# The R^2 of the training data set of model 3 is higher than the model 2 and
#       the residue of the test data set is lower.

################################################################################
################################################################################
################################################################################

# Model 4 - Linear model with G3 depending on the variables of the model 3, 
#           except the romantic variable
model4 <- lm(G3 ~ G2 + absences + famrel + higher, data = data_train)

# Summary of model 4
summary(model4)

# Model prediction with the test data set
prediction4 <- predict(model4, data_test)

# Residue value
(Residue_value4 = sum((data_test$G3 - prediction4) / sd(data_test$G3)))

# The R^2 and residue were worst than the previous model.

################################################################################
################################################################################
################################################################################

# Model 5 - Variation of model 3

data_train$absences2 <- data_train$absences^(1/25)
data_test$absences2 <- data_test$absences^(1/25)
data_train$famrel2 <- data_train$famrel^1.1
data_test$famrel2 <- data_test$famrel^1.1
data_train$G22 <- data_train$G2^1.2
data_test$G22 <- data_test$G2^1.2

# Linear model
model5 <- lm(G3 ~ G22 + absences2 * famrel2, data = data_train)

# Summary of madel 5
summary(model5)

# Model prediction with the test data set
prediction5 <- predict(model5, data_test)

# Residue value
(Residue_value5 = sum((data_test$G3 - prediction5) / sd(data_test$G3)))

# Root Mean Squared Error
(RMSE = sqrt(sum((data_test$G3 - prediction5)^2) / length(data_test$G3)))

# The R^2, residue, and RMSE were better than the model3.

data_train$absences2 <- NULL
data_test$absences2 <- NULL
data_train$famrel2 <- NULL
data_test$famrel2 <- NULL
data_train$G22 <- NULL
data_test$G22 <- NULL

################################################################################
################################################################################
################################################################################

# Model 6 - Variation of model 5


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

# Linear model
model6 <- lm(G3 ~ G22 * absences2 * famrel2 * failures2, data = data_train)

# Summary of madel 6
summary(model6)

# Model prediction with the test data set
prediction6 <- predict(model6, data_test)

# Residue value
(Residue_value6 = sum((data_test$G3 - prediction6) / sd(data_test$G3)))

# Root Mean Squared Error
(RMSE = sqrt(sum((data_test$G3 - prediction5)^2) / length(data_test$G3)))

# The R^2, residue, and RMSE values were better than the model5.

# What are the predict values bigger than 2 points in relation to the observations?
discrepancies <- data_test[abs(data_test$G3 - prediction6) >= 2, ]
View(discrepancies)
# Proportion of discrepancies
(nrow(discrepancies)/nrow(data_test))
# Correlations and histograms of the discrepancies data set
pairs.panels(discrepancies[c("age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
                  "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3")])

# Analyzing the correlation between G3 and the other variables, it shows that the 
#           variables Medu, goout, and studytime can be implemented.

data_train$absences2 <- NULL
data_test$absences2 <- NULL
data_train$famrel2 <- NULL
data_test$famrel2 <- NULL
data_train$G22 <- NULL
data_test$G22 <- NULL
data_train$failures2 <- NULL
data_test$failures2 <- NULL


################################################################################
################################################################################
################################################################################

# Model 7 - Including more variables

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

# Linear model
model7 <- lm(G3 ~ (absences2 * failures2) * (famrel2 + Medu) * (studytime + G22), data = data_train)

# Summary of madel 6
summary(model7)

# Model prediction with the test data set
prediction7 <- predict(model7, data_test)

# Residue value
(Residue_value7 = sum((data_test$G3 - prediction7) / sd(data_test$G3)))

# Root Mean Squared Error
(RMSE = sqrt(sum((data_test$G3 - prediction7)^2) / length(data_test$G3)))

# The R^2, residue, and RMSE values were better than the model6.

# Predict values biggee than 2 points in relation to the observations
discrepancies <- data_test[abs(data_test$G3 - prediction7) >= 2, ]
View(discrepancies)
# Proportion of discrepancies
(nrow(discrepancies)/nrow(data_test))


# The best so far.


# Plots
# Residue histogram
Residues7 <- (data_test$G3 - prediction7) / sd(data_test$G3)
hist(Residues7)

# Residue boxplot
boxplot(Residues7)
# It shows some outliers

# Plot of prediction and G3 of test data set
plot(data_test$G3, pch = 20, xlab = 'Student', ylab = 'Final grade (G3)')
points(prediction7, pch = 20, col = 'blue')
for (i in 1:length(data_test$G3)) {
  lines(c(i, i), c(data_test$G3[i], prediction7[i]), col = 'red', lw = 2)
}


data_train$absences2 <- NULL
data_test$absences2 <- NULL
data_train$famrel2 <- NULL
data_test$famrel2 <- NULL
data_train$G22 <- NULL
data_test$G22 <- NULL
data_train$failures2 <- NULL
data_test$failures2 <- NULL


### The continuation of this analysis with the Decision Tree algorithm is in the 
### file "student_grades_PART2_Decision_Tree.R"