# Removes all existing objects and packages from the current workspace
rm(list = ls())
lapply(paste("package:", names(sessionInfo()$otherPkgs), sep=""), detach, character.only = TRUE, unload = TRUE)
# Working directory 
setwd("~/Documents/learning_Data_Science/R_scripts")
getwd()

# Problem: Predict student final grades (G3)
# Data set site: https://archive.ics.uci.edu/ml/datasets/Student+Performance
# Method: Linear Model


# Packages
library(corrplot)
library(ggplot2)
library(caTools)
library(dplyr)
# library(ggthemes)
# library(dplyr)


# Read the data set
df <- read.csv2('students.csv')
View(df)

### Exploratory data analysis
summary(df)
str(df)
any(is.na(df))


# Numeric columns
numeric_columns <- sapply(df, is.numeric)
numeric_columns

# Correlation between numeric variable
data_cor <- cor(df[,numeric_columns])
data_cor

# Correlation plot
corrplot(data_cor, method = 'color')
# This plot shows that there are three major positive correlation spots near the 
# main diagonal: the first is between Medu and Fedu, the second has freetime, 
# goout, Dalc, and Walc, and the third is the strongest and it has G1, G2, and G3.

# G3 histogram
ggplot(df, aes(x = G3)) + 
  geom_histogram(bins = 20, color = 'black' ,fill = 'green') + 
  ylab('Frequency') +
  theme_minimal()

# 70% of training data set
# 30% of test data set
sp <- sample.split(df$age, SplitRatio = 0.70)

train_ds <- subset(df, sp == TRUE)
head(train_ds)
dim(train_ds)

test_ds <- subset(df, sp == FALSE)
head(test_ds)
dim(test_ds)

################################################################################
################################################################################
################################################################################

# Model 8 - Linear model with G3 depending on the variables of model 7

train_ds <- train_ds %>%
  mutate(absences2 = ifelse (absences > 0, 1, 0))
test_ds <- test_ds %>%
  mutate(absences2 = ifelse (absences > 0, 1, 0))

train_ds <- train_ds %>%
  mutate(failures2 = ifelse (failures > 0, 1, 0))
test_ds <- test_ds %>%
  mutate(failures2 = ifelse (failures > 0, 1, 0))

train_ds$famrel2 <- train_ds$famrel^0.01
test_ds$famrel2 <- test_ds$famrel^0.01

train_ds$G22 <- train_ds$G2^1.2
test_ds$G22 <- test_ds$G2^1.2

model_8 <- lm(G3 ~ (absences2 * failures2) * (famrel2 + Medu) * (studytime + G22),
              data = train_ds)

# Summary
summary(model_8)

# Residuals
res <- residuals(model_8)
res <- as.data.frame(res)

# Residue histogram
ggplot(res, aes(res)) +  
  geom_histogram(color = 'black', fill = 'red', binwidth = 1) + 
  xlab('Residue') +
  ylab('Frequency') +
  theme_minimal()

# Plot of the model 8
par(mfrow = c(2,2))
plot(model_8)
par(mfrow = c(1,1))
# The Normal Q-Q residual plot of model 8 lost the normality for values 
# under z = 1.5. It can be seen in the previous histogram.

# Predictions
predictions_8 <- predict(model_8, test_ds)

# Negative predicton grades
sum(predictions_8 < 0)

neg <- function(x){
  if  (x < 0){
    return(0)
  }else{
    return(x)
  }
}

predictions_8 <- sapply(predictions_8, neg)

# Root Mean Squared Error
(RMSE = sqrt((mean(test_ds$G3 - predictions_8)^2)))

# Residual sum of squares
(RSS = sum((test_ds$G3 - predictions_8)^2))
# Total sum of squares
(TSS = sum((test_ds$G3 - mean(df$G3))^2))

# R-Squared
(R2 = 1 - (RSS/TSS))

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
