# Removes all existing objects and packages from the current workspace
rm(list = ls())
lapply(paste("package:", names(sessionInfo()$otherPkgs), sep=""), detach, character.only = TRUE, unload = TRUE)
# Working directory 
setwd("~/Documents/learning_Data_Science/R_scripts")
getwd()

# Problem: Predict student final grades (G3)
# Data set site: https://archive.ics.uci.edu/ml/datasets/Student+Performance
# Method: Neural Network with a linear model

# Packages
library(corrplot)
library(ggplot2)
library(neuralnet)
library(caTools)
library(dplyr)

# Read the data set
df <- read.csv2('students.csv', stringsAsFactors = F)
View(df)

### Exploratory data analysis
### The exploratory data analysis was done in PART 1 with the Linear Model
summary(df)
str(df)
any(is.na(df))

# Numeric columns
numeric_columns <- sapply(df, is.numeric)
numeric_columns

# Normalization
maxs <- apply(df[ , numeric_columns], 2, max) 
maxs
mins <- apply(df[ , numeric_columns], 2, min)
mins

?scale
normalized_df <- as.data.frame(scale(df[ , numeric_columns], center = mins, scale = maxs - mins))
head(normalized_df)
summary(normalized_df)

# Correlation between numeric variable
cor_data <- cor(df[ , numeric_columns])
cor_data

# Correlation plot
corrplot(cor_data, method = 'color')
# This plot shows that there are three major positive correlation spots near the 
# main diagonal: the first is between Medu and Fedu, the second has freetime, 
# goout, Dalc, and Walc, and the third is the strongest and it has G1, G2, and G3.

# G3 histogram
ggplot(df, aes(x = G3)) + 
  geom_histogram(bins = 20, color = 'black' ,fill = 'green') + 
  ylab('Frequency') +
  theme_minimal()

# Character variables transformed in numeric data
factor_columns <- sapply(df, is.factor)
summary(df[, factor_columns])

normalized_df$school <- ifelse(df$school == 'GP', 0, 1)
normalized_df$sex <- ifelse(df$sex == 'F', 0, 1)
normalized_df$address <- ifelse(df$address == 'R', 0, 1)
normalized_df$famsize <- ifelse(df$famsize == 'GT3', 0, 1)
normalized_df$Pstatus <- ifelse(df$Pstatus == 'A', 0, 1)
normalized_df$schoolup <- ifelse(df$schoolup == 'no', 0, 1)
normalized_df$famsup <- ifelse(df$famsup == 'no', 0, 1)
normalized_df$paid <- ifelse(df$paid == 'no', 0, 1)
normalized_df$activities <- ifelse(df$activities == 'no', 0, 1)
normalized_df$nursery <- ifelse(df$nursery == 'no', 0, 1)
normalized_df$higher <- ifelse(df$higher == 'no', 0, 1)
normalized_df$internet <- ifelse(df$internet == 'no', 0, 1)
normalized_df$romantic <- ifelse(df$romantic == 'no', 0, 1)
normalized_df$Mjob <- ifelse(df$Mjob == 'at_home', 0,
                             ifelse(df$Mjob == 'health', 0.25,
                                    ifelse(df$Mjob == 'other', 0.50,
                                           ifelse(df$Mjob == 'services', 0.75, 1))))
normalized_df$Fjob <- ifelse(df$Fjob == 'at_home', 0,
                             ifelse(df$Fjob == 'health', 0.25,
                                    ifelse(df$Fjob == 'other', 0.50,
                                           ifelse(df$Fjob == 'services', 0.75, 1))))
normalized_df$reason <- ifelse(df$reason == 'course', 0,
                             ifelse(df$reason == 'home', 0.33,
                                    ifelse(df$reason == 'other', 0.67, 1)))
normalized_df$guardian <- ifelse(df$guardian == 'other', 0,
                               ifelse(df$guardian == 'father', 0.50, 1))

str(normalized_df)
summary(normalized_df)


# 70% of training data set
# 30% of test data set
sp <- sample.split(normalized_df$school, SplitRatio = 0.70)

train_ds <- subset(normalized_df, sp == TRUE)
head(train_ds)
dim(train_ds)

test_ds <- subset(normalized_df, sp == FALSE)
head(test_ds)
dim(test_ds)

################################################################################
################################################################################
################################################################################

# Model 1 - Neural Network model with G3 depending on all variables

# Neural Network Model
neural_model_1 <- neuralnet(G3 ~ ., 
                            data = train_ds, 
                            hidden = c(20, 10, 6, 2), 
                            rep = 3, 
                            linear.output = TRUE)

# Plot
# plot(neural_model_1)

# Prediction using the test data set
predictions_1 <- predict(neural_model_1, test_ds)

# Convertendo os dados de teste
predictions_1_conv <- predictions_1 * (max(df$G3) - min(df$G3)) + min(df$G3)
predictions_1_conv
test_ds_conv <- (test_ds$G3) * (max(df$G3) - min(df$G3)) + min(df$G3)
test_ds_conv

# Negative predicton grades
sum(predictions_1_conv < 0)
neg <- function(x){
  if  (x < 0){
    return(0)
  }else{
    return(x)
  }
}

predictions_1_conv <- sapply(predictions_1_conv, neg)


# Root Mean Squared Error
(RMSE = sqrt((mean(test_ds_conv - predictions_1_conv)^2)))

# Residual sum of squares
(RSS = sum((test_ds_conv - predictions_1_conv)^2))
# Total sum of squares
(TSS = sum((test_ds_conv - mean(predictions_1_conv))^2))

# R-Squared
(R2 = 1 - (RSS/TSS))

# Proportion of correct predections
sum(test_ds_conv == round(predictions_1_conv))/length(test_ds_conv)

# Test and predictions data frame
error.df <- data.frame(test_ds_conv, predictions_1_conv = round(predictions_1_conv))
head(error.df)

# Error plot
ggplot(error.df, aes(x = test_ds_conv, y = predictions_1_conv)) + 
  geom_point() + stat_smooth()


################################################################################
################################################################################
################################################################################

# Model 2 - Linear model with G3 depending on the variables of model 7 of PART 1

# Neural Network Model
neural_model_2 <- 
  neuralnet(G3 ~ absences + failures + famrel + Medu + studytime + G2, 
                            data = train_ds, 
                            hidden = c(20, 10, 6, 2), 
                            rep = 10, 
                            linear.output = TRUE)

# Plot
# plot(neural_model_2)

# Prediction using the test data set
predictions_2 <- predict(neural_model_2, test_ds)

# Convertendo os dados de teste
predictions_2_conv <- predictions_2 * (max(df$G3) - min(df$G3)) + min(df$G3)
# predictions_2_conv
test_ds_conv <- (test_ds$G3) * (max(df$G3) - min(df$G3)) + min(df$G3)
# test_ds_conv

# Negative predicton grades
sum(predictions_2_conv < 0)
neg <- function(x){
  if  (x < 0){
    return(0)
  }else{
    return(x)
  }
}

predictions_2_conv <- sapply(predictions_2_conv, neg)


# Root Mean Squared Error
(RMSE = sqrt((mean(test_ds_conv - predictions_2_conv)^2)))

# Residual sum of squares
(RSS = sum((test_ds_conv - predictions_2_conv)^2))
# Total sum of squares
(TSS = sum((test_ds_conv - mean(predictions_2_conv))^2))

# R-Squared
(R2 = 1 - (RSS/TSS))

# Proportion of correct predections
sum(test_ds_conv == round(predictions_2_conv))/length(test_ds_conv)

# Test and predictions data frame
error.df <- data.frame(test_ds_conv, predictions_2_conv = round(predictions_2_conv))
head(error.df)

# Error plot
ggplot(error.df, aes(x = test_ds_conv, y = predictions_2_conv)) + 
  geom_point() + stat_smooth()


################################################################################
################################################################################
################################################################################

# Model 3 - NNM with one neuron

# Neural Network Model
neural_model_3 <- 
  neuralnet(G3 ~ Medu + studytime + G1 + G2, 
            data = train_ds, 
            hidden = 1, 
            rep = 10, 
            linear.output = TRUE)

# Plot
# plot(neural_model_3)

# Prediction using the test data set
predictions_3 <- predict(neural_model_3, test_ds)

# Convertendo os dados de teste
predictions_3_conv <- predictions_3 * (max(df$G3) - min(df$G3)) + min(df$G3)
# predictions_2_conv
test_ds_conv <- (test_ds$G3) * (max(df$G3) - min(df$G3)) + min(df$G3)
# test_ds_conv

# Negative predicton grades
sum(predictions_3_conv < 0)
neg <- function(x){
  if  (x < 0){
    return(0)
  }else{
    return(x)
  }
}

predictions_3_conv <- sapply(predictions_3_conv, neg)


# Root Mean Squared Error
(RMSE = sqrt((mean(test_ds_conv - predictions_3_conv)^2)))

# Residual sum of squares
(RSS = sum((test_ds_conv - predictions_3_conv)^2))
# Total sum of squares
(TSS = sum((test_ds_conv - mean(predictions_3_conv))^2))

# R-Squared
(R2 = 1 - (RSS/TSS))

# Proportion of correct predections
sum(test_ds_conv == round(predictions_3_conv))/length(test_ds_conv)

# Test and predictions data frame
error.df <- data.frame(test_ds_conv, predictions_3_conv = round(predictions_3_conv))
head(error.df)

# Error plot
ggplot(error.df, aes(x = test_ds_conv, y = predictions_3_conv)) + 
  geom_point() + stat_smooth()


################################################################################
################################################################################
################################################################################
