# Removes all existing objects and packages from the current workspace
rm(list = ls())
lapply(paste("package:", names(sessionInfo()$otherPkgs), sep=""), detach, character.only = TRUE, unload = TRUE)
# Working directory 
setwd("~/Documents/learning_Data_Science/R_scripts")
getwd()

# Problem: Predict student final grades (G3)
# Data set site: https://archive.ics.uci.edu/ml/datasets/Student+Performance
# Method: Neural Network with a linear model

# Definindo o Problema: Analisando dados das casas de Boston, nos EUA e fazendo previsoes.

# The Boston Housing Dataset
# http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html

# Seu modelo deve prever a MEDV (Valor da Mediana de ocupação das casas). 

# Packages
library(corrplot)
library(ggplot2)
library(neuralnet)
library(caTools)

# Read the data set
df <- read.csv2('students.csv', stringsAsFactors = F)
View(df)

### Exploratory data analysis
summary(df)
str(df)
any(is.na(df))

# Numeric columns
numeric_columns <- sapply(df, is.numeric)
numeric_columns

# Normalizacao 
maxs <- apply(df[ ,numeric_columns], 2, max) 
maxs
mins <- apply(df[ ,numeric_columns], 2, min)
mins

# Normalizando
?scale
normalized_df <- as.data.frame(scale(df[ ,numeric_columns], center = mins, scale = maxs - mins))
head(normalized_df)
summary(normalized_df)

# Correlation between numeric variable
data_cor <- cor(df[ ,numeric_columns])
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

# Character variable transformed in numeric data
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

# Model 8 - Linear model with G3 depending on the variables of model 7

# Treinando o Modelo
rede_neural <- neuralnet(G3 ~ ., data = test_ds, hidden = c(20, 10, 5, 2), linear.output = TRUE)

# Plot
plot(rede_neural)

# Fazendo previsoes com os dados de teste
a <- predict(rede_neural, test_ds)

# Convertendo os dados de teste
previsoes <- a * (max(df$G3) - min(df$G3)) + min(df$G3)
teste_convert <- (test_ds$G3) * (max(df$G3) - min(df$G3)) + min(df$G3)
teste_convert

# Negative predicton grades
sum(previsoes < 0)
neg <- function(x){
  if  (x < 0){
    return(0)
  }else{
    return(x)
  }
}

previsoes <- sapply(previsoes, neg)


# Root Mean Squared Error
(RMSE = sqrt((mean(teste_convert - previsoes)^2)))

# Residual sum of squares
(RSS = sum((teste_convert - previsoes)^2))
# Total sum of squares
(TSS = sum((teste_convert - mean(previsoes))^2))

# R-Squared
(R2 = 1 - (RSS/TSS))

# Proporçaõ de notas iguais
sum(teste_convert == round(previsoes))/length(teste_convert)

# Obtendo os erros de previsao
error.df <- data.frame(teste_convert, previsoes = round(previsoes))
head(error.df)



# Plot dos erros
ggplot(error.df, aes(x = teste_convert,y = previsoes)) + 
  geom_point() + stat_smooth()



ZZZZZZZZZZZZZZZZZZZZZZZ



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
