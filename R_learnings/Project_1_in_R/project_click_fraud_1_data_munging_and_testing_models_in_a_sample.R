#### PART ONE ####

# Data fields
# Each row of the training data contains a click record, with the following features.

# ip: ip address of click.
# app: app id for marketing.
# device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
# os: os version id of user mobile phone
# channel: channel id of mobile ad publisher
# click_time: timestamp of click (UTC)
# attributed_time: if user download the app for after clicking an ad, this is the time of the app download
# is_attributed: the target that is to be predicted, indicating the app was downloaded
# Note that ip, app, device, os, and channel are encoded.

# Problem: Predict the is_attributed features
# Data set site: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data

# The solution to this problem was divided into four parts. The first part is 
# in this script. It deals with the data munging and the testing of many machine 
# learning models using the train_sample.csv file and testing with 1E+07 rows of 
# the train.csv. The data of this file was used as the test dataset because the 
# dataset provided did not have the target variable.

# The second part of the solution got the main tidying lines of part one to tidy 
# the full training dataset, nominated train.csv. In the third part, the tidying 
# training dataset was taken with the best model acquired in part one to train 
# the model, but the number of the trees of the random forest model was reduced 
# due to my notebook capacity. In the fourth part, the trained model was applied 
# to the provided test dataset, test.csv. Afterward, the predicted results were 
# matched with the click_id to produce the submission file.


# Removes all existing objects and packages from the current workspace
rm(list = ls())
# Working directory 
# setwd("~/Documents/DSA/Big_Data_Analytics_com_R_e_Microsoft_Azure_Machine_Learning/Cap20_Projetos_com_feedback/project1")
# getwd()

# Packages
library(dplyr)
library(lubridate)
library(ggplot2)
library(ggthemes)
library(mltools)
library(data.table)
library(caret)
library(ROCR) 

# Read the data sets
train_set <- read.csv(file = 'train_sample.csv', header = T)
#test_set <- fread(file = 'test.csv', header = T)

# The train dataset named train.csv can be found on the web site
# https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data
test_set <- fread(file = 'train.csv', header = T, nrows = 1e7)

########## Exploratory data analysis ##########

# Missing values
any(is.na(train_set))
any(is.na(test_set))

# Overview
dim(train_set)
head(train_set)
str(train_set)

dim(test_set)
head(test_set)
str(test_set)


# The target variable is categorical
train_set$is_attributed <- as.factor(train_set$is_attributed)
test_set$is_attributed <- as.factor(test_set$is_attributed)
table(train_set$is_attributed)
# table(test_set$is_attributed)
                                        # It has other categorical ip, variables,
                                        # like the app, device, os, and channel, 
                                        # but it seems to be not practical to 
                                        # convert these variables at this time.

# Train dataset summary
summary(train_set)
summary(test_set)

# Unique values of the ip feature
length(unique(train_set$ip))        # 34857 values in 100000 of the total.
length(unique(test_set$ip))         # 93936 values in 18790469 of the total.
head(rev(sort(table(train_set$ip))))
head(rev(sort(table(test_set$ip)))) # Waw!!! It has many ip repetitions.
                                    # I thought it had had much less than that.
                                    # Some ips have so many repetitions that
                                    # I think I will have to make classes to
                                    # compute this dependency and analyze if
                                    # the target variable has a strong 
                                    # dependency on the ip variable.
                                    # Maybe classes depending on the number 
                                    # of repetitions.

# Duplicated ips
dupl_ips_train <- train_set[duplicated(train_set$ip), 1]
length(dupl_ips_train)
length(unique(dupl_ips_train))
round(prop.table(table(train_set$is_attributed[train_set$ip %in% 
                                unique(dupl_ips_train)])) * 100, 2)

dupl_ips_test <- train_set[duplicated(test_set$ip), 1]
length(dupl_ips_test)
length(unique(dupl_ips_test))
round(prop.table(table(train_set$is_attributed[train_set$ip %in% 
                                unique(dupl_ips_test)])) * 100, 2)

# Repeated ips in order
n_dupl_ips_train <- train_set %>%
  count(ip, wt = n() ) %>%
  arrange(desc(n))

head(n_dupl_ips_train)

n_dupl_ips_test <- test_set %>%
  count(ip, wt = n() ) %>%
  arrange(desc(n))

head(n_dupl_ips_test)

# Verifyind the total of lines
sum(n_dupl_ips_train$n)
sum(n_dupl_ips_test$n)

# Number of duplicate ips column
train_set <- left_join(train_set, n_dupl_ips_train, by = 'ip')
head(train_set)
test_set <- left_join(test_set, n_dupl_ips_test, by = 'ip')
head(test_set)

# Rename the n columns
names(train_set)[9] <- 'repetitions'
labels(train_set)[[2]]
names(test_set)[9] <- 'repetitions'
# names(test_set)[8] <- 'repetitions'
labels(test_set)[[2]]

# The number of ips repeated depending on the number of repetitions
c = 1
values <- unique(n_dupl_ips_train$n)
df <- data.frame(repetitions = rep(NA, length(values)))
for (i in values) {
  df$repetitions[c] <- i
  
  tab <- table(train_set$is_attributed[train_set$repetitions == i])
  df$no[c] <- tab[1]
  df$no_prop[c] <- round(tab[1] * 100 / sum(tab), 2)
  df$yes[c] <- tab[2]
  df$yes_prop[c] <- round(tab[2] * 100 / sum(tab), 2)
  
  c = c + 1
}

# Verifying the number rows of  train data set is correct
sum(df$no, df$yes)

# Filter and sorting df in relation to the proportion of yes to the app
df_prop <- df %>% 
  filter(yes_prop > 0) %>%
  arrange(desc(yes_prop))

df_prop

# Scatter plot of the yes/no downloading app and the number of ips repetitions
brks <- cut(df_prop$repetitions, breaks = c(0, 5, 10, 100, 700))
ggplot(data = df_prop) +
  geom_point(aes(no/1000, yes, color = brks,
                 size = yes_prop), alpha = 0.8) +
  xlab('no (x10³)') +
  scale_color_manual(values = c(1,3,2,4)) +
  scale_size(breaks = c(0.5, 1, 2)) +
  coord_trans(x = 'log', y = 'log') +
  ggtitle('The app was downloaded') +
  theme_linedraw()
  
# Scatter plot of the yes/no downloading app and the number of ips repetitions
ggplot(data = df_prop) +
  geom_point(aes(repetitions, yes_prop, size = yes_prop), alpha = 0.8) +
  scale_color_manual(values = c(1,3,2,4)) +
  scale_size(breaks = c(0.5, 1, 2)) +
  coord_trans(x = 'log') +
  ggtitle('The app was downloaded') +
  theme_linedraw()
                                    # the yes proportions seems to behave like
                                    # a sinusoid

# Iserting anoter columns according to the yes_prop values
gt <- df_prop$repetitions[df_prop$yes_prop > 0.4]

train_set <- train_set %>% 
  mutate(yes_prop = ifelse(repetitions %in% gt, 1, 0))
                                    # I forget that did not have the 
                                    # is_attributed features in test data set.
                                    # I will make classes for repititions 
                                    # feature.

# repetitions classes
train_set$repetitions_fac <- cut(train_set$repetitions, 
                                 breaks = c(0,5,nrow(train_set)), 
                                 labels = c(1, 2))

test_set$repetitions_fac <- cut(test_set$repetitions, 
                                 breaks = c(0,5,nrow(test_set)), 
                                 labels = c(1, 2))

#########  TIME VARIABLE  ##################

# The click_time feature of the train data set
train_set$click_time <- as.Date(train_set$click_time, format = '%Y-%m-%d')
unique(months(train_set$click_time))      # only in november
unique(year(train_set$click_time))        # only in 2017
unique(day(train_set$click_time))         # the days are between 6 and 9
unique(weekdays(train_set$click_time))    # days 6 = Monday, 7 = Tuesday,
                                          #      8 = Wednesday, and 9 = Thursday

# The click_time feature of the test data set
test_set$click_time <- as.Date(test_set$click_time, format = '%Y-%m-%d')
unique(months(test_set$click_time))       # only in november
unique(year(test_set$click_time))         # only in 2017
unique(day(test_set$click_time))          # only the day 10th
unique(weekdays(test_set$click_time))     # day  10 = friday

# New feature containing the day of the click_time
train_set$click_day <- day(train_set$click_time)
test_set$click_day <- day(test_set$click_time)

# train click_day plot
ggplot(train_set, aes(click_day)) +
  geom_histogram(binwidth = 1, fill = 'green', col = 'black', alpha = 0.6) +
  theme_bw()
                                          # maybe the day 6 has only the 
                                          # partial data

# Number of clicks in function of the day (train)
train_set %>%
  count(click_day)

# The train attributed_time feature. It is not in the test data set.
train_set$attributed_time <- ymd_hms(train_set$attributed_time)

# New features containing the day and hour of the train attributed_time
train_set$attributed_day <- day(train_set$attributed_time)
train_set$attributed_hour <- hour(train_set$attributed_time) + 
  ifelse(minute(train_set$attributed_time) >= 30, 1, 0)

ggplot(train_set, aes(attributed_day, attributed_hour, 
                      fill = cut(attributed_hour, breaks = c(0,12,18,24)))) +
  geom_bar(stat = "identity", alpha = 0.6)

# Number of clicks in function of the day (train)
train_set %>%
  count(attributed_day)
                                          # As shown in the last results, days 6 
                                          # and 9 have fewer observations than
                                          # the other days. It seems that the 
                                          # observations of the day 6 were in 
                                          # the final of the afternoon and 
                                          # day 9 until the middle of the 
                                          # afternoon. I will eliminate the 
                                          # day 6 and 9 to have two entire days.
                                          # This data set dos not have much
                                          # positive targets. I will not
                                          # delete the day 6 and 9.

# Erase the day 6 and 9 to have two entire days (train)
# train_set <- train_set[train_set$click_day != 6 & train_set$click_day != 9, ]
# dim(train_set)

# Hour of the day that the app was downloaded
hist(train_set$attributed_hour[train_set$attributed_day == 7], 
     col = rgb(1,0,0,0.5), breaks = 24,
     main = 'Histogram of the app downloaded per hour', xlab = 'Hour')
hist(train_set$attributed_hour[train_set$attributed_day == 8], 
     col = rgb(0,0,1,0.5), breaks = 24, add = T)
legend(x = "topright", legend = c('Day 7','Day 8'), 
       col = c(rgb(1,0,0,0.5), rgb(0,0,1,0.5)), pch = 15)

hist(train_set$attributed_hour, 
     breaks = 24, main = 'Histogram of the app downloaded per hour in the two days', 
     xlab = 'Hour')
                                          # Strangely, the number of downloads
                                          # has a great decrease after 16 hours

# app feature
sort(unique(train_set$app))
sort(unique(test_set$app))

div_app<- bin_data(c(train_set$app, test_set$app), bins = 4, binType = "quantile")
levels(div_app)

train_set$app_fac <- cut(train_set$app, breaks = c(0, 3, 12, 18, nrow(train_set)), 
                     right = F, labels = c(1, 2, 3, 4))

test_set$app_fac <- cut(test_set$app, breaks = c(0, 3, 12, 18, nrow(test_set)), 
                     right = F, labels = c(1, 2, 3, 4))


plot(train_set$app_fac, xlab = 'App id class (train data set)', ylab = 'Frequency')
plot(test_set$app_fac, xlab = 'App id class (test data set)', ylab = 'Frequency')

# device feature
sort(unique(train_set$device))
sort(unique(test_set$device))

summary(train_set$device)
summary(test_set$device)

hist(train_set$device, freq = F, breaks = 40, col = rgb(1,0,0,0.5),
     main = 'Device histograms', xlab = 'Devices')
hist(test_set$device, freq = F, breaks = 40, col = rgb(0,0,1,0.5), add = T)
legend(x = "topright", legend = c('Train data set','Test data set'), 
       col = c(rgb(1,0,0,0.5), rgb(0,0,1,0.5)), pch = 15)

a <- train_set %>%
  count(device, sort = T)
head(a)

b <- test_set %>%
  count(device, sort = T)
head(b)

# Type 1 device proportion
( a[1,2]/sum(a) )                       # 58% of all devices are of type 1
( b[1,2]/sum(b) )                       # 81% of all devices are of type 1

# Making two classes of devices: one for type 1 and the other for the others
class_device <- function(x) {ifelse(x == 1, 1, 2)} 

train_set$device_fac <- as.factor(class_device(train_set$device))
levels(train_set$device_fac)

test_set$device_fac <- as.factor(class_device(test_set$device))
levels(test_set$device_fac)

# OS feature
sort(unique(train_set$os))
sort(unique(test_set$os))

summary(train_set$os)
summary(test_set$os)

# histograms
hist(train_set$os, freq = F, xlim = c(0,800), ylim = c(0, 0.07), breaks = 100, 
     col = rgb(1,0,0,0.5), main = 'OS histograms', xlab = 'OS id')
hist(test_set$os, freq = F, xlim = c(0,800), breaks = 50,
     col = rgb(0,0,1,0.5), add = T)
legend(x = "topright", legend = c('Train data set','Test data set'), 
       col = c(rgb(1,0,0,0.5), rgb(0,0,1,0.5)), pch = 15)

# smaller domain
hist(train_set$os, freq = F, xlim = c(0, 100), ylim = c(0, 0.07), breaks = 100, 
     col = rgb(1,0,0,0.5), main = 'OS histograms', xlab = 'OS id')
hist(test_set$os, freq = F, xlim = c(0,100), breaks = 50,
     col = rgb(0,0,1,0.5), add = T)
legend(x = "topright", legend = c('Train data set','Test data set'), 
       col = c(rgb(1,0,0,0.5), rgb(0,0,1,0.5)), pch = 15)

# Countings
a <- train_set %>%
  count(os, sort = T)
head(a)

b <- test_set %>%
  count(os, sort = T)
head(b)

# Type 19 and 13 os proportion
( (a[1,2] + a[2,2]) / sum(a) )
( (b[1,2] + b[2,2]) / sum(b) )
                                        # Type 19 and 13 os represent almost
                                        # 40% of the systems

# Making classes for os features
class_os <- function(x) {
  if (x == 13) {2}
  else if (x == 19) {3}
  else if (x > 19) {4}
  else {1}
}

train_set$os_fac <- as.factor(sapply(train_set$os, class_os))
plot(train_set$os_fac, xlab = 'OS classes (train data set)', ylab = 'Frequency')
test_set$os_fac <- as.factor(sapply(test_set$os, class_os))
plot(test_set$os_fac, xlab = 'OS classes (test data set)', ylab = 'Frequency')

# Channel feature
sort(unique(train_set$channel))
sort(unique(test_set$channel))

summary(train_set$channel)
summary(test_set$channel)

# histograms
hist(train_set$channel, freq = F, ylim = c(0, 0.01), breaks = 20, 
     col = rgb(1,0,0,0.5), main = 'Channel histograms', xlab = 'Channel id')
hist(test_set$channel, freq = F, breaks = 20,
     col = rgb(0,0,1,0.5), add = T)
legend(x = "topright", legend = c('Train data set','Test data set'), 
       col = c(rgb(1,0,0,0.5), rgb(0,0,1,0.5)), pch = 15)

# Countings
a <- train_set %>%
  count(channel, sort = T)
head(a)

b <- test_set %>%
  count(channel, sort = T)
head(b)

#  Balancing the four channel classes
div_channel <- bin_data(c(train_set$channel, test_set$channel), 
                        bins = 4, binType = "quantile")
levels(div_channel)

train_set$channel_fac <- cut(train_set$channel, 
                             breaks = c(0, 135, 236, 401, nrow(train_set)),
                             right = F, labels = c(1, 2, 3, 4))

test_set$channel_fac <- cut(test_set$channel, 
                            breaks = c(0, 135, 236, 401, nrow(test_set)),
                            right = F, labels = c(1, 2, 3, 4))

plot(train_set$channel_fac, xlab = 'Channel id class (train data set)', 
     ylab = 'Frequency')
plot(test_set$channel_fac, xlab = 'Channel id class (test data set)', 
     ylab = 'Frequency')

# Features that does not contain missing values
dim(train_set)
any(is.na(train_set[,1:6]))
any(is.na(train_set[,8:11]))
any(is.na(train_set[,14:17]))

# Dealing with the features with missing values
any(is.na(train_set[,7]))
labels(train_set)[[2]][7]
unique(train_set$attributed_time)
                                        # This features will not be utilized

any(is.na(train_set[,12]))
labels(train_set)[[2]][12]
unique(train_set$attributed_day)
                                        # This features will not be utilized

any(is.na(train_set[,13]))
labels(train_set)[[2]][13]
unique(train_set$attributed_hour)
                                        # This features will not be utilized


# Reducing the quantity of not downloaded to balance the train target feature
n <- nrow(train_set[train_set$is_attributed == 1, ])
n

train_no <- train_set %>%
  filter(is_attributed == 0) %>%
  slice_sample(n = n, replace = F)
nrow(train_no)

train_yes <- train_set %>%
  filter(is_attributed == 1)
nrow(train_yes)

train_set1 <- rbind(train_no, train_yes)
train_set1 <- train_set1 %>% 
  slice_sample(n = nrow(train_set1), replace = F)
nrow(train_set1)/2

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'train_set1', 'test_set')))
ls()
gc()

################################################################################
################################################################################
################################################################################

# logistic regression model
labels(test_set)[[2]]

model1 <- glm(is_attributed ~ repetitions_fac + app_fac + 
                device_fac + os_fac + channel_fac, 
                data = train_set1, 
                family = "binomial")

# Summary of the model
summary(model1)

# Predictions
predictions1 <- predict(model1, test_set, type="response")
predictions1 <- round(predictions1)

# Evaluation
confusionMatrix(as.factor(predictions1), 
                reference = test_set$is_attributed, positive = '1')

# ROC curve
predictions1_roc <- prediction(predictions1, test_set$is_attributed)
source("plot_utils.R") 
par(mfrow = c(1,2))
plot.roc.curve(predictions1_roc, title.text = "Curva ROC")
plot.pr.curve(predictions1_roc, title.text = "Curva Precision/Recall")
par(mfrow = c(1,1))

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'train_set1', 'test_set')))
gc()

################################################################################
################################################################################
################################################################################

# logistic regression model with the most significant variables
model2 <- glm(is_attributed ~ repetitions + device_fac + os_fac, 
              data = train_set1, 
              family = "binomial")

# Summary of the model
summary(model2)

# Predictions
predictions2 <- predict(model2, test_set, type="response")
predictions2 <- round(predictions2)

# Evaluation
confusionMatrix(as.factor(predictions2), 
                reference = test_set$is_attributed, positive = '1')

# Criando curvas ROC
predictions2_roc <- prediction(predictions2, test_set$is_attributed)
source("plot_utils.R") 
par(mfrow = c(1,2))
plot.roc.curve(predictions2_roc, title.text = "Curva ROC")
plot.pr.curve(predictions2_roc, title.text = "Curva Precision/Recall")
par(mfrow = c(1,1))

#  Conclusion: the AUC value decrease in relation to the previous model.
#              The AUC value is the Balanced Accuracy of the
#              confusionMatrix results.

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'train_set1', 'test_set')))
gc()
detach(package:ROCR) 

################################################################################
################################################################################
################################################################################

# KSVM model with rbf kernel
library(kernlab)
model3 <- ksvm(is_attributed ~ repetitions + app_fac + 
                device_fac + os_fac + channel_fac, 
              data = train_set1, 
              kernel = 'rbf')

# Summary of the model
summary(model3)

# Predictions
predictions3 <- predict(model3, test_set, type="response")

# Evaluation
confusionMatrix(predictions3, 
                reference = test_set$is_attributed, positive = '1')

# Conclusion: the first model is still the best one.

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'train_set1', 'test_set')))
gc()

################################################################################
################################################################################
################################################################################

# KSVM model with rbf kernel and the most significant variables
model4 <- ksvm(is_attributed ~ repetitions + device_fac + os_fac,
               data = train_set1, 
               kernel = 'rbf')

# Predictions
predictions4 <- predict(model4, test_set, type="response")

# Evaluation
confusionMatrix(predictions4, 
                reference = test_set$is_attributed, positive = '1')

# Conclusion: the first model is still the best one. It is worst than the
#             previous model.

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'train_set1', 'test_set')))
gc()

################################################################################
################################################################################
################################################################################

# KSVM model with vanilladot Linear kernel
model5 <- ksvm(is_attributed ~ repetitions + app_fac + 
                 device_fac + os_fac + channel_fac, 
               data = train_set1, 
               kernel = 'vanilla')

# Predictions
predictions5 <- predict(model5, test_set, type="response")

# Evaluation
confusionMatrix(predictions5, 
                reference = test_set$is_attributed, positive = '1')

# Conclusion: now this is the best model so far.

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'train_set1', 'test_set')))
gc()

################################################################################
################################################################################
################################################################################

# KSVM model with vanilladot Linear kernel and the most significant variables
model6 <- ksvm(is_attributed ~ repetitions + device_fac + os_fac,
               data = train_set1, 
               kernel = 'vanilla')

# Predictions
predictions6 <- predict(model6, test_set, type="response")

# Evaluation
confusionMatrix(predictions6, 
                reference = test_set$is_attributed, positive = '1')

# Conclusion: the model 5 scores better.

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'train_set1', 'test_set')))
gc()
detach(package:kernlab)

################################################################################
################################################################################
################################################################################

# SVM model with radial kernel
library(e1071)
model7 <- svm(is_attributed ~ repetitions + app_fac + 
                 device_fac + os_fac + channel_fac, 
               data = train_set1, 
               kernel = 'radial')

# Predictions
predictions7 <- predict(model7, test_set, type="response")

# Evaluation
confusionMatrix(predictions7, 
                reference = test_set$is_attributed, positive = '1')

# Conclusion: the model 5 scores better.

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'train_set1', 'test_set')))
gc()

################################################################################
################################################################################
################################################################################

# SVM model with radial kernel and the most significant variables
model8 <- svm(is_attributed ~ repetitions + device_fac + os_fac, 
              data = train_set1, 
              kernel = 'radial')

# Predictions
predictions8 <- predict(model8, test_set, type="response")

# Evaluation
confusionMatrix(predictions8, 
                reference = test_set$is_attributed, positive = '1')

# Conclusion: this model is not good.

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'train_set1', 'test_set')))
gc()

################################################################################
################################################################################
################################################################################

# SVM model with linear kernel
model9 <- svm(is_attributed ~ repetitions + app_fac + 
                device_fac + os_fac + channel_fac, 
              data = train_set1, 
              kernel = 'linear',
              type = 'C-classification')

# Predictions
predictions9 <- predict(model9, test_set, type="response")

# Evaluation
confusionMatrix(predictions9, 
                reference = test_set$is_attributed, positive = '1')

# Conclusion: it is equal to model 5.

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'train_set1', 'test_set')))
gc()

################################################################################
################################################################################
################################################################################

# SVM model with linear kernel and the most significant variables
model10 <- svm(is_attributed ~ repetitions + device_fac + os_fac, 
              data = train_set1, 
              kernel = 'linear',
              type = 'C-classification')

# Predictions
predictions10 <- predict(model10, test_set, type="response")

# Evaluation
confusionMatrix(predictions10, 
                reference = test_set$is_attributed, positive = '1')

# Conclusion: the model 5 scores better.

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'train_set1', 'test_set')))
gc()
detach(package:e1071)

################################################################################
################################################################################
################################################################################

# Regression Trees model
library(rpart.plot)
model11 <- rpart(is_attributed ~ repetitions + app_fac + 
                   device_fac + os_fac + channel_fac, 
                 data = train_set1)

# Predictions
predictions11 <- predict(model11, test_set, type="class")

# Evaluation
confusionMatrix(predictions11, 
                reference = test_set$is_attributed, positive = '1')

# Conclusion: it is the best model so far.

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'train_set1', 'test_set')))
gc()

################################################################################
################################################################################
################################################################################

# Evaluation of the most important features for the model
model12 <- train(is_attributed ~ repetitions + app_fac + 
                   device_fac + os_fac + channel_fac, 
                 data = train_set1,
                 method = 'rpart')
varImp(model12)

# Regression Trees model with the most significant variables
model12 <- rpart(is_attributed ~ repetitions + app_fac + 
                   device_fac + channel_fac, 
                 data = train_set1)

# Predictions
predictions12 <- predict(model12, test_set, type="class")

# Evaluation
confusionMatrix(predictions12, 
                reference = test_set$is_attributed, positive = '1')

# Conclusion: the model 11 is still the best model.

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'train_set1', 'test_set')))
gc()
detach(package:rpart.plot)

################################################################################
################################################################################
################################################################################

# Another Regression Trees model
library(C50)
model13 <- C5.0(is_attributed ~ repetitions_fac + app_fac + 
                  device_fac + os_fac + channel_fac, 
                data = train_set1, 
                trials = 10,
                cost = matrix(c(0, 8, 1, 0), nrow = 2, 
                              dimnames = list(c('0','1'), c('0', '1'))))

# Predictions
predictions13 <- predict(model13, test_set, type="class")

# Evaluation
confusionMatrix(predictions13, 
                reference = test_set$is_attributed, positive = '1')

# Conclusion: it is the best model so far.

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'train_set1', 'test_set')))
gc()

################################################################################
################################################################################
################################################################################

# Another Regression Trees model with the most significant variables
model14 <- C5.0(is_attributed ~ repetitions + app_fac + 
                  device_fac + channel_fac, 
                data = train_set1, 
                trials = 10,
                cost = matrix(c(0, 2, 1, 0), nrow = 2, 
                              dimnames = list(c('0','1'), c('0', '1'))))

# Predictions
predictions14 <- predict(model14, test_set, type="class")

# Evaluation
confusionMatrix(predictions14, 
                reference = test_set$is_attributed, positive = '1')

# Conclusion: The previous model was better.

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'train_set1', 'test_set')))
gc()
detach(package:C50)

################################################################################
################################################################################
################################################################################

# Random Forest model
library(randomForest)

# Feature importances
model <- randomForest(is_attributed ~ . - click_time - click_day 
                      - attributed_time - attributed_day - attributed_hour, 
                      data = train_set1, 
                      ntree = 30,
                      nodesize = 1, importance = T)

varImpPlot(model)

# Random forest model
model15 <- randomForest(is_attributed ~ repetitions_fac * app + 
                          channel * app_fac, 
                data = train_set1, 
                ntree = 30,
                nodesize = 1)

# Predictions
predictions15 <- predict(model15, test_set, type="class")

# Evaluation
confusionMatrix(predictions15, 
                reference = test_set$is_attributed, positive = '1')

# Conclusion: This is the best model.

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'train_set1', 'test_set')))
gc()

################################################################################
################################################################################
################################################################################

# Reducing the quantity of not downloaded to balance the train target feature
train_set1 <- downSample(x = train_set %>% select(-is_attributed),
                         y = train_set$is_attributed, yname = 'is_attributed')
table(train_set1$is_attributed) 

# Random forest model
model15 <- randomForest(is_attributed ~ repetitions_fac * app + 
                          channel * app_fac, 
                        data = train_set1, 
                        ntree = 30,
                        nodesize = 1)

# Predictions
predictions15 <- predict(model15, test_set, type="class")

# Evaluation
confusionMatrix(predictions15, 
                reference = test_set$is_attributed, positive = '1')

# Conclusion: Reducing the major target class by the downSample method did not 
#             change the results.

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'test_set')))
gc()

################################################################################
################################################################################
################################################################################

# Increasing minor target class
train_set1 <- upSample(x = train_set %>% select(-is_attributed),
                         y = train_set$is_attributed, yname = 'is_attributed')
table(train_set1$is_attributed) 

# Random forest model
model15 <- randomForest(is_attributed ~ repetitions_fac * app + 
                          channel * app_fac, 
                        data = train_set1, 
                        ntree = 30,
                        nodesize = 1)

# Predictions
predictions15 <- predict(model15, test_set, type="class")

# Evaluation
confusionMatrix(predictions15, 
                reference = test_set$is_attributed, positive = '1')

# Conclusion: Enlarging the minor target class make the results worst.

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'test_set')))
gc()

################################################################################
################################################################################
################################################################################

# Balancing the target class
library(DMwR)
train_set1 <- train_set %>% 
  select(is_attributed, repetitions_fac, app, channel, app_fac)

train_set1 <- SMOTE(is_attributed ~ ., data  = train_set1)
table(train_set1$is_attributed) 

# Random forest model
model15 <- randomForest(is_attributed ~ repetitions_fac * app + 
                          channel * app_fac, 
                        data = train_set1, 
                        ntree = 30,
                        nodesize = 1)

# Predictions
predictions15 <- predict(model15, test_set, type="class")

# Evaluation
confusionMatrix(predictions15, 
                reference = test_set$is_attributed, positive = '1')

# Conclusion: Balancing the data with SMOTE improved slightly the results.

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'test_set')))
gc()
detach(package:DMwR)

################################################################################
################################################################################
################################################################################

# Balancing the target class
library(ROSE)
train_set1 <- train_set %>% 
  select(is_attributed, repetitions_fac, app, channel, app_fac)

train_set1 <- ROSE(is_attributed ~ ., data  = train_set1)$data                       
table(train_set1$is_attributed) 

# Random forest model
model15 <- randomForest(is_attributed ~ repetitions_fac * app + 
                          channel * app_fac, 
                        data = train_set1, 
                        ntree = 30,
                        nodesize = 1)

# Predictions
predictions15 <- predict(model15, test_set, type="class")

# Evaluation
confusionMatrix(predictions15, 
                reference = test_set$is_attributed, positive = '1')

# Conclusion: This worse the results.

# Cleaning the house
rm(list = setdiff(ls(), c('train_set', 'test_set')))
gc()

################################################################################

# Continue on part two, 
#                  filename project_click_fraud_2_tidying_in_the_train_dataset.R