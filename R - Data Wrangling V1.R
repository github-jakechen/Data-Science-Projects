#Test Cheatsheet

#Packages needed
install.packages("readxl")
install.packages("car")
install.packages("estimatr")
install.packages("tidyverse")
install.packages("caret")
install.packages("ggplot2")
install.packages("mice")
install.packages("tidyr")
install.packages("forecast")

library(readxl)
library(car)
library(estimatr)
library(tidyverse)
library(caret)
library(ggplot2)
library(mice)
library(tidyr)
library(glmnet)
library(forecast)

#-------------------------------------------------------
#                     Import Data

#Import data
path <-"C:\\Users\\jake.chen\\Desktop\\MMA\\MMA 860 Acquisition and Management of Data\\S5 - Insights from Regression\\Cheat_Sheet_Chow_Test_V1_0.xlsx"
sheetname <-"Structural_Change_Data"
data1 <- read_excel(path,sheet = sheetname)
#If you're analyzing the first X rows, then you need the function "n_max=X" at the end.
#If you're skipping the first x rows, then you need the function "skip = x" at the end.

#For CSV
data1 <- read.csv(file = 'C:\\Users\\dimen\\Documents\\MMA\\MMA 831 Marketing Analytics\\Assignment 1\\eureka_data_final.csv')

#Data examination
str(data1) #show the structure of (data types in) the diamond.data dataframe
head(data1, 4) #show the first 4 rows in the diamond.data dataframe
tail(data1,4) #show the last 4 rows in the diamond.data dataframe
summary(data1) #show summary statistics of the diamond.data dataframe

#-------------------------------------------------------
#                     Outlier
  
plot(reg1)  
  

#-------------------------------------------------------
#                     Missing Data




#-----------------------------------------------------
#                     Regression


#Run the regression

reg1 <- lm(Y ~ X1 + X2, data1)

summary(reg1)


plot(reg1)
#Residual vs Fitted
#a random dispersion around 0 is what we want
#patterns means non-normal errors, could transform the variable to correct for this.

#Normal QQ: Quantile Quantile
#looking for the dots to hug that line directly, otherwise your model has non-normal, meaning that we're not capturing something quite right

#Scale location
#similar to "Residual vs Fitted"
#tells the same story
#should be able to tell heteroskedasticity. If it's heteroskedastic, you can't trust your F-test.

#Residual vs. Levarage
#it's about finding outliers
#the key is whether the outlier is an "influential" outlier or not
#how to differentiate the two: if it's influential, then it will be outside the red lines. If it's non-influential, it will be horizontally within the red outlier
#should remove influential outlier, and ignore non-influential outliers

plot(density(resid(reg1)))
#Plot your residuals on a density graph *highly recommended
#probability density of the error term
#if bimodal, then it means it has 2 discrete group, each with an associated patterns

#-------------------------------------------------------
#                       Visualizations

#Chart
ggplot(data1, aes(y=y_variable, x=x_variable))+geom_point() +labs(title="title", xlim(1,31))
plot(Y ~ X, data=data1)

#Histogram
histogram <- hist(data1$variable_name1)


#Partition the plotting window into 4 sections
par(mfrow=c(1,4))
plot(reg1)

#-------------------------------------------------------
#           Addressing for Heteroskedasticity

#Heteroskedasticity: the error term has a variance that follows a pattern. 
#Implications of heteroskedasticity
  #results not biased, but is inefficient (inaccurate)
  #hypothesis tests will be inaccurate

#Ways to detect heteroskedasticity
  #1. Residual vs. Fitted Plot
  #2. Breusch-Pagan test


#Breusch-Pagan test
ncvTest(reg1)
#If P<alpha, then reject H0: no heteroskedasticity

#HCCME method to produce robust standard errors in the presence of heteroskedasticity
rob_reg1 <- lm_robust(Y ~ X1 + X2, data1, se_type="HC3")
summary(rob_reg1)


#-------------------------------------------------------
#                     Chow Test

#Chow test is used for structural breaks, or if differences exist between subgroups within a population
#If it turns out that two data sets behave the same, it is normally better to combine the observations to estimate one set of parameters.
#If they do not, then you typically need to allow for two or more sets of relationships.(dummy variables & interaction variables)  

#Steps:
  #1. Create dummy and interaction variables
  #2. Run regression on entire model with newly created variables (this is the unrestricted model)
  #3. Run the linearHypothesis function where the newly created functions is set to 0, which is the H0.
  #4. Rjection of H0 means that the combination of these variables are significant (i.e. there is structural break)

data1$c0 <- ifelse(data1$Week == 51, 0, 1)
#Note: Equal sign is not =, it's ==

reg2 <- lm(Y ~ X1 + X2 + X3 + c0 + c1 + c2 + c3, data1)
summary(reg2)

linearHypothesis(reg2, c("c0 = 0", "c1 = 0", "c2=0", "c3=0"))


#------------------------------------------------------------------
#               Predicting data using Test & Train


#Here, 70% is a train dataset (row 1-700), and 30% is test dataset (row 701-1000)
#Requirements:
  #Test and train sets must be mutually exclusive (no overlapping data)
  #Test and train sets must contain the same pattern of data
  #if you have time series data, you should always test on the most recent data

train <- data1[1:700,]
test <- data1[701:1000,]

#alternatively, you can sample randomly without replacement (again 70% train, 30% test)
sample <- sample.int(n = nrow(data1), size = floor(.7*nrow(data1)), replace = F)
train <- data1[sample, ]
test  <- data1[-sample, ]

#OR

train<-subset(data1, variable1<=700)
test<-subset(data1, ID>700)


#Run regression on train dataset
reg1 <- lm(Y ~ X1 + X2, train)
summary(reg)
plot(reg)

#Predict function allows you to use a linear regression to predict values.
pred <- predict(reg1,test)
#if it's  log-log model then take the exponential function to transform it back to linear predictions, instead of log
pred <-exp(predict(reg1,test))

summary(pred)

#Test & train comparisons
data.frame( R2 = R2(pred, test$Grocery_Bill),
            RMSE = RMSE(pred, test$Grocery_Bill),
            MAE = MAE(pred, test$Grocery_Bill))

#R2 = R squared
#RMSE = Root Mean Square Error; in general, lower RMSE means higher fit
#MAE = Mean Absolute Error

#Export the predicted values in csv
write.csv(pred, file = "Predicted Values.csv")



#------------------------------------------------------------------
#               Cross-fold validation -- Assessing the quality of the model 

# 1. Split the data into a training & testing dataset
# 2. Run the regression on the training data, then use it to predict the output in the testing data

reg1 <- lm(Y ~ X1 + X2, train) #build a model on training data
pred <- predict(reg1,test) #predict the output using the testing data
#Note: 
#   -The "pred" is another column added.
#   -If a log model is used, then the use "pred <-exp(predict(reg1,test))" instead.


# 3. Calculate MAPE

percent.errors <- abs((test$pred)/test$Y)*100 #calculate absolute percentage errors, using the predicted values and the actual Y values in the testing dataset
mean(percent.errors) #display Mean Absolute Percentage Error (MAPE)

# 4. Compare the MAPE of the current model with the MAPE of a different model (e.g. log model). The lower the MAPE, the better the model.

#------------------------------------------------------------------
#                  Interactions






#------------------------------------------------------------------
#               Variable selection


reg.step <-step(lm(Y~X1*X2*X3, direction="backward"))
summary(reg.step)

reg.step <-step(lm(Y~X1*X2*X3, direction ="both"))
summary(reg.step)


#-------------------------------------------------------------------
#               Working with Time-series data

# ts function defines the dataset as timeseries starting Jan 2004 and having seasonality of frequency 12 (monthly)
output_var<- ts(data1$var1,start=2004, frequency=12)


