#clear the environment
rm(list=ls())
#importing packages
#setwd("C:/Users/moksh/OneDrive - The University of Texas at Dallas/UTD/SEM 1 Fall 2022/ba with R/Project/telco customer churn")

#ggplot2:
#install.packages("ggplot2")
library(ggplot2)

#lattice:
library(lattice)

#caret:
#install.packages("caret")
library(caret)

#install.packages("pROC")
library(pROC)
#install.packages("plyr")
library(plyr)

#import file
churn.df <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

#converting to dataframe
churn.df <- as.data.frame(unclass(churn.df),stringsAsFactors=TRUE)
churn.df['SeniorCitizen'] = as.factor(churn.df$SeniorCitizen)

#finding the missing values
sum(is.na(churn.df))
#we find that the set has 11 missing values

#we remove the missing values
churn.df = na.omit(churn.df)

#Using ggplot2 package to explore the data
#finding the customer churn in the previous month
ggplot(data = churn.df , aes(x = Churn,fill = Churn)) + geom_bar()  + geom_text(stat = 'count' , aes(label = paste("n = " , formatC(..count..)))) + ggtitle("Customer churn")

#finding the histogram of monthly charges
ggplot(data = churn.df , aes(x = MonthlyCharges)) + geom_histogram(aes(y = ..density..),color = "red" , binwidth = 5) + ggtitle("Histogram of MonthlyCharges")

#finding the boxplot of monthly charges
ggplot(data = churn.df , aes(x = MonthlyCharges)) + geom_boxplot(fill = "grey") + ggtitle("Boxplot of MonthlyCharges")

#Feature selection
monthlycharges_cat <- subset(churn.df , select  = -c(MonthlyCharges))
chisq_test <- lapply(monthlycharges_cat[2:17] , function(x) chisq.test(monthlycharges_cat[,18] , x)$p.value)
df_chisq_test = ldply(chisq_test,data.frame)
names(df_chisq_test)[1] <- "Variable"
names(df_chisq_test)[2] <- "P_value"
ggplot(data = df_chisq_test , aes(y = Variable , x = P_value)) + geom_bar(stat = "identity" ) + geom_vline(xintercept = 0.05) + ggtitle("P-Value of Chi-squared test with p-value = 0.5 as threshold")

#with this we understand that phone service and gender columns can be ignored

#Splitting Data into test and train sets
set.seed(62)
split_val <- sample(nrow(churn.df) , 0.8*nrow(churn.df))
train_df <- churn.df[split_val , ]
test_df <- churn.df[-split_val , ]
# print(nrow(train_df))
# print(nrow(test_df))

#Logistic Regression

log_regression_model <- train(Churn ~ SeniorCitizen + Partner + Dependents + tenure  +MultipleLines + InternetService + OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport + StreamingTV + StreamingMovies + Contract + PaperlessBilling + PaymentMethod + MonthlyCharges, data=train_df, method='glmnet', preProcess = c("center", "scale") )

log_predict_test <- predict(log_regression_model , test_df)

confusionMatrix(log_predict_test, test_df$Churn, positive = "Yes", mode = "everything")

coef(log_regression_model$finalModel,log_regression_model$bestTune$lambda)

#Random Forest

rf_model <- train(Churn ~ SeniorCitizen + Partner + Dependents + tenure  + MultipleLines + InternetService + OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport + StreamingTV + StreamingMovies + Contract + PaperlessBilling + PaymentMethod + MonthlyCharges, data=train_df, method='rf', preProcess = c("center", "scale"))

rf_predict_test <- predict(rf_model , test_df)

confusionMatrix(rf_predict_test, test_df$Churn, positive = "Yes", mode = "everything")


#Implementing K nearest neighbors
knn_model <- train(Churn~ SeniorCitizen + Partner + Dependents + tenure+ MultipleLines + InternetService + OnlineSecurity +  OnlineBackup + DeviceProtection + TechSupport +  StreamingTV + StreamingMovies + Contract +  PaperlessBilling + PaymentMethod + MonthlyCharges,data=train_df,method='knn',preProcess = c("center", "scale"))

knn_predict_test <- predict(knn_model , test_df)

confusionMatrix(knn_predict_test, test_df$Churn, positive = "Yes", mode = "everything")

#ROC curve
plot(plot.roc(test_df$Churn , as.numeric(knn_predict_test)) , print.auc = TRUE , col = 'red' , add = TRUE , print.auc.y = 0.6 )

#evaluate all the models

plot.roc(test_df$Churn , as.numeric(rf_predict_test) , print.auc = TRUE , col = 'black' , print.auc.y = 0.7)
plot.roc(test_df$Churn , as.numeric(knn_predict_test) , print.auc = TRUE , col = 'green2' , add = TRUE , print.auc.y = 0.6 )
plot.roc(test_df$Churn , as.numeric(log_predict_test) , print.auc = TRUE , col = 'blue', add = TRUE , print.auc.y = 0.5)
legend(0.5 , 0.26, legend = c("Random Forest" , "K-Nearest Neighbors" , "Logistic Regression" ) , col = c("black","green2","blue") ,lwd = 4)
