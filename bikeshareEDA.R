install.packages("DataExplorer")
library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(DataExplorer)
# setwd("C:\\Users\\madel\\OneDrive\\Documents\\Stat 348\\BikeShare")


# read in the data
bike_train <- vroom("train.csv") %>%
  select(-casual, -registered)

bike_test <- vroom("test.csv")



# EDA
dplyr::glimpse(bike_train) 

DataExplorer::plot_intro(bike_train)

DataExplorer::plot_correlation(bike_train)

DataExplorer::plot_histogram(bike_train)


plot1 <- ggplot(data=bike_train, aes(x=registered, y=count)) +
geom_point() +
geom_smooth(se=FALSE)
plot1

plot2 <- ggplot(data=bike_train, aes(x=windspeed, y=count)) +
  geom_point() +
  geom_smooth(se=FALSE)
plot2

plot3 <- ggplot(bike_train, aes(x = weather)) +
  geom_bar()
plot3

plot4 <- ggplot(bike_train, aes(x = humidity)) +
  geom_bar()
plot4


(plot1 + plot2) / (plot3 + plot4)



# Linear Regression

## Setup and Fit the Linear Regression Model
my_linear_model <- linear_reg() %>% #Type of model
  set_engine("lm") %>% # Engine = What R function to use
  set_mode("regression") %>% # Regression just means quantitative response
  fit(formula=count~.-datetime, data=bike_train)

## Generate Predictions Using Linear Model
bike_predictions <- predict(my_linear_model,
                            new_data=bike_test) # Use fit to predict
bike_predictions ## Look at the output


# Format the Predictions for Submission to Kaggle
kaggle_submission <- bike_predictions %>%
bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")




