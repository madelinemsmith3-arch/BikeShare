install.packages("DataExplorer")
install.packages("glmnet")

library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(DataExplorer)
library(dplyr)
library(glmnet)
# setwd("C:\\Users\\madel\\OneDrive\\Documents\\Stat 348\\BikeShare")


# read in the data
bike_train <- vroom("train.csv")

bike_train <- vroom("train.csv")%>%
  select(-casual, -registered) %>%
  mutate(count = log(count))


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


# Cleaning
my_recipe <- recipe(count ~ ., data = bike_train) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_mutate(weather = factor(weather)) %>%
  step_mutate(season = factor(season)) %>%
  step_rm(datetime) %>%   # <--- REMOVE datetime
  step_corr(all_numeric_predictors(), threshold = 0.5) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())
prepped_recipe <- prep(my_recipe)
baked_train <- bake(prepped_recipe, new_data=bike_train)

head(baked_train, 5)


# Penalized Regression Model
## Penalized regression model9
preg_model <- linear_reg(penalty=5, mixture=0.25) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R11

preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model) %>%
  fit(data=bike_train)
lin_preds <- predict(preg_wf, new_data=bike_test)
lin_preds$count_pred <- exp(lin_preds$.pred)

# Kaggle submission
kaggle_submission <- lin_preds %>%
  bind_cols(., bike_test) %>% 
  select(datetime, count_pred) %>% 
  rename(count = count_pred) %>% 
  mutate(count=pmax(0, count)) %>% 
  mutate(datetime=as.character(format(datetime))) 

## Write out the file
vroom_write(x=kaggle_submission, file="./LinearPreds7.csv", delim=",")

# 3: p=0.001, v=0.5
# 4: p=0.01, v=0.2
# 5: p=0.001, v=0.8
# 6: p=0.001, v=0.1
# 7: p=5, v=0.25



# Linear Regression

## Setup and Fit the Linear Regression Model
my_linear_model <- linear_reg() %>% #Type of model
  set_engine("lm") %>% # Engine = What R function to use
  set_mode("regression") # Regression just means quantitative response

# Combine into a workflow and fit
bike_workflow <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(my_linear_model) |>
  fit(data=bike_train)

lin_preds <- predict(bike_workflow, new_data = bike_test)
lin_preds$count_pred <- exp(lin_preds$.pred)

# ## Generate Predictions Using Linear Model
# bike_predictions <- predict(my_linear_model,
#                             new_data=bike_test) # Use fit to predict
# bike_predictions ## Look at the output

# Format the Predictions for Submission to Kaggle
kaggle_submission <- lin_preds %>%
bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, count_pred) %>% #Just keep datetime and prediction variables
  rename(count = count_pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file="./LinearPreds2.csv", delim=",")




