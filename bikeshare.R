install.packages("DataExplorer")
install.packages("glmnet")
install.packages("rpart")
install.packages("ranger")
install.packages("bonsai")
install.packages("lightgbm")
install.packages("dbarts")
install.packages("agua")

library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(DataExplorer)
library(dplyr)
library(glmnet)
library(rpart)
library(ranger)
library(bonsai)
library(lightgbm)
library(agua)
# setwd("C:\\Users\\madel\\OneDrive\\Documents\\Stat 348\\BikeShare")

# read in the data
bike_train <- vroom("train.csv") %>%
  select(-casual, -registered) %>%
  mutate(count = log1p(count))   # <-- use log1p instead of log


bike_test <- vroom("test.csv")

## Initialize an h2o session
h2o::h2o.init()

my_recipe <- recipe(count ~ ., data = bike_train) %>%
  
  # --- 1. Extract datetime features ---
  step_mutate(
    hour       = hour(datetime),
    wday       = wday(datetime, week_start = 1),
    month      = month(datetime),
    year       = year(datetime),
    is_weekend = ifelse(wday >= 6, 1, 0)
  ) %>%
  
  # --- 3. Cyclical encodings for hour, wday, month ---
  step_mutate(
    hour_sin   = sin(2 * pi * hour / 24),
    hour_cos   = cos(2 * pi * hour / 24),
    wday_sin   = sin(2 * pi * wday / 7),
    wday_cos   = cos(2 * pi * wday / 7),
    month_sin  = sin(2 * pi * month / 12),
    month_cos  = cos(2 * pi * month / 12)
  ) %>%
  
  # --- 7. Convert categorical variables to factors ---
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = factor(weather)) %>%
  
  # --- 8. Drop unneeded columns ---
  step_rm(datetime, hour, wday, month) %>%
  
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())


prepped_recipe <- prep(my_recipe)
baked_train <- bake(prepped_recipe, new_data = bike_train)

vroom_write(x = baked_train, file = ".BakedTrain1.csv", delim = ",")

baked_test <- bake(prepped_recipe, new_data = bike_test)
vroom_write(x = baked_test, file = "./BakedTest.csv", delim = ",")


datarobot_preds <- vroom("data_robot_preds.csv")

kaggle_submission <- bike_test %>%
  select(datetime) %>%
  bind_cols(datarobot_preds %>% select(count_PREDICTION)) %>%
  rename(count = count_PREDICTION) %>%
  mutate(
    # Backtransform from log1p
    count = pmax(0, expm1(count)),   # <-- backtransform here and prevent negatives
    datetime = format(datetime, "%Y-%m-%d %H:%M:%S")  # force exact format
  )

# Write out with vroom
vroom_write(kaggle_submission, file = "datarobot.csv", delim = ",")


##############################################################################
## enhanced feature engineering
my_recipe <- recipe(count ~ ., data = bike_train) %>%
  
  # --- 1. Extract datetime features ---
  step_mutate(
    hour       = hour(datetime),
    wday       = wday(datetime, week_start = 1),
    month      = month(datetime),
    year       = year(datetime),
    is_weekend = ifelse(wday >= 6, 1, 0)
  ) %>%
  
  # --- 2. Derived features / interactions ---
  step_mutate(
    hour_weekend = hour * is_weekend,             # interaction: hour × weekend
    is_summer    = ifelse(month %in% c(6,7,8), 1, 0),
    year_numeric = year - min(year)              # keep year as numeric
  ) %>%
  
  # --- 3. Cyclical encodings for hour, wday, month ---
  step_mutate(
    hour_sin   = sin(2 * pi * hour / 24),
    hour_cos   = cos(2 * pi * hour / 24),
    wday_sin   = sin(2 * pi * wday / 7),
    wday_cos   = cos(2 * pi * wday / 7),
    month_sin  = sin(2 * pi * month / 12),
    month_cos  = cos(2 * pi * month / 12)
  ) %>%
  
  # --- 4. Linear copies of cyclical features for model ---
  step_mutate(
    hour_lin  = hour,
    wday_lin  = wday,
    month_lin = month
  ) %>%
  
  # --- 5. Fix weather ---
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  
  # --- 6. Numeric feature interactions ---
  step_mutate(
    temp2      = temp^2,
    wind_temp  = windspeed * temp,
    hum_temp   = humidity * temp
  ) %>%
  
  # --- 7. Convert categorical variables to factors ---
  step_mutate(
    weather = factor(weather),
    season  = factor(season)
  ) %>%
  
  # --- 8. Drop unneeded columns ---
  step_rm(datetime, hour, wday, month, year) %>%
  
  # --- 9. Normalize numeric predictors ---
  step_normalize(all_numeric_predictors()) %>%
  
  # --- 10. One-hot encode categorical variables ---
  step_dummy(all_nominal_predictors()) %>%
  
  # --- 11. Remove zero-variance predictors ---
  step_zv(all_predictors())

##########################################################################

#stacking


## Define the model
auto_model <- auto_ml() %>%
  set_engine("h2o", max_runtime_secs=600, max_models=20) %>%
  set_mode("regression")

## Combine into Workflow
automl_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(auto_model) %>%
  fit(data = bike_train)

## Predict
final_preds <- predict(automl_wf, new_data = bike_test)
final_preds$count_pred <- expm1(final_preds$.pred)   # <-- use expm1 instead of exp


# Kaggle submission
kaggle_submission <- final_preds %>%
  bind_cols(., bike_test) %>%
  select(datetime, count_pred) %>%
  rename(count = count_pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

## Write out the file
vroom_write(x = kaggle_submission, file = "./LinearPreds_new.csv", delim = ",")



## make hour a factor!!!, step time pull out hour and mak eit a variable





## Create a workflow with model & recipe
my_recipe <- recipe(count ~ ., data = bike_train) %>%
  # Fix weather 4 -> 3
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  
  # Extract time/date features
  step_time(datetime, features = "hour") %>%
  step_date(datetime, features = c("dow", "month", "year"), ordinal = TRUE) %>%
  
  # Cyclical encodings (force numeric for safety)
  step_mutate(
    dow_sin   = sin(2 * pi * as.numeric(datetime_dow) / 7),
    dow_cos   = cos(2 * pi * as.numeric(datetime_dow) / 7),
    month_sin = sin(2 * pi * as.numeric(datetime_month) / 12),
    month_cos = cos(2 * pi * as.numeric(datetime_month) / 12),
    hour_sin  = sin(2 * pi * datetime_hour / 24),
    hour_cos  = cos(2 * pi * datetime_hour / 24)
  ) %>%
  
  # Keep season/weather categorical for dummies
  step_mutate(
    weather = factor(weather),
    season  = factor(season)
  ) %>%
  
  # Drop original datetime and year
  step_rm(datetime, datetime_year) %>%
  
  # One-hot encode all categoricals
  step_dummy(all_nominal_predictors())


prepped_recipe <- prep(my_recipe)
baked_train <- bake(prepped_recipe, new_data=bike_train)

head(baked_train)


bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("regression")



preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_model)

## Set up grid of tuning values
grid_of_tuning_params <- grid_regular(trees(),
                                      levels = 5) ## L^2 total tuning possibilities
## Set up K-fold CV
folds <- vfold_cv(bike_train, v = 5, repeats=1)


CV_results <- preg_wf %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse, mae)) #Or leave metrics NULL

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric="rmse")

## Finalize workflow and predict
final_wf <-
  preg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=bike_train)

## Predict
final_preds <- predict(final_wf, new_data = bike_test)
final_preds$count_pred <- exp(final_preds$.pred)

# Kaggle submission
kaggle_submission <- final_preds %>%
  bind_cols(., bike_test) %>%
  select(datetime, count_pred) %>%
  rename(count = count_pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

## Write out the file
vroom_write(x=kaggle_submission, file="./LinearPreds_bart.csv", delim=",")






my_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("regression")


my_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # What R function to use
  set_mode("regression")



#Cleaning
# my_recipe <- recipe(count ~ ., data = bike_train) %>%
#   step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
#   step_time(datetime, features = "hour") %>%
#   step_mutate(weather = factor(weather)) %>%
#   step_mutate(season = factor(season)) %>%
#   step_rm(datetime) %>%   # <--- REMOVE datetime
#   step_corr(all_numeric_predictors(), threshold = 0.5) %>%
#   step_dummy(all_nominal_predictors()) %>%
#   step_normalize(all_numeric_predictors())
# prepped_recipe <- prep(my_recipe)
# baked_train <- bake(prepped_recipe, new_data=bike_train)


my_recipe <- recipe(count ~ ., data = bike_train) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_date(datetime, features = c("dow", "month", "year")) %>%
  # Convert them to numeric
  step_mutate_at(c("datetime_hour", "datetime_dow", "datetime_month"),
                 fn = as.numeric) %>%
  
  # Cyclical encodings
  step_mutate(
    hour_sin  = sin(2 * pi * datetime_hour / 24),
    hour_cos  = cos(2 * pi * datetime_hour / 24),
    dow_sin   = sin(2 * pi * datetime_dow / 7),
    dow_cos   = cos(2 * pi * datetime_dow / 7),
    month_sin = sin(2 * pi * datetime_month / 12),
    month_cos = cos(2 * pi * datetime_month / 12)
  ) %>%

  step_mutate(weather = factor(weather),
              season = factor(season)) %>%
  step_rm(datetime, datetime_hour, datetime_dow, datetime_month, datetime_year) %>%
  step_corr(all_numeric_predictors(), threshold = 0.5) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

prepped_recipe <- prep(my_recipe)
baked_train <- bake(prepped_recipe, new_data=bike_train)

# head(baked_train, 5)

## Penalized regression model
preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

## Set Workflow
preg_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(preg_model)

## Grid of values to tune over
grid_of_tuning_params <- grid_regular(penalty(),
                                      mixture(),
                                      levels = 5) ## L^2 total tuning possibilities16

## Split data for CV1
folds <- vfold_cv(bike_train, v = 10, repeats=1)


## Run the CV
CV_results <- preg_wf %>%
tune_grid(resamples=folds,
          grid=grid_of_tuning_params,
          metrics=metric_set(rmse, mae)) #Or leave metrics NULL

## Plot Results (example)
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
geom_line()

## Find Best Tuning Parameters
bestTune <- CV_results %>%
select_best(metric="rmse")


## Finalize the Workflow & fit it
final_wf <-
preg_wf %>%
finalize_workflow(bestTune) %>%
fit(data=bike_train)

## Predict
final_preds <- predict(final_wf, new_data = bike_test)
final_preds$count_pred <- exp(final_preds$.pred)


# Kaggle submission
kaggle_submission <- final_preds %>%
  bind_cols(bike_test) %>%
  select(datetime, count_pred) %>%
  rename(count = count_pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

## Write out the file
vroom_write(x=kaggle_submission, file="./LinearPreds12.csv", delim=",")


###############################################################################


# Penalized Regression Model
## Penalized regression model9
preg_model <- linear_reg(penalty=0.01, mixture=0.2) %>% #Set model and tuning
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
vroom_write(x=kaggle_submission, file="./LinearPreds8.csv", delim=",")

# 3: p=0.001, v=0.5
# 4: p=0.01, v=0.2
# 5: p=0.001, v=0.8
# 6: p=0.001, v=0.1
# 7: p=5, v=0.25
# try p=0, v = 0.01

###############################################################################

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




