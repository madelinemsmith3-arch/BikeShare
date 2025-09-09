install.packages("DataExplorer")
library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(DataExplorer)


bike_train <- vroom("train.csv")

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


