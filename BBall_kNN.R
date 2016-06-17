





































###-----------------------------------------------------------------------###
### Predicting the salaries of baseball players with k nearest neighbours ###
###-----------------------------------------------------------------------###

# "Introduction to statistical learning" 
#  - Fantastic book, must read for machine learning intro 
#  - get the pdf for free! http://www-bcf.usc.edu/~gareth/ISL/
library(ISLR)

# extensive and powerful plotting library
# find more here http://ggplot2.org/ and countless online tutorials
library(ggplot2)



# 1. Explore
# ----------

# we will analyse a dataset called 'Hitters', 
# lets explore it and see whats inside
?Hitters
class(Hitters)
head(Hitters)
str(Hitters)
summary(Hitters)
?class

# we want to predict the salary so lets see what its distribution is
hist(log10(Hitters$Salary), breaks = 30)
?hist
# note that salary is given in thousands of dollars, 
# you can check with ?Hitters

# a prettier plot with ggplot
ggplot(Hitters, aes(x = log10(Salary))) +
  geom_histogram()

# what variables might be correlated with the salary?
ggplot(Hitters, aes(x = Years, y = Hits)) +
  geom_point(aes(colour = log10(Salary)))



# 2. Data Munging!
# ----------------

# creating a new object, a new df that contains only the 
# predictors of interest
predictors <- data.frame(
  hits = Hitters$Hits,
  years = Hitters$Years
)
# store the response variable seperately
response <- log10(Hitters$Salary)

# remove NA's (remove rows containing missing values for salary)
to_remove <- which(is.na(response))
predictors <- predictors[-to_remove, ]
response <- response[-to_remove]

# mean standardise and scale
# this trick can be used to bring different numeric variables 
# onto comparable scales
shits <- scale(predictors$hits)
syears <- scale(predictors$years)
spredictors <- data.frame(shits, syears)

ggplot(spredictors, aes(x = syears, y = shits)) +
  geom_point(aes(colour = log10(response)), 
             size = 3, 
             data = as.data.frame(response))

# 3. Fit a Model
# --------------

# fast nearest neighbours
library(FNN)

# the default calculates 'leave-one-out' predictions
model <- knn.reg(train = spredictors, 
                 y = response, 
                 k = 10)
# rsq between real values and predicted values 
# is an indicator of performance
model

# k is arbitrarily selected so far, how do we tune K?
# lets try by measuring the rsq for predictions when we vary K.
# first we need a data.frame to store the results
N <- nrow(spredictors) - 1
tuningK <- data.frame(
  k = 1:N,
  rsq = NA
)

# now lets loop through and extract the rsq for each value of K
for(i in 1:N){ 
  tuningK$rsq[i] <- knn.reg(
    train = spredictors, 
    y = response, 
    k = i
  )$R2Pred
}

# the best value of K is the one that results in the highest value of rsq
# (possibly)
bestK <- which.max(tuningK$rsq)
bestK

# visualise the tuning of K
ggplot(tuningK, aes(x = k, y = rsq)) + 
  geom_line() +
  geom_vline(xintercept = bestK, colour = "red", lty = 3)



# 4. Evaluate model behaviour
# ---------------------------

# re-calculate LOO predictions with tuned K
model <- knn.reg(train = spredictors, 
                 y = response, 
                 k = bestK)

# collect the results
results <- data.frame(
  salary = response, 
  preds = model$pred,
  resids = model$residuals
)

# lets plot the predicted salary over the real salary, 
# the same as the correlation we have been measuring with rsq.
# but visualising allows us to see *where* the model fails
ggplot(results, aes(x = preds, y = salary)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1)

# the difference between the real values and the predicted values
# is called the residuals we can inspect the residuals
median(abs(model$residuals))
ggplot(results, aes(x = salary, y = resids)) +
  geom_point() +
  geom_hline(yintercept = 0)

# we can also examine the predicted values in theoretical space
# of the explanatory variables. this allows us to see what the model
# has learned.
# first we need some new fake variables
hits_grid <- seq(min(shits), 
                  max(shits), 
                  length.out = 50)
years_grid <- seq(min(syears), 
                 max(syears), 
                 length.out = 50)

# the very useful expand.grid() function will return a data frame
# containing every combination of its arguments
grid_data <- expand.grid(
  years = years_grid,
  hits = hits_grid
)

# now we can calculate predicted values on our theoretical grid
grid_results <- knn.reg(
  train = spredictors, 
  test = grid_data, 
  y = response, 
  k = bestK
)

# collect the results
grid_data <- cbind(grid_data, preds = grid_results$pred)

# and visualise
ggplot(grid_data, aes(x = hits, y = years)) +
  geom_raster(aes(fill = preds))

