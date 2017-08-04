
###########################   LOGISTIC REGRESSION   #################################

##   H2o GLM ##
## Load packages
library(h2o)
library(caret)
library(pROC)

# Load the occupancy data 
occupancy_train <- read.csv("/occupation_detection/datatraining.txt",stringsAsFactors = T)
occupancy_test <- read.csv("/occupation_detection/datatest.txt",stringsAsFactors = T)

# Define input (x) and output (y) variables"
x = c("Temperature", "Humidity", "Light", "CO2", "HumidityRatio")
y = "Occupancy"

# Convert the outcome variable into factor
occupancy_train$Occupancy <- as.factor(occupancy_train$Occupancy)
occupancy_test$Occupancy <- as.factor(occupancy_test$Occupancy)

occupancy_train.hex <- as.h2o(x = occupancy_train, destination_frame = "occupancy_train.hex")
occupancy_test.hex <- as.h2o(x = occupancy_test, destination_frame = "occupancy_test.hex")


# Train the GLM model
occupancy_train.glm <- h2o.glm(x = x,  # Vector of predictor variable names
                             y = y,    # Name of response/dependent variable
                             training_frame = occupancy_train.hex, # Training data set
                             seed = 1234567,     # Seed for random numbers
                             family = "binomial",  
                             lambda_search = TRUE,
                             alpha = 0.5,
                             nfolds = 5    # N-fold cross validation
                             )

# Training accuracy (AUC)
occupancy_train.glm@model$training_metrics@metrics$AUC

# Cross validation accuracy (AUC)
occupancy_train.glm@model$cross_validation_metrics@metrics$AUC

# Predict on test data
yhat <- h2o.predict(occupancy_train.glm, occupancy_test.hex)

# Check the accuracy on test data using confusion matrix
table(actual = as.matrix(occupancy_test.hex$Occupancy) , predicted = as.matrix(yhat$predict))

# Test accuracy (AUC)
yhat$pmax <- pmax(yhat$p0, yhat$p1, na.rm = TRUE) 
roc_obj <- pROC::roc(c(as.matrix(occupancy_test.hex$Occupancy)), c(as.matrix(yhat$pmax)))
auc(roc_obj)

#compute variable importance and performance
h2o.varimp_plot(occupancy_train.glm,num_of_features = 5)


##############################   USING TENSORFLOW  - Logistic Regression

# Loading input and test data
xFeatures = c("Temperature", "Humidity", "Light", "CO2", "HumidityRatio")
yFeatures = "Occupancy"
occupancy_train <-read.csv("C:/occupation_detection/datatraining.txt",stringsAsFactors = T)
occupancy_test <- read.csv("C:/occupation_detection/datatest.txt",stringsAsFactors = T)

# subset features for modeling and transform to numeric values
occupancy_train<-apply(occupancy_train[, c(xFeatures, yFeatures)], 2, FUN=as.numeric) 
occupancy_test<-apply(occupancy_test[, c(xFeatures, yFeatures)], 2, FUN=as.numeric)

# Data dimensions
nFeatures<-length(xFeatures)
nRow<-nrow(occupancy_train)

# Reset the graph
tf$reset_default_graph()
# Starting session as interactive session
sess<-tf$InteractiveSession()

# Setting-up Logistic regression graph
x <- tf$constant(unlist(occupancy_train[, xFeatures]), shape=c(nRow, nFeatures), dtype=np$float32) # 
W <- tf$Variable(tf$random_uniform(shape(nFeatures, 1L)))
b <- tf$Variable(tf$zeros(shape(1L)))
y <- tf$matmul(x, W) + b

# Setting-up cost function and optimizer
y_ <- tf$constant(unlist(occupancy_train[, yFeatures]), dtype="float32", shape=c(nRow, 1L))
cross_entropy<-tf$reduce_mean(tf$nn$sigmoid_cross_entropy_with_logits(labels=y_, logits=y, name="cross_entropy"))
optimizer <- tf$train$GradientDescentOptimizer(0.15)$minimize(cross_entropy)

# Start a session
init <- tf$global_variables_initializer()
sess$run(init)

# Running optimization 
for (step in 1:5000) {
  sess$run(optimizer)
  if (step %% 20== 0)
    cat(step, "-", sess$run(W), sess$run(b), "==>", sess$run(cross_entropy), "\n")
}

# Performance on Train
library(pROC) 
ypred <- sess$run(tf$nn$sigmoid(tf$matmul(x, W) + b))
roc_obj <- roc(occupancy_train[, yFeatures], as.numeric(ypred))


# Performance on test
nRowt<-nrow(occupancy_test)
xt <- tf$constant(unlist(occupancy_test[, xFeatures]), shape=c(nRowt, nFeatures), dtype=np$float32) #
ypredt <- sess$run(tf$nn$sigmoid(tf$matmul(xt, W) + b))
roc_objt <- roc(occupancy_test[, yFeatures], as.numeric(ypredt))

plot.roc(roc_obj, col = "green", lty=2, lwd=2)
plot.roc(roc_objt, add=T, col="red", lty=4, lwd=2)


#################   Tensorboard   ####################
# Create Writer Obj for log
log_writer = tf$summary$FileWriter('c:/log', sess$graph)

# Adding histogram summary to weight and bias variable
w_hist = tf$histogram_summary("weights", W)
b_hist = tf$histogram_summary("biases", b)

# Set-up cross entropy for test
nRowt<-nrow(occupancy_test)
xt <- tf$constant(unlist(occupancy_test[, xFeatures]), shape=c(nRowt, nFeatures), dtype=np$float32)
ypredt <- tf$nn$sigmoid(tf$matmul(xt, W) + b)
yt_ <- tf$constant(unlist(occupancy_test[, yFeatures]), dtype="float32", shape=c(nRowt, 1L))
cross_entropy_tst<-tf$reduce_mean(tf$nn$sigmoid_cross_entropy_with_logits(labels=yt_, logits=ypredt, name="cross_entropy_tst"))

# Add summary ops to collect data
w_hist = tf$summary$histogram("weights", W)
b_hist = tf$summary$histogram("biases", b)
crossEntropySummary<-tf$summary$scalar("costFunction", cross_entropy)
crossEntropyTstSummary<-tf$summary$scalar("costFunction_test", cross_entropy_tst)

# Create Writer Obj for log
log_writer = tf$summary$FileWriter('c:/log', sess$graph)

for (step in 1:2500) {
  sess$run(optimizer)
  
  # Evaluate performance on training and test data after 50 Iteration
  if (step %% 50== 0){
    ### Performance on Train
    ypred <- sess$run(tf$nn$sigmoid(tf$matmul(x, W) + b))
    roc_obj <- roc(occupancy_train[, yFeatures], as.numeric(ypred))
    
    ### Performance on Test
    ypredt <- sess$run(tf$nn$sigmoid(tf$matmul(xt, W) + b))
    roc_objt <- roc(occupancy_test[, yFeatures], as.numeric(ypredt))
    cat("train AUC: ", auc(roc_obj), " Test AUC: ", auc(roc_objt), "n")
    
    # Save summary of Bias and weights
    log_writer$add_summary(sess$run(b_hist), global_step=step)
    log_writer$add_summary(sess$run(w_hist), global_step=step)
    log_writer$add_summary(sess$run(crossEntropySummary), global_step=step)
    log_writer$add_summary(sess$run(crossEntropyTstSummary), global_step=step)
  } }

summary = tf$summary$merge_all() 

log_writer = tf$summary$FileWriter('c:/log', sess$graph)
summary_str = sess$run(summary)
log_writer$add_summary(summary_str, step)
log_writer$close()

###########################   NEURAL NETWORKS   #################################

############################## USING H2O package

library(h2o)
library(caret)
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE,min_mem_size = "20G",nthreads = 8)

# Load the occupancy data 
occupancy_train <- read.csv("/occupation_detection/datatraining.txt",stringsAsFactors = T)
occupancy_test <- read.csv("/occupation_detection/datatest.txt",stringsAsFactors = T)

# Define input (x) and output (y) variables"
x = c("Temperature", "Humidity", "Light", "CO2", "HumidityRatio")
y = "Occupancy"

# Convert the outcome variable into factor
occupancy_train$Occupancy <- as.factor(occupancy_train$Occupancy)
occupancy_test$Occupancy <- as.factor(occupancy_test$Occupancy)

occupancy_train.hex <- as.h2o(x = occupancy_train, destination_frame = "occupancy_train.hex")
occupancy_test.hex <- as.h2o(x = occupancy_test, destination_frame = "occupancy_test.hex")


# Train the model
occupancy.deepmodel <- h2o.deeplearning(x = x,	 	 
                              y = y, 	 
                              training_frame = occupancy_train.hex,	 	 
                              validation_frame = occupancy_test.hex,	 	 
                              standardize = F,	 	 
                              activation = "Rectifier",	 	 
                              epochs = 50,	 	 
                              seed = 1234567,	 	 
                              hidden = 5,	 	 
                              variable_importances = T,
                              nfolds = 5,
                              adpative_rate = TRUE)	

# Get the validation accuracy (AUC)
xval_performance <- h2o.performance(occupancy.deepmodel,xval = T)
xval_performance@metrics$AUC

# Get the training accuracy (AUC)
train_performance <- h2o.performance(occupancy.deepmodel,train = T)
train_performance@metrics$AUC

# Get the testing accuracy(AUC)
test_performance <- h2o.performance(occupancy.deepmodel,valid = T)
test_performance@metrics$AUC

# Perform hyper parameter tuning
activation_opt <- c("Rectifier","RectifierWithDropout", "Maxout","MaxoutWithDropout")
hidden_opt <- list(5, c(5,5))
epoch_opt <- c(10,50,100)
l1_opt <- c(0,1e-3,1e-4)
l2_opt <- c(0,1e-3,1e-4)

hyper_params <- list( activation = activation_opt,
                      hidden = hidden_opt,
                      epochs = epoch_opt,
                      l1 = l1_opt,
                      l2 = l2_opt
                      )

#set search criteria
search_criteria <- list(strategy = "RandomDiscrete", max_models=300)

# Perform grid search on training data
dl_grid <- h2o.grid(x = x,
                    y = y,
                    algorithm = "deeplearning",
                    grid_id = "deep_learn",
                    hyper_params = hyper_params,
                    search_criteria = search_criteria,
                    training_frame = occupancy_train.hex,
                    nfolds = 5)
                   

#get best model based on auc
d_grid <- h2o.getGrid("deep_learn",sort_by = "auc", decreasing = T)
best_dl_model <- h2o.getModel(d_grid@model_ids[[1]])

# Performance of Cross validation after grid search
xval_performance.grid <- h2o.performance(best_dl_model,xval = T)
xval_performance.grid@metrics$AUC

# Performance of Training after grid search
train_performance.grid <- h2o.performance(best_dl_model,train = T)
train_performance.grid@metrics$AUC

# Performance of Model on Test data after grid search
yhat <- h2o.predict(best_dl_model, occupancy_test.hex)

# Check the accuracy on test data using confusion matrix
table(actual = as.matrix(occupancy_test.hex$Occupancy) , predicted = as.matrix(yhat$predict))

# Test accuracy (AUC)
yhat$pmax <- pmax(yhat$p0, yhat$p1, na.rm = TRUE) 
roc_obj <- pROC::roc(c(as.matrix(occupancy_test.hex$Occupancy)), c(as.matrix(yhat$pmax)))
pROC::auc(roc_obj)



######################## USING MXNET package (neural Network)
library(mxnet)

# load the data
# Load the occupancy data 
occupancy_train <- read.csv("/occupation_detection/datatraining.txt",stringsAsFactors = T)
occupancy_test <- read.csv("/occupation_detection/datatest.txt",stringsAsFactors = T)

# Define input (x) and output (y) variables"
x = c("Temperature", "Humidity", "Light", "CO2", "HumidityRatio")
y = "Occupancy"

# convert the train data into matrix
occupancy_train.x <- data.matrix(occupancy_train[,x])
occupancy_train.y <- occupancy_train$Occupancy

# convert the test data into matrix
occupancy_test.x <- data.matrix(occupancy_test[,x])
occupancy_test.y <- occupancy_test$Occupancy

# Train a multilayered perceptron

#set seed to reproduce results
mx.set.seed(1234567)

# create NN structure
smb.data <- mx.symbol.Variable("data")
smb.fc <- mx.symbol.FullyConnected(smb.data, num_hidden=5) 
smb.soft <- mx.symbol.SoftmaxOutput(smb.fc)

# Train the network
nnmodel <- mx.model.FeedForward.create(symbol = smb.soft,
                                       X = occupancy_train.x,
                                       y = occupancy_train.y,
                                       ctx = mx.cpu(),
                                       num.round = 100,
                                       eval.metric = mx.metric.accuracy,
                                       array.batch.size = 100,
                                       learning.rate = 0.01)

# Train accuracy (AUC)
train_pred <- predict(nnmodel,occupancy_train.x)
train_yhat <- max.col(t(train_pred))-1
roc_obj <- pROC::roc(c(occupancy_train.y), c(train_yhat))
pROC::auc(roc_obj)

# Predict on the test data
test_pred <- predict(nnmodel,occupancy_test.x)
test_yhat <- max.col(t(test_pred))-1

#Test accuracy (AUC)
roc_obj <- pROC::roc(c(occupancy_test.y), c(test_yhat))
pROC::auc(roc_obj)



######################## USING TENSORFLOW (neural network)
library(tensorflow)

np <- import("numpy")
tf <- import("tensorflow")

# Loading input and test data
xFeatures = c("Temperature", "Humidity", "Light", "CO2", "HumidityRatio")
yFeatures = "Occupancy"
occupancy_train <-read.csv("/occupation_detection/datatraining.txt",stringsAsFactors = T)
occupancy_test <- read.csv("/occupation_detection/datatest.txt",stringsAsFactors = T)

# subset features for modeling and transform to numeric values
occupancy_train<-apply(occupancy_train[, c(xFeatures, yFeatures)], 2, FUN=as.numeric) 
occupancy_test<-apply(occupancy_test[, c(xFeatures, yFeatures)], 2, FUN=as.numeric)

# Data dimensions
nFeatures<-length(xFeatures)
nRow<-nrow(occupancy_train)

# Reset the graph
tf$reset_default_graph()
# Starting session as interactive session
sess<-tf$InteractiveSession()

# Network Parameters
n_hidden_1 = 5L # 1st layer number of features
n_hidden_2 = 5L # 2nd layer number of features
n_input = 5L    # 5 attributes
n_classes = 1L  # Binary class

# Model Parameters
learning_rate = 0.001
training_epochs = 10000

# Graph input
x = tf$constant(unlist(occupancy_train[,xFeatures]), shape=c(nRow, n_input), dtype=np$float32)
y = tf$constant(unlist(occupancy_train[,yFeatures]), dtype="float32", shape=c(nRow, 1L))

# Create model
multilayer_perceptron <- function(x, weights, biases){
  # Hidden layer with RELU activation
  layer_1 = tf$add(tf$matmul(x, weights[["h1"]]), biases[["b1"]])
  layer_1 = tf$nn$relu(layer_1)
  # Hidden layer with RELU activation
  layer_2 = tf$add(tf$matmul(layer_1, weights[["h2"]]), biases[["b2"]])
  layer_2 = tf$nn$relu(layer_2)
  # Output layer with linear activation
  out_layer = tf$matmul(layer_2, weights[["out"]]) + biases[["out"]]
  return(out_layer)
}

# Initialises and store hidden layer's weight & bias
weights = list(
  "h1" = tf$Variable(tf$random_normal(c(n_input, n_hidden_1))),
  "h2" = tf$Variable(tf$random_normal(c(n_hidden_1, n_hidden_2))),
  "out" = tf$Variable(tf$random_normal(c(n_hidden_2, n_classes)))
)
biases = list(
  "b1" =  tf$Variable(tf$random_normal(c(1L,n_hidden_1))),
  "b2" = tf$Variable(tf$random_normal(c(1L,n_hidden_2))),
  "out" = tf$Variable(tf$random_normal(c(1L,n_classes)))
)

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf$reduce_mean(tf$nn$sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf$train$AdamOptimizer(learning_rate=learning_rate)$minimize(cost)

# Initializing the global variables
init = tf$global_variables_initializer()
sess$run(init)

# Training cycle
for(epoch in 1:training_epochs){
    sess$run(optimizer)
  if (epoch %% 20== 0)
    cat(epoch, "-", sess$run(cost), "\n")                                   
}

# Performance on Train
library(pROC) 
ypred <- sess$run(tf$nn$sigmoid(multilayer_perceptron(x, weights, biases)))
roc_obj <- roc(occupancy_train[, yFeatures], as.numeric(ypred))

# Performance on Test
nRowt<-nrow(occupancy_test)
xt <- tf$constant(unlist(occupancy_test[, xFeatures]), shape=c(nRowt, nFeatures), dtype=np$float32) #
ypredt <- sess$run(tf$nn$sigmoid(multilayer_perceptron(xt, weights, biases)))
roc_objt <- roc(occupancy_test[, yFeatures], as.numeric(ypredt))

plot.roc(roc_obj, col = "green", lty=2, lwd=2)
plot.roc(roc_objt, add=T, col="red", lty=4, lwd=2)

  

