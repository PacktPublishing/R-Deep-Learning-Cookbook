# Function to load occupancy dataset
load_occupancy_data<-function(train){
  xFeatures = c("Temperature", "Humidity", "Light", "CO2", "HumidityRatio")
  yFeatures = "Occupancy"
  if(train){
    occupancy_ds <- as.matrix(read.csv("datatraining.txt",stringsAsFactors = T))
  } else
  {
    occupancy_ds <- as.matrix(read.csv("datatest.txt",stringsAsFactors = T))
  }
  occupancy_ds<-apply(occupancy_ds[, c(xFeatures, yFeatures)], 2, FUN=as.numeric) 
  return(list("data"=occupancy_ds, "xFeatures"=xFeatures, "yFeatures"=yFeatures))
}

# Function for min-max normalization
minmax.normalize<-function(ds, scaler=NULL){
  if(is.null(scaler)){
    for(f in ds$xFeatures){
      scaler[[f]]$minval<-min(ds$data[,f])
      scaler[[f]]$maxval<-max(ds$data[,f])
      ds$data[,f]<-(ds$data[,f]-scaler[[f]]$minval)/(scaler[[f]]$maxval-scaler[[f]]$minval)
    }
    ds$scaler<-scaler
  } else
  {
    for(f in ds$xFeatures){
      ds$data[,f]<-(ds$data[,f]-scaler[[f]]$minval)/(scaler[[f]]$maxval-scaler[[f]]$minval)
    }
  }
  return(ds)
}


# Load packages and tensorflow for Python3
Sys.setenv(TENSORFLOW_PYTHON="C:/PROGRA~3/ANACON~1/python.exe")
Sys.setenv(TENSORFLOW_PYTHON_VERSION = 3)
library(tensorflow)
np <- import("numpy")

# Load dataset
occupancy_train <-load_occupancy_data(train=T)
occupancy_test <- load_occupancy_data(train = F)

# Data normalization and plotting
require(GGally)
ggpairs(occupancy_train$data[, occupancy_train$xFeatures]) # pair plot of dataset
occupancy_train<-minmax.normalize(occupancy_train, scaler = NULL) # Min-Max Normalization
ggpairs(occupancy_train$data[, occupancy_train$xFeatures]) # pair plot of normalized dataset

# Reset the graph and set-up a interactive session
tf$reset_default_graph()
sess<-tf$InteractiveSession()

# Network Parameters
n_hidden_1 = 5 # 1st layer num features
xFeatures<-occupancy_train$xFeatures
n_input = length(xFeatures) # Number of input features
nRow<-nrow(occupancy_train$data)


# Define input variable
x <- tf$constant(unlist(occupancy_train$data[, xFeatures]), shape=c(nRow, n_input), dtype=np$float32) 
hiddenLayerEncoder<-tf$Variable(tf$random_normal(shape(n_input, n_hidden_1)), dtype=np$float32)
biasEncoder <- tf$Variable(tf$zeros(shape(n_hidden_1)), dtype=np$float32)
hiddenLayerDecoder<-tf$Variable(tf$random_normal(shape(n_hidden_1, n_input)))
biasDecoder <- tf$Variable(tf$zeros(shape(n_input)))

# Function to compute output based on Weights and Bias 
auto_encoder<-function(x, hiddenLayerEncoder, biasEncoder){
  x_transform <- tf$nn$sigmoid(tf$add(tf$matmul(x, hiddenLayerEncoder), biasEncoder))
  x_transform
}

# Encoder object
encoder_obj = auto_encoder(x,hiddenLayerEncoder, biasEncoder)
y_pred = auto_encoder(encoder_obj, hiddenLayerDecoder, biasDecoder)


# Define loss and optimizer, minimize the squared error
learning_rate = 0.01
cost = tf$reduce_mean(tf$pow(x - y_pred, 2))
optimizer = tf$train$RMSPropOptimizer(learning_rate)$minimize(cost)


# Test set-up
occupancy_test<-minmax.normalize(occupancy_test, scaler = occupancy_train$scaler)
xt <- tf$constant(unlist(occupancy_test$data[, xFeatures]), shape=c(nrow(occupancy_test$data), n_input), dtype=np$float32) 
encoder_obj_t = auto_encoder(xt,hiddenLayerEncoder, biasEncoder)
y_predt = auto_encoder(encoder_obj_t, hiddenLayerDecoder, biasDecoder)
costt = tf$reduce_mean(tf$pow(xt - y_predt, 2))


# Initializing the variables
init = tf$global_variables_initializer()
sess$run(init)

# Optimization
costconvergence<-NULL
for (step in 1:1000) {
  sess$run(optimizer)
  if (step %% 20==0){
    costconvergence<-rbind(costconvergence, c(step, sess$run(cost), sess$run(costt)))
    cat(step, "-", "Traing Cost ==>", sess$run(cost), "\n")
  }
}

# Plotting convergence
costconvergence<-data.frame(costconvergence)
colnames(costconvergence)<-c("iter", "train", "test")
plot(costconvergence[, "iter"], costconvergence[, "train"], type = "l", col="blue", xlab = "Iteration", ylab = "MSE")
lines(costconvergence[, "iter"], costconvergence[, "test"], col="red")
legend(500,0.12, c("Train","Test"), lty=c(1,1), lwd=c(2.5,2.5),col=c("blue","red"))

# writing log events for tensorboard
tf$summary$FileWriter('C:/log', tf$get_default_graph())$close()

s