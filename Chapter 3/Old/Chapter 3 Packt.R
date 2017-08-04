# Load Libraries
library(tensorflow)
library(imager)

# Import numpy
np <- import("numpy")

# Create TensorFlow session
# Reset the graph
tf$reset_default_graph()
# Starting session as interactive session
sess <- tf$InteractiveSession()

# Input data (Cifar 10)
download.cifar.data <- function(data_dir) {
  dir.create(data_dir, showWarnings = FALSE)
  setwd(data_dir)
  if (!file.exists('cifar-10-binary.tar.gz')){
    download.file(url='http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz', destfile='cifar-10-binary.tar.gz', method='wget')
    untar("cifar-10-binary.tar.gz") # Unzip files
    file.remove("cifar-10-binary.tar.gz") # remove zip file
  }
  setwd("..")
}

labels <- read.table("/Cifar_10/batches.meta.txt")
num.images = 10000 

download.cifar.data(data_dir="/Cifar_10/")

# Function to read cifar data
read.cifar.data <- function(filenames,num.images=10000){
  images.rgb <- list()
  images.lab <- list()
  for (f in 1:length(filenames)) {
    to.read <- file(paste("/Cifar_10/",filenames[f], sep=""), "rb")
    for(i in 1:num.images) {
      l <- readBin(to.read, integer(), size=1, n=1, endian="big")
      r <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
      g <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
      b <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
      index <- num.images * (f-1) + i
      images.rgb[[index]] = data.frame(r, g, b)
      images.lab[[index]] = l+1
    }
    close(to.read)
    cat("completed :",  filenames[f], "\n")
    remove(l,r,g,b,f,i,index, to.read)
  }
  return(list("images.rgb"=images.rgb,"images.lab"=images.lab))
}

# Train dataset (1st to 5th data_batch_bins)
cifar_train <- read.cifar.data(filenames = c("data_batch_1.bin","data_batch_2.bin","data_batch_3.bin","data_batch_4.bin","data_batch_5.bin"))  
images.rgb.train <- cifar_train$images.rgb
images.lab.train <- cifar_train$images.lab
rm(cifar_train)

# Test dataset 
cifar_test <- read.cifar.data(filenames = c("test_batch.bin"))
images.rgb.test <- cifar_test$images.rgb
images.lab.test <- cifar_test$images.lab
rm(cifar_test)

# Function to flatten the data
flat_data <- function(x_listdata,y_listdata){
  # Flatten input x variables
  x_listdata <- lapply(x_listdata,function(x){unlist(x)})
  x_listdata <- do.call(rbind,x_listdata)
  # Flatten outcome y variables
  y_listdata <- lapply(y_listdata,function(x){a=c(rep(0,10)); a[x]=1; return(a)})
  y_listdata <- do.call(rbind,y_listdata)
  # Return flattened x and y variables
  return(list("images"=x_listdata, "labels"=y_listdata))
}

# Generate flattened train and test datasets
train_data <- flat_data(x_listdata = images.rgb.train, y_listdata = images.lab.train)
test_data <- flat_data(x_listdata = images.rgb.test, y_listdata = images.lab.test)

# function to run sanity check on photos & labels import
drawImage <- function(index, images.rgb, images.lab=NULL) {
  require(imager)
  # Testing the parsing: Convert each color layer into a matrix,
  # combine into an rgb object, and display as a plot
  img <- images.rgb[[index]]
  img.r.mat <- as.cimg(matrix(img$r, ncol=32, byrow = FALSE))
  img.g.mat <- as.cimg(matrix(img$g, ncol=32, byrow = FALSE))
  img.b.mat <- as.cimg(matrix(img$b, ncol=32, byrow = FALSE))
  img.col.mat <- imappend(list(img.r.mat,img.g.mat,img.b.mat),"c") #Bind the three channels into one image
  
  # Extract the label
  if(!is.null(images.lab)){
    lab = labels[[1]][images.lab[[index]]] 
  }
  
  # Plot and output label
  plot(img.col.mat,main=paste0(lab,":32x32 size",sep=" "),xaxt="n")
  axis(side=1, xaxp=c(10, 50, 4), las=1)
  
  return(list("Image label" =lab,"Image description" =img.col.mat))
}

# Draw a random image along with its label and description from train dataset
drawImage(sample(1:(num.images), size=1), images.rgb.train, images.lab.train)

# Configuration of Neural Network

# Convolutional Layer 1.
filter_size1 = 5L          
num_filters1 = 64L         

# Convolutional Layer 2.
filter_size2 = 5L         
num_filters2 = 64L         

# Fully-connected layer.
fc_size = 1024L             

# Data Dimensions
# CIFAR images are 32 pixels in each dimension.
img_size = 32L

# Number of colour channels for the images: 3 channel for reg, blue, green scales.
num_channels = 3L

# Images are stored in one-dimensional arrays of length.
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = c(img_size, img_size)

# Number of classes, one class for each of 10 images
num_classes = 10L

# Multilayer ConvNet
# Weight Initialization
weight_variable <- function(shape) {
  initial <- tf$truncated_normal(shape, stddev=0.1)
  tf$Variable(initial)
}

bias_variable <- function(shape) {
  initial <- tf$constant(0.1, shape=shape)
  tf$Variable(initial)
}

# Create a new convolution layer
create_conv_layer <- function(input,             
                              num_input_channels, 
                              filter_size,        
                              num_filters,        
                              use_pooling=True)   
{
  # Shape of the filter-weights for the convolution.
  shape1 = shape(filter_size, filter_size, num_input_channels, num_filters)
  
  # Create new weights 
  weights = weight_variable(shape=shape1)
  
  # Create new biases
  biases = bias_variable(shape=shape(num_filters))
  
  # Create the TensorFlow operation for convolution.
  layer = tf$nn$conv2d(input=input,
                       filter=weights,
                       strides=shape(1L, 1L, 1L ,1L),
                       padding="SAME")
  
  # Add the biases to the results of the convolution.
  layer = layer + biases
  
  # Use pooling (binary flag) to reduce the image resolution
  if(use_pooling){
    layer = tf$nn$max_pool(value=layer,
                           ksize=shape(1L, 2L, 2L, 1L),
                           strides=shape(1L, 2L, 2L, 1L), 
                           padding='SAME')
  }
  
  # Add non-linearity using Rectified Linear Unit (ReLU).
  layer = tf$nn$relu(layer)
  
  # Retrun resulting layer and updated weights
  return(list("layer" = layer, "weights" = weights))
}

# Function to flatten the layer
flatten_conv_layer <- function(layer){
  # Extract the shape of the input layer
  layer_shape = layer$get_shape()
  
  # Calculate the number of features as img_height * img_width * num_channels
  num_features = prod(c(layer_shape$as_list()[[2]],layer_shape$as_list()[[3]],layer_shape$as_list()[[4]]))
  
  # Reshape the layer to [num_images, num_features].
  layer_flat = tf$reshape(layer, shape(-1, num_features))
  
  # Return both the flattened layer and the number of features.
  return(list("layer_flat"=layer_flat, "num_features"=num_features))
}

# Create a new fully connected layer
create_fc_layer <- function(input,        
                            num_inputs,     
                            num_outputs,    
                            use_relu=True) 
{
  # Create new weights and biases.
  weights = weight_variable(shape=shape(num_inputs, num_outputs))
  biases = bias_variable(shape=shape(num_outputs))
  
  # Perform matrix multiplication of input layer with weights and then add biases
  layer = tf$matmul(input, weights) + biases
  
  # Use ReLU?
  if(use_relu){
    layer = tf$nn$relu(layer)
  }
  
  return(layer)
}

# Placeholder variables
x = tf$placeholder(tf$float32, shape=shape(NULL, img_size_flat), name='x')
x_image = tf$reshape(x, shape(-1L, img_size, img_size, num_channels))
y_true = tf$placeholder(tf$float32, shape=shape(NULL, num_classes), name='y_true')

# Convolutional Layer 1
conv1 <- create_conv_layer(input=x_image,
                           num_input_channels=num_channels,
                           filter_size=filter_size1,
                           num_filters=num_filters1,
                           use_pooling=TRUE)

layer_conv1 <- conv1$layer
weights_conv1  <- conv1$weights

# Convolutional Layer 2
conv2 <- create_conv_layer(input=layer_conv1,
                           num_input_channels=num_filters1,
                           filter_size=filter_size2,
                           num_filters=num_filters2,
                           use_pooling=TRUE)

layer_conv2 <- conv2$layer
weights_conv2 <- conv2$weights

# Flatten Layer
flatten_lay <- flatten_conv_layer(layer_conv2)
layer_flat <- flatten_lay$layer_flat
num_features <- flatten_lay$num_features

# Fully-Connected Layer 1

layer_fc1 = create_fc_layer(input=layer_flat,
                            num_inputs=num_features,
                            num_outputs=fc_size,
                            use_relu=TRUE)

# Dropout to avoid overfitting
keep_prob <- tf$placeholder(tf$float32)
layer_fc1_drop <- tf$nn$dropout(layer_fc1, keep_prob)

# Fully-Connected Layer 2

layer_fc2 = create_fc_layer(input=layer_fc1_drop,
                            num_inputs=fc_size,
                            num_outputs=num_classes,
                            use_relu=FALSE)

# Dropout to avoid overfitting
layer_fc2_drop <- tf$nn$dropout(layer_fc2, keep_prob)

# Predicted Class
y_pred = tf$nn$softmax(layer_fc2_drop)
y_pred_cls = tf$argmax(y_pred, dimension=1L)

# Cost-function to be optimized
cross_entropy = tf$nn$softmax_cross_entropy_with_logits(logits=layer_fc2_drop, labels=y_true)
cost = tf$reduce_mean(cross_entropy)

# Optimization Method
optimizer = tf$train$AdamOptimizer(learning_rate=1e-4)$minimize(cost)

# Performance Measures
y_true_cls = tf$argmax(y_true, dimension=1L)
correct_prediction = tf$equal(y_pred_cls, y_true_cls)
accuracy = tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

# Run the Tensorflow
# Initialize variables
sess$run(tf$global_variables_initializer())

# Train the model
train_batch_size = 50L
for (i in 1:100) {
  spls <- sample(1:dim(train_data$images)[1],train_batch_size)
  if (i %% 10 == 0) {
    train_accuracy <- accuracy$eval(feed_dict = dict(
      x = train_data$images[spls,], y_true = train_data$labels[spls,], keep_prob = 1.0))
    cat(sprintf("step %d, training accuracy %g\n", i, train_accuracy))
  }
  optimizer$run(feed_dict = dict(
    x = train_data$images[spls,], y_true = train_data$labels[spls,], keep_prob = 0.5))
}

# Test the model
test_accuracy <- accuracy$eval(feed_dict = dict(
  x = test_data$images, y_true = test_data$labels, keep_prob = 1.0))
cat(sprintf("test accuracy %g", test_accuracy))

# Get Actual/True labels of test data
test_pred_class <- y_pred_cls$eval(feed_dict = dict(
  x = test_data$images, y_true = test_data$labels, keep_prob = 1.0))
test_pred_class <- test_pred_class + 1
test_true_class <- c(unlist(images.lab.test))

# Confusion matrix with its plot
table(actual = test_true_class, predicted = test_pred_class)

confusion <- as.data.frame(table(actual = test_true_class, predicted = test_pred_class))
plot <- ggplot(confusion)
plot + geom_tile(aes(x=actual, y=predicted, fill=Freq)) + scale_x_discrete(name="Actual Class") + scale_y_discrete(name="Predicted Class") + scale_fill_gradient(breaks=seq(from=-0, to=10, by=1)) + labs(fill="Normalized\nFrequency")

# Plot errors
check.image <- function(images.rgb,index,true_lab, pred_lab) {
  require(imager)
  # Testing the parsing: Convert each color layer into a matrix,
  # combine into an rgb object, and display as a plot
  img <- images.rgb[[index]]
  img.r.mat <- as.cimg(matrix(img$r, ncol=32, byrow = FALSE))
  img.g.mat <- as.cimg(matrix(img$g, ncol=32, byrow = FALSE))
  img.b.mat <- as.cimg(matrix(img$b, ncol=32, byrow = FALSE))
  img.col.mat <- imappend(list(img.r.mat,img.g.mat,img.b.mat),"c") 
  
  # Plot with actual and predicted label
  plot(img.col.mat,main=paste0("True: ", true_lab,":: Pred: ", pred_lab),xaxt="n")
  axis(side=1, xaxp=c(10, 50, 4), las=1)
}

# Plot misclassified test images
labels <- c("airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck")
plot.misclass.images <- function(images.rgb, y_actual, y_predicted,labels){
  
  # Get indices of misclassified
  indices <- which(!(y_actual == y_predicted))
  id <- sample(indices,1)
  
  # plot the image with true and predicted class
  true_lab <- labels[y_actual[id]]
  pred_lab <- labels[y_predicted[id]]
  check.image(images.rgb,index=id, true_lab=true_lab,pred_lab=pred_lab)
}
plot.misclass.images(images.rgb=images.rgb.test,y_actual=test_true_class,y_predicted=test_pred_class,labels=labels)

