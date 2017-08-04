Sys.setenv(TENSORFLOW_PYTHON="C:/PROGRA~3/ANACON~1/python.exe")
Sys.setenv(TENSORFLOW_PYTHON_VERSION = 3)
library(tensorflow)
require(imager)
require(caret)

# Load mnist dataset from tensorflow library
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)


# Function to plot MNIST dataset
plot_mnist<-function(imageD, pixel.y=16){
  require(imager)
  actImage<-matrix(imageD, ncol=pixel.y, byrow=FALSE)
  img.col.mat <- imappend(list(as.cimg(actImage)), "c")
  plot(img.col.mat, axes=F)
}

# Reduce Image Size
reduceImage<-function(actds, n.pixel.x=16, n.pixel.y=16){
  actImage<-matrix(actds, ncol=28, byrow=FALSE)
  img.col.mat <- imappend(list(as.cimg(actImage)),"c")
  thmb <- resize(img.col.mat, n.pixel.x, n.pixel.y)
  outputImage<-matrix(thmb[,,1,1], nrow = 1, byrow = F)
  return(outputImage)
}

# Covert train data to 16 x 16  pixel image
trainData<-t(apply(mnist$train$images, 1, FUN=reduceImage))
validData<-t(apply(mnist$test$images, 1, FUN=reduceImage))
labels <- mnist$train$labels
labels_valid <- mnist$test$labels
rm(mnist)


# Reset the graph and set-up a interactive session
tf$reset_default_graph()
sess<-tf$InteractiveSession()

# Define Model parameter
n_input<-16
step_size<-16
n.hidden<-64
n.class<-10

# Define training parameter
lr<-0.01
batch<-500
iteration = 100

# Set up a most basic RNN
rnn<-function(x, weight, bias){
  # Unstack input into step_size
  x = tf$unstack(x, step_size, 1)
  
  # Define a most basic RNN 
  rnn_cell = tf$contrib$rnn$BasicRNNCell(n.hidden)
  
  # create a recurrent neural network
  cell_output = tf$contrib$rnn$static_rnn(rnn_cell, x, dtype=tf$float32)
  
  # Linear activation, using rnn inner loop 
  last_vec=tail(cell_output[[1]], n=1)[[1]]
  return(tf$matmul(last_vec, weights) + bias)
}

# Function to evaluate mean accuracy
eval_acc<-function(yhat, y){
  # Count correct solution
  correct_Count = tf$equal(tf$argmax(yhat,1L), tf$argmax(y,1L))
  
  # Mean accuracy
  mean_accuracy = tf$reduce_mean(tf$cast(correct_Count, tf$float32))
  
  return(mean_accuracy)
}

with(tf$name_scope('input'), {
# Define placeholder for input data
x = tf$placeholder(tf$float32, shape=shape(NULL, step_size, n_input), name='x')
y <- tf$placeholder(tf$float32, shape(NULL, n.class), name='y')

# Define Weights and bias
weights <- tf$Variable(tf$random_normal(shape(n.hidden, n.class)))
bias <- tf$Variable(tf$random_normal(shape(n.class)))
})

# Evaluate rnn cell output
yhat = rnn(x, weights, bias)

# Define loss and optimizer
cost = tf$reduce_mean(tf$nn$softmax_cross_entropy_with_logits(logits=yhat, labels=y))
optimizer = tf$train$AdamOptimizer(learning_rate=lr)$minimize(cost)



# Run optimization
sess$run(tf$global_variables_initializer())

# Running optimization
for(i in 1:iteration){
  spls <- sample(1:dim(trainData)[1],batch)
  sample_data<-trainData[spls,]
  sample_y<-labels[spls,]
  
  # Reshape sample into 16 sequence with each of 16 element
  sample_data=tf$reshape(sample_data, shape(batch, step_size, n_input))
  out<-optimizer$run(feed_dict = dict(x=sample_data$eval(), y=sample_y))
  
  if (i %% 1 == 0){
    cat("iteration - ", i, "Training Loss - ",  cost$eval(feed_dict = dict(x=sample_data$eval(), y=sample_y)), "\n")
  }
}


# Calculate accuracy for 128 mnist test images
accuracy<-eval_acc(yhat, y)
valid_data=tf$reshape(validData, shape(-1, step_size, n_input))
yhat<-sess$run(tf$argmax(yhat, 1L), feed_dict = dict(x = valid_data$eval()))
image(t(matrix(validData[20,], ncol = 16, nrow = 16, byrow = T)), col  = gray((0:32)/32))
image(t(matrix(trainData[20,], ncol = 16, nrow = 16, byrow = T)), col  = gray((0:32)/32))

cost$eval(feed_dict=dict(x=valid_data$eval(), y=labels_valid))
