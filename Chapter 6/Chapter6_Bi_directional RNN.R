Sys.setenv(TENSORFLOW_PYTHON="C:/PROGRA~3/ANACON~1/python.exe")
Sys.setenv(TENSORFLOW_PYTHON_VERSION = 3)
library(tensorflow)
require(imager)
require(caret)

# Load mnist dataset from tensorflow library
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

# Covert train data to 16 x 16  pixel image
trainData<-t(apply(mnist$train$images, 1, FUN=reduceImage))
validData<-t(apply(mnist$test$images, 1, FUN=reduceImage))
labels=mnist$train$labels
labels_valid<-mnist$test$labels
rm(mnist)



# Reset the graph and set-up a interactive session
tf$reset_default_graph()
sess<-tf$InteractiveSession()

# Parameters
lr = 0.001
batch<-500
iteration = 100

# Define Model parameter
n_input<-16
step_size<-16
n.hidden<-64
n.class<-10

with(tf$name_scope('input'), {
  # Define placeholder for input data
  x = tf$placeholder(tf$float32, shape=shape(NULL, step_size, n_input), name='x')
  y <- tf$placeholder(tf$float32, shape(NULL, n.class), name='y')
  
  # Define Weights and bias
  weights <- tf$Variable(tf$random_normal(shape(2*n.hidden, n.class)))
  bias <- tf$Variable(tf$random_normal(shape(n.class)))
})


# Evaluation of Bidirection RNN cell
bidirectionRNN<-function(x, weights, bias){
  # Unstack input into step_size
  x = tf$unstack(x, step_size, 1)
  
  # Forward lstm cell
  rnn_cell_forward = tf$contrib$rnn$BasicRNNCell(n.hidden)
  
  # Backward lstm cell
  rnn_cell_backward = tf$contrib$rnn$BasicRNNCell(n.hidden)
  
  # Get lstm cell output
  cell_output = tf$contrib$rnn$static_bidirectional_rnn(rnn_cell_forward, rnn_cell_backward, x, dtype=tf$float32)
  
  # Linear activation, using rnn inner loop last output
  last_vec=tail(cell_output[[1]], n=1)[[1]]
  return(tf$matmul(last_vec, weights) + bias)
}

# Eval accuracy
eval_acc<-function(yhat, y){
  # Count correct solution
  correct_Count = tf$equal(tf$argmax(yhat,1L), tf$argmax(y,1L))
  
  # Mean accuracy
  mean_accuracy = tf$reduce_mean(tf$cast(correct_Count, tf$float32))
  
  return(mean_accuracy)
}


# Evaluate rnn cell output
yhat = bidirectionRNN(x, weights, bias)

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

valid_data=tf$reshape(validData, shape(-1, step_size, n_input))
cost$eval(feed_dict = dict(x=valid_data$eval(), y=labels_valid))
  