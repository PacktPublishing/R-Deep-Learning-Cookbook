######  DEEP BELIEF NETWORKS
# Import tenforflow libraries
# Sys.setenv(TENSORFLOW_PYTHON="C:/PROGRA~1/Python35/python.exe")
# Sys.setenv(TENSORFLOW_PYTHON_VERSION = 3)
library(tensorflow)
np <- import("numpy")

# Create TensorFlow session
# Reset the graph
tf$reset_default_graph()
# Starting session as interactive session
sess <- tf$InteractiveSession()

# Input data (MNIST)
mnist <- tf$examples$tutorials$mnist$input_data$read_data_sets("MNIST-data/",one_hot=TRUE)
trainX <- mnist$train$images
trainY <- mnist$train$labels
testX <- mnist$test$images
testY <- mnist$test$labels

# Creating DBN
RBM_hidden_sizes = c(900, 500 , 300 ) 


# Function to initialize RBM
RBM <- function(input_data, 
                num_input, 
                num_output,
                epochs = 5,
                alpha = 0.1,
                batchsize=100){
  
  # Placeholder variables
  vb <- tf$placeholder(tf$float32, shape = shape(num_input))
  hb <- tf$placeholder(tf$float32, shape = shape(num_output))
  W <- tf$placeholder(tf$float32, shape = shape(num_input, num_output))
  
  # Phase 1 : Forward Pass
  X = tf$placeholder(tf$float32, shape=shape(NULL, num_input))
  prob_h0= tf$nn$sigmoid(tf$matmul(X, W) + hb)  #probabilities of the hidden units
  h0 = tf$nn$relu(tf$sign(prob_h0 - tf$random_uniform(tf$shape(prob_h0)))) #sample_h_given_X
  
  # Phase 2 : Backward Pass
  prob_v1 = tf$nn$sigmoid(tf$matmul(h0, tf$transpose(W)) + vb) 
  v1 = tf$nn$relu(tf$sign(prob_v1 - tf$random_uniform(tf$shape(prob_v1)))) 
  h1 = tf$nn$sigmoid(tf$matmul(v1, W) + hb)   
  
  # Calculate gradients
  w_pos_grad = tf$matmul(tf$transpose(X), h0)
  w_neg_grad = tf$matmul(tf$transpose(v1), h1)
  CD = (w_pos_grad - w_neg_grad) / tf$to_float(tf$shape(X)[0])
  update_w = W + alpha * CD
  update_vb = vb + alpha * tf$reduce_mean(X - v1)
  update_hb = hb + alpha * tf$reduce_mean(h0 - h1)
  
  # Objective function
  err = tf$reduce_mean(tf$square(X - v1))
  
  # Initialise variables
  cur_w = tf$Variable(tf$zeros(shape = shape(num_input, num_output), dtype=tf$float32))
  cur_vb = tf$Variable(tf$zeros(shape = shape(num_input), dtype=tf$float32))
  cur_hb = tf$Variable(tf$zeros(shape = shape(num_output), dtype=tf$float32))
  prv_w = tf$Variable(tf$random_normal(shape=shape(num_input, num_output), stddev=0.01, dtype=tf$float32))
  prv_vb = tf$Variable(tf$zeros(shape = shape(num_input), dtype=tf$float32))
  prv_hb = tf$Variable(tf$zeros(shape = shape(num_output), dtype=tf$float32)) 
  
  # Start tensorflow session
  sess$run(tf$global_variables_initializer())
  output <- sess$run(list(update_w, update_vb, update_hb), feed_dict = dict(X=input_data,
                                                                            W = prv_w$eval(),
                                                                            vb = prv_vb$eval(),
                                                                            hb = prv_hb$eval()))
  prv_w <- output[[1]] 
  prv_vb <- output[[2]]
  prv_hb <-  output[[3]]
  sess$run(err, feed_dict=dict(X= input_data, W= prv_w, vb= prv_vb, hb= prv_hb))
  
  errors <- list()
  weights <- list()
  u=1
  for(ep in 1:epochs){
    for(i in seq(0,(dim(input_data)[1]-batchsize),batchsize)){
      batchX <- input_data[(i+1):(i+batchsize),]
      output <- sess$run(list(update_w, update_vb, update_hb), feed_dict = dict(X=batchX,
                                                                                W = prv_w,
                                                                                vb = prv_vb,
                                                                                hb = prv_hb))
      prv_w <- output[[1]] 
      prv_vb <- output[[2]]
      prv_hb <-  output[[3]]
      if(i%%10000 == 0){
        errors[[u]] <- sess$run(err, feed_dict=dict(X= batchX, W= prv_w, vb= prv_vb, hb= prv_hb))
        weights[[u]] <- output[[1]]
        u=u+1
        cat(i , " : ")
      }
    }
    cat("epoch :", ep, " : reconstruction error : ", errors[length(errors)][[1]],"\n")
  }
  
  w <- prv_w
  vb <- prv_vb
  hb <- prv_hb
  
  # Get the output
  input_X = tf$constant(input_data)
  ph_w = tf$constant(w)
  ph_hb = tf$constant(hb)
  
  out = tf$nn$sigmoid(tf$matmul(input_X, ph_w) + ph_hb)

  sess$run(tf$global_variables_initializer())
  return(list(output_data = sess$run(out),
              error_list=errors,
              weight_list=weights,
              weight_final=w,
              bias_final=hb))
}

#Since we are training, set input as training data
inpX = trainX

#Size of inputs is the number of inputs in the training set
num_input = ncol(inpX)

#Train RBM
RBM_output <- list()
for(i in 1:length(RBM_hidden_sizes)){
  size <- RBM_hidden_sizes[i]
  
  # Train the RBM
  RBM_output[[i]] <- RBM(input_data=inpX, 
                         num_input=num_input, 
                         num_output=size,
                         epochs = 5,
                         alpha = 0.1,
                         batchsize=100)
  
  # Update the input data
  inpX <- RBM_output[[i]]$output_data
   
  
  # Update the input_size
  num_input = size
  
  cat("completed size :", size,"\n")
}

# Plot reconstruction error
error_df <- data.frame("error"=c(unlist(RBM_output[[1]]$error_list),unlist(RBM_output[[2]]$error_list),unlist(RBM_output[[3]]$error_list)),
                       "batches"=c(rep(seq(1:length(unlist(RBM_output[[1]]$error_list))),times=3)),
                       "hidden_layer"=c(rep(c(1,2,3),each=length(unlist(RBM_output[[1]]$error_list)))),
                       stringsAsFactors = FALSE)

plot(error ~ batches,
     xlab = "# of batches",
     ylab = "Reconstruction Error",
     pch = c(1, 7, 16)[hidden_layer], 
     main = "Stacked RBM-Reconstruction MSE plot",
     data = error_df)

legend('topright',
       c("H1_900","H2_500","H3_300"), 
       pch = c(1, 7, 16))


