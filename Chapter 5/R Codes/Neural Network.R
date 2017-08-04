## Neural network 
NN_train <- function(Xdata,Ydata,Xtestdata,Ytestdata,input_size,
                     learning_rate=0.1,momentum = 0.1,epochs=10,
                     batchsize=100,rbm_list,dbn_sizes){
  library(stringi)
  weight_list <- list()
  bias_list <- list()
  # Initialise variables
  for(size in c(dbn_sizes,ncol(Ydata))){
    #Initialize weights through a random uniform distribution
    weight_list <- c(weight_list,tf$random_normal(shape=shape(input_size, size), stddev=0.01, dtype=tf$float32))
    #Initialize bias as zeroes
    bias_list <- c(bias_list, tf$zeros(shape = shape(size), dtype=tf$float32))
    input_size = size
  }
  
  # Load from RBM
  #Check if expected dbn_sizes are correct
  if(length(dbn_sizes)!=length(rbm_list)){
    stop("number of hidden dbn_sizes not equal to number of rbm outputs generated")
    # check if expected sized are correct
    for(i in 1:length(dbn_sizes)){
      if(dbn_sizes[i] != dbn_sizes[i])
        stop("Number of hidden dbn_sizes do not match")
    }
  }
  
  # Bring the weights and biases
  for(i in 1:length(dbn_sizes)){
    weight_list[[i]] <- rbm_list[[i]]$weight_final
    bias_list[[i]] <- rbm_list[[i]]$bias_final
  }
  
  # Lets train the neural network
  # create placeholders for input, weights, biases, output
  input_sub <- list()
  weight <- list()
  bias <- list()
  input <- tf$placeholder(tf$float32, shape = shape(NULL,ncol(Xdata)))
  output <- tf$placeholder(tf$float32, shape = shape(NULL,ncol(Ydata)))
  
  # Define variables and activation function
  for(i in 1:(length(dbn_sizes)+1)){
    weight[[i]] <- tf$cast(tf$Variable(weight_list[[i]]),tf$float32)
    bias[[i]] <- tf$cast(tf$Variable(bias_list[[i]]),tf$float32)
  }
  
  input_sub[[1]] <- tf$nn$sigmoid(tf$matmul(input, weight[[1]]) + bias[[1]])
  for(i in 2:(length(dbn_sizes)+1)){
    input_sub[[i]] <- tf$nn$sigmoid(tf$matmul(input_sub[[i-1]], weight[[i]]) + bias[[i]])
  }
  
  #Define the cost function
  cost = tf$reduce_mean(tf$square(input_sub[[length(input_sub)]] - output))
  
  #Define the training operation (Momentum Optimizer minimizing the Cost function)
  train_op <- tf$train$MomentumOptimizer(learning_rate, momentum)$minimize(cost)
  
  #Prediction operation
  predict_op = tf$argmax(input_sub[[length(input_sub)]],axis=tf$cast(1.0,tf$int32))
  
  #Training Loop
  #Initialize Variables
  sess$run(tf$global_variables_initializer())
  train_accuracy <- c()
  test_accuracy <- c()
  for(ep in 1:epochs){
    for(i in seq(0,(dim(Xdata)[1]-batchsize),batchsize)){
      batchX <- Xdata[(i+1):(i+batchsize),]
      batchY <- Ydata[(i+1):(i+batchsize),]
      
      #Run the training operation on the input data
      sess$run(train_op,feed_dict=dict(input = batchX,
                                       output = batchY))
    }
    for(j in 1:(length(dbn_sizes)+1)){
      # Retrieve weights and biases
      weight_list[[j]] <- sess$run(weight[[j]])
      bias_list[[j]] <- sess$ run(bias[[j]])
    }
    train_result <- sess$run(predict_op, feed_dict = dict(input=Xdata, output=Ydata))+1
    train_actual <- as.numeric(stringi::stri_sub(colnames(as.data.frame(Ydata))[max.col(as.data.frame(Ydata),ties.method="first")],2))
    test_result <- sess$run(predict_op, feed_dict = dict(input=Xtestdata, output=Ytestdata))+1
    test_actual <- as.numeric(stringi::stri_sub(colnames(as.data.frame(Ytestdata))[max.col(as.data.frame(Ytestdata),ties.method="first")],2))
    train_accuracy <-  c(train_accuracy,mean(train_actual==train_result))
    test_accuracy <- c(test_accuracy,mean(test_actual==test_result))
    cat("epoch:", ep, " Train Accuracy: ",train_accuracy[ep]," Test Accuracy : ",test_accuracy[ep],"\n")
  }
  return(list(train_accuracy=train_accuracy,
              test_accuracy=test_accuracy,
              weight_list=weight_list,
              bias_list=bias_list))
}

## Results after running Neural Network
Xdata=trainX
Ydata=trainY
Xtestdata=testX
Ytestdata=testY
input_size=ncol(trainX)
dbn_sizes = RBM_hidden_sizes
rbm_list = RBM_output

NN_results <- NN_train(Xdata=trainX,
                       Ydata=trainY,
                       Xtestdata=testX,
                       Ytestdata=testY,
                       input_size=ncol(trainX),
                       rbm_list=RBM_output,
                       dbn_sizes = RBM_hidden_sizes)


# Plot NN Accuracy
accuracy_df <- data.frame("accuracy"=c(NN_results$train_accuracy,NN_results$test_accuracy),
                          "epochs"=c(rep(1:10,times=2)),
                          "datatype"=c(rep(c(1,2),each=10)),
                          stringsAsFactors = FALSE)

plot(accuracy ~ epochs,
     xlab = "# of epochs",
     ylab = "Accuracy in %",
     pch = c(16, 1)[datatype], 
     main = "Neural Network - Accuracy in %",
     data = accuracy_df)

legend('bottomright',
       c("train","test"), 
       pch = c( 16, 1))



