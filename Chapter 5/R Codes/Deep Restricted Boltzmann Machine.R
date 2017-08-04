###############  DEEP Restricted Boltzmann Machine
# Input data (MNIST)
mnist <- tf$examples$tutorials$mnist$input_data$read_data_sets("MNIST-data/",one_hot=TRUE)
trainX <- mnist$train$images
trainY <- mnist$train$labels
testX <- mnist$test$images
testY <- mnist$test$labels

# Global Parameters
learning_rate      = 0.005     
momentum      = 0.005     
minbatch_size      = 25        
hidden_layers = c(400,100) 
biases  = list(-1,-1)   

# Helper functions
arcsigm <- function(x){
  return(atanh((2*x)-1)*2)
}

sigm <- function(x){
  return(tanh((x/2)+1)/2)
}

binarize <- function(x){
  # truncated rnorm
  trnrom <- function(n, mean, sd, minval = -Inf, maxval = Inf){
    qnorm(runif(n, pnorm(minval, mean, sd), pnorm(maxval, mean, sd)), mean, sd)
  }
  return((x > matrix( trnrom(n=nrow(x)*ncol(x),mean=0,sd=1,minval=0,maxval=1), nrow(x), ncol(x)))*1)
}

re_construct <- function(x){
  x = x - min(x) + 1e-9
  x = x / (max(x) + 1e-9)
  return(x*255)
}

gibbs <- function(X,l,initials){
  if(l>1){
    bu <- (X[l-1][[1]] - matrix(rep(initials$param_O[[l-1]],minbatch_size),minbatch_size,byrow=TRUE)) %*%
      initials$param_W[l-1][[1]]
  } else {
    bu <- 0
  }
  if((l+1) < length(X)){
    td <- (X[l+1][[1]] - matrix(rep(initials$param_O[[l+1]],minbatch_size),minbatch_size,byrow=TRUE))%*%
      t(initials$param_W[l][[1]])
  } else {
    td <- 0
  }
  X[[l]] <- binarize(sigm(bu+td+matrix(rep(initials$param_B[[l]],minbatch_size),minbatch_size,byrow=TRUE)))
  return(X[[l]])
}

# Reparameterization
reparamBias <- function(X,l,initials){
  if(l>1){
    bu <- colMeans((X[[l-1]] - matrix(rep(initials$param_O[[l-1]],minbatch_size),minbatch_size,byrow=TRUE))%*%
                     initials$param_W[[l-1]])
  } else {
    bu <- 0
  }
  if((l+1) < length(X)){
    td <- colMeans((X[[l+1]] - matrix(rep(initials$param_O[[l+1]],minbatch_size),minbatch_size,byrow=TRUE))%*%
                     t(initials$param_W[[l]]))
  } else {
    td <- 0
  }
  initials$param_B[[l]] <- (1-momentum)*initials$param_B[[l]] + momentum*(initials$param_B[[l]] + bu + td)
  return(initials$param_B[[l]])
}

reparamO <- function(X,l,initials){
  initials$param_O[[l]] <- colMeans((1-momentum)*matrix(rep(initials$param_O[[l]],minbatch_size),minbatch_size,byrow=TRUE) + momentum*(X[[l]]))
  return(initials$param_O[[l]])
}

DRBM_initialize <- function(layers,bias_list){
  # Initialize model parameters and particles
  param_W <- list()
  for(i in 1:(length(layers)-1)){
    param_W[[i]] <- matrix(0L, nrow=layers[i], ncol=layers[i+1])
  }
  param_B <- list()
  for(i in 1:length(layers)){
    param_B[[i]] <- matrix(0L, nrow=layers[i], ncol=1) + bias_list[[i]]
  }
  param_O <- list()
  for(i in 1:length(param_B)){
    param_O[[i]] <- sigm(param_B[[i]])
  }
  param_X <- list()
  for(i in 1:length(layers)){
    param_X[[i]] <- matrix(0L, nrow=minbatch_size, ncol=layers[i]) + matrix(rep(param_O[[i]],minbatch_size),minbatch_size,byrow=TRUE)
  }  
  return(list(param_W=param_W,param_B=param_B,param_O=param_O,param_X=param_X))
}

# Run Initialize
X <- trainX/255
layers <- c(784,hidden_layers)
bias_list <- list(arcsigm(pmax(colMeans(X),0.001)),biases[[1]],biases[[2]])
initials <-DRBM_initialize(layers,bias_list)

# START TRAINING
batchX <- X[sample(nrow(X))[1:minbatch_size],]
for(iter in 1:1000){
  # Perform some learnings
  for(j in 1:100){
    # Initialize a data particle
    dat <- list()
    dat[[1]] <- binarize(batchX)
    for(l in 2:length(initials$param_X)){
      dat[[l]] <- initials$param_X[l][[1]]*0 + matrix(rep(initials$param_O[l][[1]],minbatch_size),minbatch_size,byrow=TRUE)
    }
    # Alternate gibbs sampler on data and free particles
    for(l in rep(c(seq(2,length(initials$param_X),2), seq(3,length(initials$param_X),2)),5)){
      dat[[l]] <- gibbs(dat,l,initials)
    }
    
    for(l in rep(c(seq(2,length(initials$param_X),2), seq(1,length(initials$param_X),2)),1)){
      initials$param_X[[l]] <- gibbs(initials$param_X,l,initials)
    }
    
    # Parameter update
    for(i in 1:length(initials$param_W)){
      initials$param_W[[i]] <- initials$param_W[[i]] + (learning_rate*((t(dat[[i]] - matrix(rep(initials$param_O[i][[1]],minbatch_size),minbatch_size,byrow=TRUE)) %*%
                                                                          (dat[[i+1]] - matrix(rep(initials$param_O[i+1][[1]],minbatch_size),minbatch_size,byrow=TRUE))) - 
                                                                         (t(initials$param_X[[i]] - matrix(rep(initials$param_O[i][[1]],minbatch_size),minbatch_size,byrow=TRUE)) %*%
                                                                            (initials$param_X[[i+1]] - matrix(rep(initials$param_O[i+1][[1]],minbatch_size),minbatch_size,byrow=TRUE))))/nrow(batchX))
    }
    
    for(i in 1:length(initials$param_B)){
      initials$param_B[[i]] <- colMeans(matrix(rep(initials$param_B[[i]],minbatch_size),minbatch_size,byrow=TRUE) + (learning_rate*(dat[[i]] - initials$param_X[[i]])))
    }
    
    # Reparameterization
    for(l in 1:length(initials$param_B)){
      initials$param_B[[l]] <- reparamBias(dat,l,initials)
    }
    for(l in 1:length(initials$param_O)){
      initials$param_O[[l]] <- reparamO(dat,l,initials)
    }
  }
  
  # Generate necessary outputs
  cat("Iteration:",iter," ","Mean of W of VL-HL1:",mean(initials$param_W[[1]])," ","Mean of W of HL1-HL2:",mean(initials$param_W[[2]]) ,"\n")
  cat("Iteration:",iter," ","SDev of W of VL-HL1:",sd(initials$param_W[[1]])," ","SDev of W of HL1-HL2:",sd(initials$param_W[[2]]) ,"\n")
  
  # Plot weight matrices
  W=diag(nrow(initials$param_W[[1]]))
  for(l in 1:length(initials$param_W)){
    W = W %*% initials$param_W[[l]]
    m = dim(W)[2] * 0.05
    w1_arr <- matrix(0,28*m,28*m)
    i=1
    for(k in 1:m){
      for(j in 1:28){
        vec <- c(W[(28*j-28+1):(28*j),(k*m-m+1):(k*m)])
        w1_arr[i,] <- vec
        i=i+1
      }
    }
    w1_arr = re_construct(w1_arr)
    w1_arr <- floor(w1_arr)
    image(w1_arr,axes = TRUE, col = grey(seq(0, 1, length = 256)))
  }
  
}


