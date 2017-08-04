Sys.setenv(TENSORFLOW_PYTHON="C:/PROGRA~3/ANACON~1/python.exe")
Sys.setenv(TENSORFLOW_PYTHON_VERSION = 3)
require(tensorflow)
require(imager)
require(caret)

# Load mnist dataset from tensorflow library
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

# Function to plot MNIST dataset
plot_mnist<-function(imageD, pixel.y=16){
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
rm(mnist)


# Plot a sample image
plot_mnist(trainData[20,])

# Normalize Dataset
normalizeObj<-preProcess(trainData, method="range")
trainData<-predict(normalizeObj, trainData)
validData<-predict(normalizeObj, validData)


#############
# Set-up VAE Model
#############
tf$reset_default_graph()
sess<-tf$InteractiveSession()

# Define Input as Placeholder variables
n_input=256
n.hidden.enc.1<-64
n.hidden.enc.2<-32
n_h=16
n.hidden.dec.1<-32
n.hidden.dec.2<-64

lr<-0.01 # learning rate
ITERATION = 100
BATCH<-500

# Xavier Initialization using Uniform distribution 
xavier_init<-function(n_inputs, n_outputs, constant=1){
  low = -constant*sqrt(6.0/(n_inputs + n_outputs)) 
  high = constant*sqrt(6.0/(n_inputs + n_outputs))
  return(tf$random_uniform(shape(n_inputs, n_outputs), minval=low, maxval=high, dtype=tf$float32))
}

model_init<-function(n.hidden.enc.1, n.hidden.enc.2, 
                               n.hidden.dec.1,  n.hidden.dec.2, 
                               n_input, n_h){
  weights<-NULL
  ############################
  # Set-up Encoder
  ############################
  # Initialize Layer 1 of encoder
  weights[["encoder_w"]][["h1"]]=tf$Variable(xavier_init(n_input, n.hidden.enc.1))
  weights[["encoder_w"]][["h2"]]=tf$Variable(xavier_init(n.hidden.enc.1, n.hidden.enc.2))
  weights[["encoder_w"]][["out_mean"]]=tf$Variable(xavier_init(n.hidden.enc.2, n_h))
  weights[["encoder_w"]][["out_log_sigma"]]=tf$Variable(xavier_init(n.hidden.enc.2, n_h))
  weights[["encoder_b"]][["b1"]]=tf$Variable(tf$zeros(shape(n.hidden.enc.1), dtype=tf$float32))
  weights[["encoder_b"]][["b2"]]=tf$Variable(tf$zeros(shape(n.hidden.enc.2), dtype=tf$float32))
  weights[["encoder_b"]][["out_mean"]]=tf$Variable(tf$zeros(shape(n_h), dtype=tf$float32))
  weights[["encoder_b"]][["out_log_sigma"]]=tf$Variable(tf$zeros(shape(n_h), dtype=tf$float32))
  
  ############################
  # Set-up Decoder
  ############################
  weights[['decoder_w']][["h1"]]=tf$Variable(xavier_init(n_h, n.hidden.dec.1))
  weights[['decoder_w']][["h2"]]=tf$Variable(xavier_init(n.hidden.dec.1, n.hidden.dec.2))
  weights[['decoder_w']][["out_mean"]]=tf$Variable(xavier_init(n.hidden.dec.2, n_input))
  weights[['decoder_w']][["out_log_sigma"]]=tf$Variable(xavier_init(n.hidden.dec.2, n_input))
  weights[['decoder_b']][["b1"]]=tf$Variable(tf$zeros(shape(n.hidden.dec.1), dtype=tf$float32))
  weights[['decoder_b']][["b2"]]=tf$Variable(tf$zeros(shape(n.hidden.dec.2), dtype=tf$float32))
  weights[['decoder_b']][["out_mean"]]=tf$Variable(tf$zeros(shape(n_input), dtype=tf$float32))
  weights[['decoder_b']][["out_log_sigma"]]=tf$Variable(tf$zeros(shape(n_input), dtype=tf$float32))
  return(weights)
}

# Encoder update function
vae_encoder<-function(x, weights, biases){
  layer_1 = tf$nn$softplus(tf$add(tf$matmul(x, weights[['h1']]), biases[['b1']])) 
  layer_2 = tf$nn$softplus(tf$add(tf$matmul(layer_1, weights[['h2']]), biases[['b2']])) 
  z_mean = tf$add(tf$matmul(layer_2, weights[['out_mean']]), biases[['out_mean']])
  z_log_sigma_sq = tf$add(tf$matmul(layer_2, weights[['out_log_sigma']]), biases[['out_log_sigma']])
  return (list("z_mean"=z_mean, "z_log_sigma_sq"=z_log_sigma_sq))
}


# Decoder update function
vae_decoder<-function(z, weights, biases){
  layer1<-tf$nn$softplus(tf$add(tf$matmul(z, weights[["h1"]]), biases[["b1"]]))
  layer2<-tf$nn$softplus(tf$add(tf$matmul(layer1, weights[["h2"]]), biases[["b2"]]))
  x_reconstr_mean<-tf$nn$sigmoid(tf$add(tf$matmul(layer2, weights[['out_mean']]), biases[['out_mean']]))
  return(x_reconstr_mean)
}

# Parameter evaluation
network_ParEval<-function(x, network_weights, n_h){

  distParameter<-vae_encoder(x, network_weights[["encoder_w"]], network_weights[["encoder_b"]])
  z_mean<-distParameter$z_mean
  z_log_sigma_sq <-distParameter$z_log_sigma_sq
  
  # Draw one sample z from Gaussian distribution
  eps = tf$random_normal(shape(BATCH, n_h), 0, 1, dtype=tf$float32)
  
  # z = mu + sigma*epsilon
  z = tf$add(z_mean, tf$multiply(tf$sqrt(tf$exp(z_log_sigma_sq)), eps))
  
  # Use generator to determine mean of
  # Bernoulli distribution of reconstructed input
  x_reconstr_mean <- vae_decoder(z, network_weights[["decoder_w"]], network_weights[["decoder_b"]])
  return(list("x_reconstr_mean"=x_reconstr_mean, "z_log_sigma_sq"=z_log_sigma_sq, "z_mean"=z_mean))
}

# VAE cost function
vae_optimizer<-function(x, networkOutput){
  x_reconstr_mean<-networkOutput$x_reconstr_mean
  z_log_sigma_sq<-networkOutput$z_log_sigma_sq
  z_mean<-networkOutput$z_mean
  loss_reconstruction<--1*tf$reduce_sum(x*tf$log(1e-10 + x_reconstr_mean)+
                                       (1-x)*tf$log(1e-10 + 1 - x_reconstr_mean), reduction_indices=shape(1))
  loss_latent<--0.5*tf$reduce_sum(1+z_log_sigma_sq-tf$square(z_mean)-
                                    tf$exp(z_log_sigma_sq), reduction_indices=shape(1))
  cost = tf$reduce_mean(loss_reconstruction + loss_latent)
  return(cost)
}

# VAE Initialization
x = tf$placeholder(tf$float32, shape=shape(NULL, img_size_flat), name='x')
network_weights<-model_init(n.hidden.enc.1, n.hidden.enc.2, 
                            n.hidden.dec.1,  n.hidden.dec.2, 
                            n_input, n_h)
networkOutput<-network_ParEval(x, network_weights, n_h)
cost=vae_optimizer(x, networkOutput)
optimizer = tf$train$AdamOptimizer(lr)$minimize(cost) 
sess$run(tf$global_variables_initializer())

# Running optimization
for(i in 1:ITERATION){
  spls <- sample(1:dim(trainData)[1],BATCH)
  splval<-sample(1:dim(validData)[1],BATCH)
  out<-optimizer$run(feed_dict = dict(x=trainData[spls,]))
  if (i %% 100 == 0){
    cat("Iteration - ", i, "Training Loss - ",  cost$eval(feed_dict = dict(x=trainData[spls,])), " Validation Error - ",  cost$eval(feed_dict = dict(x=validData[splval,])), "\n")
  }
}

# Output evaluation for a batch
spls <- sample(1:dim(trainData)[1],BATCH)
networkOutput_run<-sess$run(networkOutput, feed_dict = dict(x=trainData[spls,]))

# Plot reconstructured Image
x_sample<-trainData[spls,]
NROW<-nrow(networkOutput_run$x_reconstr_mean)
n.plot<-5
par(mfrow = c(n.plot, 2), mar = c(0.2, 0.2, 0.2, 0.2), oma = c(3, 3, 3, 3))
pltImages<-sample(1:NROW,n.plot)
for(i in pltImages){
  plot_mnist(x_sample[i,])
  plot_mnist(networkOutput_run$x_reconstr_mean[i,])
}
