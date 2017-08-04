## TENSOR FLOW Restricted Boltzman Machine
## Tensorflow implementation of Restricted Boltzman Machine for layerwise pretraining of deep autoencoders

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

# Convert thr train data into gaussian distribution format i.e mean=0 and SD=1
trainX_normalised <- t(apply(trainX,1,function(x){
  return((x-mean(x))/sd(x))
}))

# Initialise parameters
num_input<-784L
num_hidden<-900L
alpha<-0.1

# Placeholder variables
vb <- tf$placeholder(tf$float32, shape = shape(num_input))
hb <- tf$placeholder(tf$float32, shape = shape(num_hidden))
W <- tf$placeholder(tf$float32, shape = shape(num_input, num_hidden))

# Phase 1 : Forward Pass
X = tf$placeholder(tf$float32, shape=shape(NULL, num_input))
prob_h0= tf$nn$sigmoid(tf$matmul(X, W) + hb)  
h0 = tf$nn$relu(tf$sign(prob_h0 - tf$random_uniform(tf$shape(prob_h0)))) 

# Look at sampling
sess$run(tf$global_variables_initializer())
s1 <- tf$constant(value = c(0.1,0.4,0.7,0.9))
cat(sess$run(s1))
s2=sess$run(tf$random_uniform(tf$shape(s1)))
cat(s2)
cat(sess$run(s1-s2))
cat(sess$run(tf$sign(s1 - s2)))
cat(sess$run(tf$nn$relu(tf$sign(s1 - s2))))

# Phase 2 : Backward Pass
prob_v1 = tf$matmul(h0, tf$transpose(W)) + vb
v1 = prob_v1 + tf$random_normal(tf$shape(prob_v1), mean=0.0, stddev=1.0, dtype=tf$float32)
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
cur_w = tf$Variable(tf$zeros(shape = shape(num_input, num_hidden), dtype=tf$float32))
cur_vb = tf$Variable(tf$zeros(shape = shape(num_input), dtype=tf$float32))
cur_hb = tf$Variable(tf$zeros(shape = shape(num_hidden), dtype=tf$float32))
prv_w = tf$Variable(tf$random_normal(shape=shape(num_input, num_hidden), stddev=0.01, dtype=tf$float32))
prv_vb = tf$Variable(tf$zeros(shape = shape(num_input), dtype=tf$float32))
prv_hb = tf$Variable(tf$zeros(shape = shape(num_hidden), dtype=tf$float32)) 

# Start tensorflow session
sess$run(tf$global_variables_initializer())
output <- sess$run(list(update_w, update_vb, update_hb), feed_dict = dict(X=trainX_normalised,
                                                                          W = prv_w$eval(),
                                                                          vb = prv_vb$eval(),
                                                                          hb = prv_hb$eval()))
prv_w <- output[[1]] 
prv_vb <- output[[2]]
prv_hb <-  output[[3]]
sess$run(err, feed_dict=dict(X= trainX_normalised, W= prv_w, vb= prv_vb, hb= prv_hb))

epochs=14
errors <- list()
weights <- list()
u=1
for(ep in 1:epochs){
  for(i in seq(0,(dim(trainX_normalised)[1]-100),100)){
    batchX <- trainX_normalised[(i+1):(i+100),]
    output <- sess$run(list(update_w, update_vb, update_hb), feed_dict = dict(X=batchX,
                                                                              W = prv_w,
                                                                              vb = prv_vb,
                                                                              hb = prv_hb))
    prv_w <- output[[1]] 
    prv_vb <- output[[2]]
    prv_hb <-  output[[3]]
    if(i%%10000 == 0){
      errors[[u]] <- sess$run(err, feed_dict=dict(X= trainX_normalised, W= prv_w, vb= prv_vb, hb= prv_hb))
      weights[[u]] <- output[[1]]
      u <- u+1
      cat(i , " : ")
    }
  }
  cat("epoch :", ep, " : reconstruction error : ", errors[length(errors)][[1]],"\n")
}

# Plot reconstruction error
error_vec <- unlist(errors)
plot(error_vec,xlab="# of batches",ylab="mean squared reconstruction error",main="RBM-Reconstruction MSE plot")

# Plot the last obtained weights
uw = t(weights[[length(weights)]])
numXpatches = 20
numYpatches=20
pixels <- list()
op <- par(no.readonly = TRUE)
par(mfrow = c(numXpatches,numYpatches), mar = c(0.2, 0.2, 0.2, 0.2), oma = c(3, 3, 3, 3))
for (i in 1:(numXpatches*numYpatches)) {
  denom <- sqrt(sum(uw[i, ]^2))
  pixels[[i]] <- matrix(uw[i, ]/denom, nrow = numYpatches, 
                        ncol = numXpatches)
  image(pixels[[i]], axes = F, col = gray((0:32)/32))
}
par(op)

# Sample case
sample_image <- trainX[1:4,]
mw=melt(sample_image); mw$Var3=floor((mw$Var2-1)/28)+1; mw$Var2=(mw$Var2-1)%%28 + 1; mw$Var3=29-mw$Var3;
ggplot(data=mw)+geom_tile(aes(Var2,Var3,fill=value))+facet_wrap(~Var1,nrow=2)+
  scale_fill_continuous(low='white',high='black')+coord_fixed(ratio=1)+
  labs(x=NULL,y=NULL,title="Sample digits - Actual")+
  theme(legend.position="none")+
  theme(plot.title = element_text(hjust = 0.5))

# Now pass the image for its reconstruction
hh0 = tf$nn$sigmoid(tf$matmul(X, W) + hb)
vv1 = tf$nn$sigmoid(tf$matmul(hh0, tf$transpose(W)) + vb)
feed = sess$run(hh0, feed_dict=dict( X= sample_image, W= prv_w, hb= prv_hb))
rec = sess$run(vv1, feed_dict=dict( hh0= feed, W= prv_w, vb= prv_vb))

# plot reconstructed images
mw=melt(rec); mw$Var3=floor((mw$Var2-1)/28)+1; mw$Var2=(mw$Var2-1)%%28 + 1; mw$Var3=29-mw$Var3;
ggplot(data=mw)+geom_tile(aes(Var2,Var3,fill=value))+facet_wrap(~Var1,nrow=2)+
  scale_fill_continuous(low='white',high='black')+coord_fixed(ratio=1)+
  labs(x=NULL,y=NULL,title="Sample digits -Reconstructed")+
  theme(legend.position="none")+
  theme(plot.title = element_text(hjust = 0.5))
