# COMPARE PCA AND RBM
# Sys.setenv(TENSORFLOW_PYTHON_VERSION = 3)
library(tensorflow)
library(rbm)
library(ggplot2)
np <- import("numpy")

# Create TensorFlow session
# Reset the graph
tf$reset_default_graph()
# Starting session as interactive session
sess <- tf$InteractiveSession()

# Input data (MNIST)
mnist <- tf$examples$tutorials$mnist$input_data$read_data_sets("MNIST-data/",one_hot=TRUE)
trainX <- mnist$train$images[1:30000,]
trainY <- mnist$train$labels[1:30000,]
testX <- mnist$test$images
testY <- mnist$test$labels

# PCA
PCA_model <- prcomp(trainX, retx=TRUE)
RBM_model <- rbm::rbm(trainX, retx=TRUE, max_epoch=500,num_hidden =900)

# Predict on Train and Test data
PCA_pred_train <- predict(PCA_model)
RBM_pred_train <- predict(RBM_model,type='probs')

# Convert into dataframes
PCA_pred_train <- as.data.frame(PCA_pred_train)
RBM_pred_train <- as.data.frame(as.matrix(RBM_pred_train))

# Train actuals
trainY <- as.numeric(stringi::stri_sub(colnames(as.data.frame(trainY))[max.col(as.data.frame(trainY),ties.method="first")],2))

# Plot PCA and RBM
ggplot(PCA_pred_train, aes(PC1, PC2))+ 
  geom_point(aes(colour = trainY))+
  theme_bw()+labs(title="PCA - Distribution of digits")+  
  theme(plot.title = element_text(hjust = 0.5))

ggplot(RBM_pred_train, aes(Hidden_1, Hidden_2))+ 
  geom_point(aes(colour = trainY))+
  theme_bw()+labs(title="RBM - Distribution of digits")+  
  theme(plot.title = element_text(hjust = 0.5))

# Plots for variance explained in PCA
# No of Principal Components vs Cumulative Variance Explained
var_explain <- as.data.frame(PCA_model$sdev^2/sum(PCA_model$sdev^2))
var_explain <- cbind(c(1:784),var_explain,cumsum(var_explain[,1]))
colnames(var_explain) <- c("PcompNo.","Ind_Variance","Cum_Variance")
plot(var_explain$PcompNo.,var_explain$Cum_Variance, xlim = c(0,100),type='b',pch=16,xlab = "# of Principal Components",ylab = "Cumulative Variance",main = 'PCA - Explained variance')

# Plot to show reconstruction trainig error
plot(RBM_model,xlab = "# of epoch iterations",ylab = "Reconstruction error",main = 'RBM - Reconstruction Error')
