require(imager)
require(SAENET)

# Load required functions
FUN_PATH<-"D:/PKS/00 PACKT/Deep Learning Cookbook - R/00 Initial Chapters/src/Chapter 4/"
setwd(FUN_PATH)
load_functions<-c("drawImage.R", "plotImage.R", "add_noise.R", "normalize.R", "get_cifar_data.R", "flat_data.R")
sapply(load_functions, FUN=source)


## LOAD Dataset
DATA_PATH<-"D:/PKS/00 PACKT/Deep Learning Cookbook - R/00 Initial Chapters/src/Chapter 4/data/cifar-10-batches-bin/"
ds_obj<-get_cifar_data(DATA_PATH = DATA_PATH)

# flatten dataset 
train_data <- flat_data(x_listdata = ds_obj$train)
valid_data <- flat_data(x_listdata = ds_obj$valid)
test_data <- flat_data(x_listdata = ds_obj$test)

# Noramlize dataset
train=normalizeData(train_data$images, method="minmax")
valid=normalizeData(valid_data$images, method="minmax", obj = train)
test=normalizeData(test_data$images, method="minmax", obj = train)

# Building Stacked Autoencoder
require(RcppDL)
train_X<-as.matrix(train$normalize_data[1:100,])
train_Y<-as.matrix(train$normalize_data[1:100,])
sda_test <- Rsda(train_X, train_Y, hidden)
setCorruptionLevel(sda_test, 0.2)
setPretrainEpochs(sda_test, 1000)
setFinetuneEpochs(sda_test, 1000)
setPretrainLearningRate(sda_test, 0.3)
setFinetuneLearningRate(sda_test, 0.1)
summary(sda_test)
LearningRate(sda_test)
pretrain(sda_test)
finetune(sda_test)
yhat<-predict(sda_test, test_X)

# Running stacked autoencoder
SAE_obj<-SAENET.train(X.train=train$normalize_data, n.nodes=c(1024, 512, 256), unit.type ="tanh", 
                     lambda = 1e-5, beta = 1e-5, rho = 0.01, epsilon = 0.01, max.iterations=1000)










