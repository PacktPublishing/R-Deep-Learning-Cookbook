require(SAENET)

# Functions to load dataset
load_occupancy_data<-function(train){
  setwd("D:/PKS/00 PACKT/Deep Learning Cookbook - R/00 Initial Chapters/src/Chapter 2/data/")
  xFeatures = c("Temperature", "Humidity", "Light", "CO2", "HumidityRatio")
  yFeatures = "Occupancy"
  if(train){
    occupancy_ds <- as.matrix(read.csv("datatraining.txt",stringsAsFactors = T))
  } else
  {
    occupancy_ds <- as.matrix(read.csv("datatest.txt",stringsAsFactors = T))
  }
  occupancy_ds<-apply(occupancy_ds[, c(xFeatures, yFeatures)], 2, FUN=as.numeric) 
  return(list("data"=occupancy_ds, "xFeatures"=xFeatures, "yFeatures"=yFeatures))
}

# Function for min-max normalization
minmax.normalize<-function(ds, scaler=NULL){
  if(is.null(scaler)){
    for(f in ds$xFeatures){
      scaler[[f]]$minval<-min(ds$data[,f])
      scaler[[f]]$maxval<-max(ds$data[,f])
      ds$data[,f]<-(ds$data[,f]-scaler[[f]]$minval)/(scaler[[f]]$maxval-scaler[[f]]$minval)
    }
    ds$scaler<-scaler
  } else
  {
    for(f in ds$xFeatures){
      ds$data[,f]<-(ds$data[,f]-scaler[[f]]$minval)/(scaler[[f]]$maxval-scaler[[f]]$minval)
    }
  }
  return(ds)
}

# Load dataset
occupancy_train <-load_occupancy_data(train=T)
occupancy_test <- load_occupancy_data(train = F)

# Normalize dataset
occupancy_train<-minmax.normalize(occupancy_train, scaler = NULL)
occupancy_test<-minmax.normalize(occupancy_test, scaler = occupancy_train$scaler)

###################################
# PCA
###################################

pca_obj <- prcomp(subset(occupancy_train$data, select=-c(Occupancy)),
                 center = TRUE,
                 scale. = TRUE)
plot(pca_obj, type = "l")
xlab("Principal Componets")
biplot(pca_obj, scale = 0)
ds <- data.frame(occupancy_train$data, pca_obj$x[,1:2])
component_plot <- qplot(x=PC1, y=PC2, data=ds, colour=factor(Occupancy)) + theme(legend.position="none") 

# Qplot for principle components
require(ggplot2)

###################################
# Single dimension
###################################
# Building Stacked Autoencoder
SAE_obj<-SAENET.train(X.train= subset(occupancy_train$data, select=-c(Occupancy)), n.nodes=c(4, 3, 1), unit.type ="tanh", lambda = 1e-5, beta = 1e-5, rho = 0.01, epsilon = 0.01, max.iterations=1000)  

# plotting encoder train values
plot(SAE_obj[[3]]$X.output[,1], col="blue", xlab = "Node 1 of layer 3", ylab = "Values")
ix<-occupancy_train$data[,6]==1  
points(seq(1, nrow(SAE_obj[[3]]$X.output),by=1)[ix], SAE_obj[[3]]$X.output[ix,1], col="red")
legend(7000,0.45, c("0","1"), lty=c(0,0), pch=1, col=c("blue","red")) # gives the legend lines the correct color and width


###################################
# Two dimension
###################################
# Building Stacked Autoencoder
SAE_obj<-SAENET.train(X.train= subset(occupancy_train$data, select=-c(Occupancy)), n.nodes=c(4, 3, 2), unit.type ="tanh", lambda = 1e-5, beta = 1e-5, rho = 0.01, epsilon = 0.01, max.iterations=1000)  

# plotting encoder train values
plot(SAE_obj[[3]]$X.output[,1], SAE_obj[[3]]$X.output[,2], col="blue", xlab = "Node 1 of layer 3", ylab = "Node 2 of layer 3")
ix<-occupancy_train$data[,6]==1  
points(SAE_obj[[3]]$X.output[ix,1], SAE_obj[[3]]$X.output[ix,2], col="red")
legend(0,0.6, c("0","1"), lty=c(0,0), pch=1, col=c("blue","red")) # gives the legend lines the correct color and width

