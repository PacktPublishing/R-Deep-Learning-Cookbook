# Function to normalize data 
normalizeData<-function(data, method=c("minmax", "normalize"), obj=NULL){
  if(length(method)>1) method<-"minmax" # Default
  
  if(!is.null(obj) & class(obj)=="normalization"){
    method<-obj$method
    if(method=="minmax"){
      colMin<-obj$scaler1
      colMax<-obj$scaler2
    } else
    {
      colMean<-obj$scaler1
      colSD<-obj$scaler2
    }
  } else
  {
    if(method=="minmax"){
      colMin<-apply(train_data$images, 2, FUN=min)
      colMax<-apply(train_data$images, 2, FUN=max)
    } else
    {
      colMean<-apply(train_data$images, 2, FUN=mean)
      colSD<-apply(train_data$images, 2, FUN=sd)
    }
  }
  
  output<-NULL
  if(method=="minmax"){
    output$normalize_data<-sapply(seq(1, ncol(data), by = 1), 
                                  FUN=function(x, data, colMin, colMax)
                                  {
                                    (data[, x]-colMin[x])/(colMax[x]-colMin[x])
                                  }, data, colMin, colMax)
    output$scaler1<-colMin
    output$scaler2<-colMax
  } else
  {
    output$normalize_data<-sapply(seq(1, ncol(data), by = 1), 
                                  FUN=function(x, data, colMin, colMax)
                                  {
                                    (data[, x]-colMean[x])/colSD[x]
                                  }, data, colMean, colSD)
    output$scaler1<-colMean
    output$scaler2<-colSD
  }
  
  output$method<-method
  class(output)<-"normalization"
  output
}

# Function to denormalize dataset using normalization object
denomalization<-function(obj, data){
  if(class(obj)!="normalization") stop("object not defined")
  method<-obj$method
  if(method=="minmax"){
    colMin<-obj$scaler1
    colMax<-obj$scaler2
    denormalizedData<-sapply(seq(1, ncol(data), by=1), 
                             FUN=function(x, data, colMin, colMax)
                             {
                               (data[, x]*(colMax[x]-colMin[x])+-colMin[x])
                             }, data, colMin, colMax)
  } else
  {
    colMean<-obj$scaler[1]
    colSD<-obj$scaler[2]
    denormalizedData<-sapply(seq(1, dim(data), by=1), 
                             FUN=function(x, data, colMin, colMax)
                             {
                               (data[, x]*colSD[x])+colMean[x]
                             }, data, colMean, colSD)
  }
  denormalizedData
}
