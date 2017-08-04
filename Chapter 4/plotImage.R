# Plotting actual and corrupted image
plotImage<-function(actds, corrds){
  # Data re-formatting
  actImage<-matrix(actds, ncol = 3, byrow = F)
  imgcorr<-matrix(corrds, ncol = 3, byrow = F)
  
  # Image Format
  img.col.mat <- imappend(list(as.cimg(actImage[,1]),as.cimg(actImage[,2]), as.cimg(actImage[,3])),"c")
  img.col.mat.corr <- imappend(list(as.cimg(imgcorr[,1]),as.cimg(imgcorr[,2]), as.cimg(imgcorr[,3])),"c")
  # Plot images
  par(mfrow=c(1,2))
  plot(img.col.mat, main="Actual Plot")
  plot(img.col.mat.corr, main="Corrupted Plot")
  
  return(list("actImg"=img.col.mat, "corrImg"=imgcorr))
}