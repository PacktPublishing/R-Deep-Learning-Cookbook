# function to run sanity check on photos & labels import
drawImage <- function(index, images.rgb, images.lab=NULL) {
  # Testing the parsing: Convert each color layer into a matrix,
  # combine into an rgb object, and display as a plot
  img <- images.rgb[[index]]
  img.r.mat <- as.cimg(matrix(img$r, ncol=32, byrow = FALSE))
  img.g.mat <- as.cimg(matrix(img$g, ncol=32, byrow = FALSE))
  img.b.mat <- as.cimg(matrix(img$b, ncol=32, byrow = FALSE))
  img.col.mat <- imappend(list(img.r.mat,img.g.mat,img.b.mat),"c") #Bind the three channels into one image
  
  # Plot and output label
  plot(img.col.mat)
  
  if(!is.null(images.lab)){
    labels[[1]][images.lab[[index]]]  
  }
  return(img.col.mat)
}