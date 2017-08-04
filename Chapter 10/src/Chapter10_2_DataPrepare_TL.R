# Load packages
require(imager)
source("download_cifar_data.R")

# Read Dataset and labels
DATA_PATH<-paste(SOURCE_PATH, "/Chapter 4/data/cifar-10-batches-bin/", sep="")
labels <- read.table(paste(DATA_PATH, "batches.meta.txt", sep=""))
cifar_train <- read.cifar.data(filenames = c("data_batch_1.bin","data_batch_2.bin","data_batch_3.bin","data_batch_4.bin"))


# Filter data for Aeroplane and Automobile with  label 1 and 2, respectively
Classes = c(1, 2)
images.rgb.train <- cifar_train$images.rgb
images.lab.train <- cifar_train$images.lab
ix<-images.lab.train%in%Classes
images.rgb.train<-images.rgb.train[ix]
images.lab.train<-images.lab.train[ix]
rm(cifar_train)


# Function to transform to image
transform.Image <- function(index, images.rgb) {
  # Testing the parsing: Convert each color layer into a matrix,
  # combine into an rgb object, and display as a plot
  img <- images.rgb[[index]]
  img.r.mat <- as.cimg(matrix(img$r, ncol=32, byrow = FALSE))
  img.g.mat <- as.cimg(matrix(img$g, ncol=32, byrow = FALSE))
  img.b.mat <- as.cimg(matrix(img$b, ncol=32, byrow = FALSE))
  img.col.mat <- imappend(list(img.r.mat,img.g.mat,img.b.mat),"c") #Bind the three channels into one image
  return(img.col.mat)
}


# Function to pad image
image.padding <- function(x) {
  img_width <- max(dim(x)[1:2])
  img_height <- min(dim(x)[1:2])
  pad.img <- pad(x,
                 nPix = img_width - img_height,
                 axes = ifelse(dim(x)[1] < dim(x)[2], "x", "y"))
  return(pad.img)
}

# Save Image by resizing to 224 x 224 x 3
## Save as JPEG
setwd("C:/Users/pp9596/Documents/02 ZSP/00 PACKT/Deep Learning Cookbook - R/00 Initial Chapters/Chapter 10 - Transfer Learning/data/cifar_224")

# Save train images
MAX_IMAGE<-length(images.rgb.train)

# Write Aeroplane images to aero folder
sapply(1:MAX_IMAGE, FUN=function(x, images.rgb.train, images.lab.train){
  if(images.lab.train[[x]]==1){
    img<-transform.Image(x, images.rgb.train)
    pad_img <- image.padding(img)
    res_img <- resize(pad_img,  size_x = 224, size_y = 224)
    imager::save.image(res_img, paste("train/aero/aero", x, ".jpeg", sep=""))
  }
}, images.rgb.train=images.rgb.train, images.lab.train=images.lab.train)


# Write Automobile images to auto folder
sapply(1:MAX_IMAGE, FUN=function(x, images.rgb.train, images.lab.train){
  if(images.lab.train[[x]]==2){
    img<-transform.Image(x, images.rgb.train)
    pad_img <- image.padding(img)
    res_img <- resize(pad_img,  size_x = 224, size_y = 224)
    imager::save.image(res_img, paste("train/auto/auto", x, ".jpeg", sep=""))
  }
}, images.rgb.train=images.rgb.train, images.lab.train=images.lab.train)

