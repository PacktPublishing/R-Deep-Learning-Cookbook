download.cifar.data <- function(data_dir) {
  dir.create(data_dir, showWarnings = FALSE)
  setwd(data_dir)
  if (!file.exists('cifar-10-binary.tar.gz')){
    download.file(url='http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz', destfile='cifar-10-binary.tar.gz', method='wget')
    untar("cifar-10-binary.tar.gz") # Unzip files
    file.remove("cifar-10-binary.tar.gz") # remove zip file
  }
  setwd("..")
}


# Function to read cifar data
read.cifar.data <- function(filenames,num.images=10000){
  images.rgb <- list()
  images.lab <- list()
  for (f in 1:length(filenames)) {
    to.read <- file(paste(DATA_PATH,filenames[f], sep=""), "rb")
    for(i in 1:num.images) {
      l <- readBin(to.read, integer(), size=1, n=1, endian="big")
      r <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
      g <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
      b <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
      index <- num.images * (f-1) + i
      images.rgb[[index]] = data.frame(r, g, b)
      images.lab[[index]] = l+1
    }
    close(to.read)
    cat("completed :",  filenames[f], "\n")
    remove(l,r,g,b,f,i,index, to.read)
  }
  return(list("images.rgb"=images.rgb,"images.lab"=images.lab))
}