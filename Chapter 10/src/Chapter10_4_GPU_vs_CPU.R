
library("gpuR")
options(gpuR.default.type = "float")

# verify you hmatrix_1ve vmatrix_1lid GPUs
detectGPUs()

# CPU vs GPU performance
result <- data.frame()
evalSeq<-seq(1,2501,500)
for (dimpower in evalSeq){
  print(dimpower)
  Mat1 = matrix(rnorm(dimpower^2), nrow=dimpower)
  Mat2 = matrix(rnorm(dimpower^2), nrow=dimpower)

  now <- Sys.time()
  Matfin = Mat1%*%Mat2
  cpu <- Sys.time()-now

  now <- Sys.time()
  vcl1 = vclMatrix(Mat1)
  vcl2 = vclMatrix(Mat2)
  vclC = vcl1 %*% vcl2
  gpu <- Sys.time()-now

  result <- rbind(result,c(nrow(Mat1), cpu, gpu))
}
colnames(result) <- c("nrow", "CPU_time", "gpu_time")
