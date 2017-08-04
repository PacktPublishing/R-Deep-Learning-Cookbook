#################  CHAPTER 1  ##################

## IR Kernel installation for Jupyter notebook

# Install the dependencies
chooseCRANmirror(ind=55) # choose mirror for installation
install.packages(c('repr', 'IRdisplay', 'crayon', 'pbdZMQ', 'devtools', 'jsonlite', 'digest'), dependencies=TRUE)

# Install the IRkernal from github
library(devtools)
library(methods)
options(repos=c(CRAN='https://cran.rstudio.com'))
devtools::install_github('IRkernel/IRkernel')

# Supervised Learning

# Linear Regression
data <- data.frame("height" = c(131, 154, 120, 166, 108, 115, 158, 144, 131, 112),
                   "weight" = c(54, 70, 47, 79, 36, 48, 65, 63, 54, 40))

lm_model <- lm(weight ~ height, data)

plot(data, col = "red", main = "Relationship between height and weight",
     cex = 1.7, pch = 1, xlab = "Height in cms", ylab = "Weight in kgs")
abline(lm(weight ~ height, data))

# Unsupervised Learning

# K Means clustering
data(iris)
head(iris)

# Initial Visual exploration
library(ggplot2)
library(gridExtra)
plot1 <- ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species)) + geom_point(size = 2) + ggtitle("Variation by Sepal features")
plot2 <- ggplot(iris, aes(Petal.Length, Petal.Width, color = Species)) + geom_point(size = 2) + ggtitle("Variation by Petal features")
grid.arrange(plot1, plot2, ncol=2)

# Perform K means on Petal length and width
set.seed(1234567)
iris.cluster <- kmeans(iris[, c("Petal.Length","Petal.Width")], 3, nstart = 10)
iris.cluster$cluster <- as.factor(iris.cluster$cluster)

# Cross-table of Species and Clusters
table(cluster=iris.cluster$cluster,species= iris$Species)

# Post K Means Visualisation
ggplot(iris, aes(Petal.Length, Petal.Width, color = iris.cluster$cluster)) + geom_point() + ggtitle("Variation by Clusters")

########################  Setting up MXNet  #########################

install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")

########################  Setting up Tensorflow  #########################

devtools::install_github("rstudio/tensorflow")
Sys.setenv(TENSORFLOW_PYTHON="/usr/bin/python")
library(tensorflow)

########################  Setting up H2O  #########################

install.packages("h2o", dependencies = T)
library(h2o) 
localH2O = h2o.init()

localH2O = h2o.init(ip = "localhost", port = 54321, nthreads = -1)   








