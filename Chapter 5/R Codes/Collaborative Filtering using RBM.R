###########   COLLABORATIVE FILTERING WITH RBM
setwd("Set the working directory with movies.dat and ratings.dat files")

## Read movie lens data
txt <- readLines("movies.dat", encoding = "latin1")
txt_split <- lapply(strsplit(txt, "::"), function(x) as.data.frame(t(x), stringsAsFactors=FALSE))
movies_df <- do.call(rbind, txt_split)
names(movies_df) <- c("MovieID", "Title", "Genres")
movies_df$MovieID <- as.numeric(movies_df$MovieID)
movies_df$id_order <- 1:nrow(movies_df)

ratings_df <- read.table("ratings.dat", sep=":",header=FALSE,stringsAsFactors = F)
ratings_df <- ratings_df[,c(1,3,5,7)]
colnames(ratings_df) <- c("UserID","MovieID","Rating","Timestamp")

# Merge user ratings and movies
merged_df <- merge(movies_df, ratings_df, by="MovieID",all=FALSE)

# Remove unnecessary columns
merged_df[,c("Timestamp","Title","Genres")] <- NULL

# create % rating
merged_df$rating_per <- merged_df$Rating/5

# Generate a matrix of ratings
num_of_users <- 1000
num_of_movies <- length(unique(movies_df$MovieID))
trX <- matrix(0,nrow=num_of_users,ncol=num_of_movies)
for(i in 1:num_of_users){
  merged_df_user <- merged_df[merged_df$UserID %in% i,]
  trX[i,merged_df_user$id_order] <- merged_df_user$rating_per
}

# Import tenforflow libraries
# Sys.setenv(TENSORFLOW_PYTHON="C:/PROGRA~1/Python35/python.exe")
# Sys.setenv(TENSORFLOW_PYTHON_VERSION = 3)
library(tensorflow)
np <- import("numpy")

# Create TensorFlow session
# Reset the graph
tf$reset_default_graph()
# Starting session as interactive session
sess <- tf$InteractiveSession()

# Model Parameters
  num_hidden = 20
  num_input = nrow(movies_df)
  vb <- tf$placeholder(tf$float32, shape = shape(num_input))    #Number of unique movies
  hb <- tf$placeholder(tf$float32, shape = shape(num_hidden))   #Number of features we're going to learn
  W <- tf$placeholder(tf$float32, shape = shape(num_input, num_hidden))
  
#Phase 1: Input Processing
v0 = tf$placeholder(tf$float32,shape= shape(NULL, num_input))
prob_h0= tf$nn$sigmoid(tf$matmul(v0, W) + hb)
h0 = tf$nn$relu(tf$sign(prob_h0 - tf$random_uniform(tf$shape(prob_h0))))
#Phase 2: Reconstruction
prob_v1 = tf$nn$sigmoid(tf$matmul(h0, tf$transpose(W)) + vb) 
v1 = tf$nn$relu(tf$sign(prob_v1 - tf$random_uniform(tf$shape(prob_v1))))
h1 = tf$nn$sigmoid(tf$matmul(v1, W) + hb)

# RBM Parameters and functions
#Learning rate
alpha = 1.0
#Create the gradients
w_pos_grad = tf$matmul(tf$transpose(v0), h0)
w_neg_grad = tf$matmul(tf$transpose(v1), h1)
#Calculate the Contrastive Divergence to maximize
CD = (w_pos_grad - w_neg_grad) / tf$to_float(tf$shape(v0)[1])
#Create methods to update the weights and biases
update_w = W + alpha * CD
update_vb = vb + alpha * tf$reduce_mean(v0 - v1)
update_hb = hb + alpha * tf$reduce_mean(h0 - h1)

# Mean Absolute Error Function.
err = v0 - v1
err_sum = tf$reduce_mean(err * err)

# Initialise variables (current and previous)
cur_w = tf$Variable(tf$zeros(shape = shape(num_input, num_hidden), dtype=tf$float32))
cur_vb = tf$Variable(tf$zeros(shape = shape(num_input), dtype=tf$float32))
cur_hb = tf$Variable(tf$zeros(shape = shape(num_hidden), dtype=tf$float32))
prv_w = tf$Variable(tf$random_normal(shape=shape(num_input, num_hidden), stddev=0.01, dtype=tf$float32))
prv_vb = tf$Variable(tf$zeros(shape = shape(num_input), dtype=tf$float32))
prv_hb = tf$Variable(tf$zeros(shape = shape(num_hidden), dtype=tf$float32)) 

# Start tensorflow session
sess$run(tf$global_variables_initializer())
output <- sess$run(list(update_w, update_vb, update_hb), feed_dict = dict(v0=trX,
                                                                          W = prv_w$eval(),
                                                                          vb = prv_vb$eval(),
                                                                          hb = prv_hb$eval()))
prv_w <- output[[1]] 
prv_vb <- output[[2]]
prv_hb <-  output[[3]]
sess$run(err_sum, feed_dict=dict(v0=trX, W= prv_w, vb= prv_vb, hb= prv_hb))

# Train RBM
epochs= 500
errors <- list()
weights <- list()

for(ep in 1:epochs){
  for(i in seq(0,(dim(trX)[1]-100),100)){
    batchX <- trX[(i+1):(i+100),]
    output <- sess$run(list(update_w, update_vb, update_hb), feed_dict = dict(v0=batchX,
                                                                              W = prv_w,
                                                                              vb = prv_vb,
                                                                              hb = prv_hb))
    prv_w <- output[[1]] 
    prv_vb <- output[[2]]
    prv_hb <-  output[[3]]
    if(i%%1000 == 0){
      errors <- c(errors,sess$run(err_sum, feed_dict=dict(v0=batchX, W= prv_w, vb= prv_vb, hb= prv_hb)))
      weights <- c(weights,output[[1]])
      cat(i , " : ")
    }
  }
  cat("epoch :", ep, " : reconstruction error : ", errors[length(errors)][[1]],"\n")
}

# Plot reconstruction error
error_vec <- unlist(errors)
plot(error_vec,xlab="# of batches",ylab="mean squared reconstruction error",main="RBM-Reconstruction MSE plot")

# Recommendation
#Selecting the input user
inputUser = as.matrix(t(trX[75,]))
names(inputUser) <- movies_df$id_order

# Remove the movies not watched yet
inputUser <- inputUser[inputUser>0]

# Plot the top genre movies
top_rated_movies <- movies_df[as.numeric(names(inputUser)[order(inputUser,decreasing = TRUE)]),]$Title
top_rated_genres <- movies_df[as.numeric(names(inputUser)[order(inputUser,decreasing = TRUE)]),]$Genres
top_rated_genres <- as.data.frame(top_rated_genres,stringsAsFactors=F)
top_rated_genres$count <- 1
top_rated_genres <- aggregate(count~top_rated_genres,FUN=sum,data=top_rated_genres)
top_rated_genres <- top_rated_genres[with(top_rated_genres, order(-count)), ]
top_rated_genres$top_rated_genres <- factor(top_rated_genres$top_rated_genres, levels = top_rated_genres$top_rated_genres)
ggplot(top_rated_genres[top_rated_genres$count>1,],aes(x=top_rated_genres,y=count))+
  geom_bar(stat="identity")+ 
  theme_bw()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  labs(x="Genres",y="count",title="Top Rated Genres")+
  theme(plot.title = element_text(hjust = 0.5))
  

#Feeding in the user and reconstructing the input
hh0 = tf$nn$sigmoid(tf$matmul(v0, W) + hb)
vv1 = tf$nn$sigmoid(tf$matmul(hh0, tf$transpose(W)) + vb)
feed = sess$run(hh0, feed_dict=dict( v0= inputUser, W= prv_w, hb= prv_hb))
rec = sess$run(vv1, feed_dict=dict( hh0= feed, W= prv_w, vb= prv_vb))
names(rec) <- movies_df$id_order

# Select all recommended movies
top_recom_movies <- movies_df[as.numeric(names(rec)[order(rec,decreasing = TRUE)]),]$Title[1:10]
top_recom_genres <- movies_df[as.numeric(names(rec)[order(rec,decreasing = TRUE)]),]$Genres
top_recom_genres <- as.data.frame(top_recom_genres,stringsAsFactors=F)
top_recom_genres$count <- 1
top_recom_genres <- aggregate(count~top_recom_genres,FUN=sum,data=top_recom_genres)
top_recom_genres <- top_recom_genres[with(top_recom_genres, order(-count)), ]
top_recom_genres$top_recom_genres <- factor(top_recom_genres$top_recom_genres, levels = top_recom_genres$top_recom_genres)
ggplot(top_recom_genres[top_recom_genres$count>20,],aes(x=top_recom_genres,y=count))+
  geom_bar(stat="identity")+ 
  theme_bw()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  labs(x="Genres",y="count",title="Top Recommended Genres")+
  theme(plot.title = element_text(hjust = 0.5))




