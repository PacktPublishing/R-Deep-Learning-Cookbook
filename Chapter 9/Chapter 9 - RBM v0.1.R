setwd("/set/your/path/")

##################  LOAD LIBRARIES
library(reticulate)
use_condaenv("python27")
midi <- import_from_path("midi",path="C:/ProgramData/Anaconda2/Lib/site-packages")
np <- import("numpy")
msgpack <- import_from_path("msgpack",path="C:/ProgramData/Anaconda2/Lib/site-packages")
psys <- import("sys")
tqdm <- import_from_path("tqdm",path="C:/ProgramData/Anaconda2/Lib/site-packages")
midi_manipulation_updated <- import_from_path("midi_manipulation_updated",path="C:/Music_RBM")
glob <- import("glob")
library(tensorflow)

#############   FUNCTION TO READ MIDI SONGS
get_input_songs <- function(path){
  files = glob$glob(paste0(path,"/*mid*"))
  songs <- list()
  count <- 1
  for(f in files){
    songs[[count]] <- np$array(midi_manipulation_updated$midiToNoteStateMatrix(f))
    count <- count+1
  }
  return(songs)
}

path <- 'Pop_Music_Midi'
input_songs <- get_input_songs(path)

############  Parameters
# Initialise parameters
lowest_note = 24L
highest_note = 102L
note_range = highest_note-lowest_note 

num_timesteps  = 15L
num_input      = 2L*note_range*num_timesteps
num_hidden       = 50L

alpha<-0.1

############## RBM MODEL (from chapter 5)
trainX <- do.call(rbind,input_songs)
# Placeholder variables
vb <- tf$placeholder(tf$float32, shape = shape(num_input))
hb <- tf$placeholder(tf$float32, shape = shape(num_hidden))
W <- tf$placeholder(tf$float32, shape = shape(num_input, num_hidden))

# Phase 1 : Forward Pass
X = tf$placeholder(tf$float32, shape=shape(NULL, num_input))
prob_h0= tf$nn$sigmoid(tf$matmul(X, W) + hb)  
h0 = tf$nn$relu(tf$sign(prob_h0 - tf$random_uniform(tf$shape(prob_h0)))) 

# Phase 2 : Backward Pass
prob_v1 = tf$matmul(h0, tf$transpose(W)) + vb
v1 = prob_v1 + tf$random_normal(tf$shape(prob_v1), mean=0.0, stddev=1.0, dtype=tf$float32)
h1 = tf$nn$sigmoid(tf$matmul(v1, W) + hb)    

# Calculate gradients
w_pos_grad = tf$matmul(tf$transpose(X), h0)
w_neg_grad = tf$matmul(tf$transpose(v1), h1)
CD = (w_pos_grad - w_neg_grad) / tf$to_float(tf$shape(X)[0])
update_w = W + alpha * CD
update_vb = vb + alpha * tf$reduce_mean(X - v1)
update_hb = hb + alpha * tf$reduce_mean(h0 - h1)

# Objective function
err = tf$reduce_mean(tf$square(X - v1))

# Initialise variables
cur_w = tf$Variable(tf$zeros(shape = shape(num_input, num_hidden), dtype=tf$float32))
cur_vb = tf$Variable(tf$zeros(shape = shape(num_input), dtype=tf$float32))
cur_hb = tf$Variable(tf$zeros(shape = shape(num_hidden), dtype=tf$float32))
prv_w = tf$Variable(tf$random_normal(shape=shape(num_input, num_hidden), stddev=0.01, dtype=tf$float32))
prv_vb = tf$Variable(tf$zeros(shape = shape(num_input), dtype=tf$float32))
prv_hb = tf$Variable(tf$zeros(shape = shape(num_hidden), dtype=tf$float32)) 

# Start tensorflow session
sess$run(tf$global_variables_initializer())
song = np$array(trainX)
song = song[1:(np$floor(dim(song)[1]/num_timesteps)*num_timesteps),]
song = np$reshape(song, newshape=shape(dim(song)[1]/num_timesteps, dim(song)[2]*num_timesteps))
output <- sess$run(list(update_w, update_vb, update_hb), feed_dict = dict(X=song,
                                                                          W = prv_w$eval(),
                                                                          vb = prv_vb$eval(),
                                                                          hb = prv_hb$eval()))
prv_w <- output[[1]] 
prv_vb <- output[[2]]
prv_hb <-  output[[3]]
sess$run(err, feed_dict=dict(X= song, W= prv_w, vb= prv_vb, hb= prv_hb))

epochs=200
errors <- list()
weights <- list()
u=1
for(ep in 1:epochs){
  for(i in seq(0,(dim(song)[1]-100),100)){
    batchX <- song[(i+1):(i+100),]
    output <- sess$run(list(update_w, update_vb, update_hb), feed_dict = dict(X=batchX,
                                                                              W = prv_w,
                                                                              vb = prv_vb,
                                                                              hb = prv_hb))
    prv_w <- output[[1]] 
    prv_vb <- output[[2]]
    prv_hb <-  output[[3]]
    if(i%%500 == 0){
      errors[[u]] <- sess$run(err, feed_dict=dict(X= song, W= prv_w, vb= prv_vb, hb= prv_hb))
      weights[[u]] <- output[[1]]
      u <- u+1
      cat(i , " : ")
    }
  }
  cat("epoch :", ep, " : reconstruction error : ", errors[length(errors)][[1]],"\n")
}

############# Regenrate sample music notes
hh0 = tf$nn$sigmoid(tf$matmul(X, W) + hb)
vv1 = tf$nn$sigmoid(tf$matmul(hh0, tf$transpose(W)) + vb)
feed = sess$run(hh0, feed_dict=dict( X= sample_image, W= prv_w, hb= prv_hb))
rec = sess$run(vv1, feed_dict=dict( hh0= feed, W= prv_w, vb= prv_vb))

S = np$reshape(rec[1,],newshape=shape(num_timesteps,2*note_range))

midi_manipulation$noteStateMatrixToMidi(S, name=paste0("generated_chord_1"))
generated_chord_1


