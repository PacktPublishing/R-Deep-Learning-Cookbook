############## Setting-up Transfer Learning Script

require(tensorflow)
np<-import("numpy")

# Import slim from contrib libraty of tensorflow
slim = tf$contrib$slim

# Reset tensorflow Graph
tf$reset_default_graph()


# Resizing the images
input.img = tf$placeholder(tf$float32, shape(NULL, NULL, NULL, 3))
scaled.img = tf$image$resize_images(input.img, shape(224,224))

# Define VGG16 network
library(magrittr)
VGG16.model<-function(slim, input.image){
  vgg16.network = slim$conv2d(input.image, 64, shape(3,3), scope='vgg_16/conv1/conv1_1') %>%
    slim$conv2d(64, shape(3,3), scope='vgg_16/conv1/conv1_2')  %>%
    slim$max_pool2d( shape(2, 2), scope='vgg_16/pool1')  %>%

    slim$conv2d(128, shape(3,3), scope='vgg_16/conv2/conv2_1')  %>%
    slim$conv2d(128, shape(3,3), scope='vgg_16/conv2/conv2_2')  %>%
    slim$max_pool2d( shape(2, 2), scope='vgg_16/pool2')  %>%

    slim$conv2d(256, shape(3,3), scope='vgg_16/conv3/conv3_1')  %>%
    slim$conv2d(256, shape(3,3), scope='vgg_16/conv3/conv3_2')  %>%
    slim$conv2d(256, shape(3,3), scope='vgg_16/conv3/conv3_3')  %>%
    slim$max_pool2d(shape(2, 2), scope='vgg_16/pool3')  %>%

    slim$conv2d(512, shape(3,3), scope='vgg_16/conv4/conv4_1')  %>%
    slim$conv2d(512, shape(3,3), scope='vgg_16/conv4/conv4_2')  %>%
    slim$conv2d(512, shape(3,3), scope='vgg_16/conv4/conv4_3')  %>%
    slim$max_pool2d(shape(2, 2), scope='vgg_16/pool4')  %>%

    slim$conv2d(512, shape(3,3), scope='vgg_16/conv5/conv5_1')  %>%
    slim$conv2d(512, shape(3,3), scope='vgg_16/conv5/conv5_2')  %>%
    slim$conv2d(512, shape(3,3), scope='vgg_16/conv5/conv5_3')  %>%
    slim$max_pool2d(shape(2, 2), scope='vgg_16/pool5')  %>%

    slim$conv2d(4096, shape(7, 7), padding='VALID', scope='vgg_16/fc6')  %>%
    slim$conv2d(4096, shape(1, 1), scope='vgg_16/fc7') %>%

    slim$conv2d(1000, shape(1, 1), scope='vgg_16/fc8')  %>%
    tf$squeeze(shape(1, 2), name='vgg_16/fc8/squeezed')
  return(vgg16.network)
}

vgg16.network<-VGG16.model(slim, input.image = scaled.img)

# Restore the weights
restorer = tf$train$Saver()
sess = tf$Session()
restorer$restore(sess, 'vgg_16.ckpt')

### Load initial layer
WEIGHTS_PATH<-'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
load_weights<-function(sess){
  weights_dict = np$load(WEIGHTS_PATH, encoding = 'bytes')
}

# Evaluating using VGG16 network
require(jpeg)
testImgURL<-"http://farm4.static.flickr.com/3155/2591264041_273abea408.jpg"
img.test<-tempfile()
download.file(testImgURL,img.test, mode="wb")
read.image <- readJPEG(img.test)
file.remove(img.test) # cleanup

## Evaluate
size = dim(read.image)
imgs = array(255*read.image, dim = c(1, size[1], size[2], size[3]))
VGG16_eval = sess$run(vgg16.network, dict(images = imgs))
probs = exp(VGG16_eval)/sum(exp(VGG16_eval)) # 672: 'mountain bike, all-terrain bike, off-roader',



