# Load packages
require(mxnet)

# Function to load data as iterators
data.iterator <- function(data.shape, train.data, val.data, BATCHSIZE = 128) {

  # Load training data as iterator
  train <- mx.io.ImageRecordIter(
    path.imgrec = train.data,
    batch.size  = BATCHSIZE,
    data.shape  = data.shape,
    rand.crop   = TRUE,
    rand.mirror = TRUE)

  # Load validation data as iterator
  val <- mx.io.ImageRecordIter(
    path.imgrec = val.data,
    batch.size  = BATCHSIZE,
    data.shape  = data.shape,
    rand.crop   = FALSE,
    rand.mirror = FALSE
  )

  return(list(train = train, val = val))
}


# Load dataset
data  <- data.iterator(data.shape = c(224, 224, 3),
                       train.data = "data/pks.lst_train.rec",
                       val.data   = "data/pks.lst_val.rec",
                       BATCHSIZE = 8)
train <- data$train
val   <- data$val

inception_bn <- mx.model.load("/inception-bn/Inception-BN", iteration = 126)
symbol <- inception_bn$symbol

# Load model information
internals <- symbol$get.internals()
outputs <- internals$outputs
flatten <- internals$get.output(which(outputs == "flatten_output"))


# Define new layer
new_fc <- mx.symbol.FullyConnected(data = flatten,
                                   num_hidden = 2,
                                   name = "fc1")
new_soft <- mx.symbol.SoftmaxOutput(data = new_fc,
                                    name = "softmax")


# Re-initialize the weights for new layer
arg_params_new <- mxnet:::mx.model.init.params(
  symbol = new_soft,
  input.shape = c(224, 224, 3, 8),
  output.shape = NULL,
  initializer = mxnet:::mx.init.uniform(0.2),
  ctx = mx.cpu(0)
)$arg.params
fc1_weights_new <- arg_params_new[["fc1_weight"]]
fc1_bias_new <- arg_params_new[["fc1_bias"]]


arg_params_new <- inception_bn$arg.params
arg_params_new[["fc1_weight"]] <- fc1_weights_new
arg_params_new[["fc1_bias"]] <- fc1_bias_new


# Mode re-train
model <- mx.model.FeedForward.create(
  symbol             = new_soft,
  X                  = train,
  eval.data          = val,
  ctx                = mx.cpu(0),
  eval.metric        = mx.metric.accuracy,
  num.round          = 5,
  learning.rate      = 0.05,
  momentum           = 0.85,
  wd                 = 0.00001,
  kvstore            = "local",
  array.batch.size   = 128,
  epoch.end.callback = mx.callback.save.checkpoint("inception_bn"),
  batch.end.callback = mx.callback.log.train.metric(150),
  initializer        = mx.init.Xavier(factor_type = "in", magnitude = 2.34),
  optimizer          = "sgd",
  arg.params         = arg_params_new,
  aux.params         = inception_bn$aux.params
)
