###########  TEXT to VECTOR IMPLEMENTATION  ###########
library(text2vec)
library(glmnet)

# Get movie_review dataset
data("movie_review")

## Logistic Regression model
logistic_model <- function(Xtrain,Ytrain,Xtest,Ytest){
  classifier <- cv.glmnet(x=Xtrain, y=Ytrain,
                                family="binomial", alpha=1, type.measure = "auc",
                                nfolds = 5, maxit = 1000)
  plot(classifier)
  vocab_test_pred <- predict(classifier, Xtest, type = "response")
  return(cat("Train AUC : ", round(max(classifier$cvm), 4),
         "Test AUC : ",glmnet:::auc(Ytest, vocab_test_pred),"\n"))
}


# Split the data into train and test in 80:20 ratio
train_samples <- caret::createDataPartition(c(1:length(labels[,1])),p = 0.8)$Resample1
train_movie <- movie_review[train_samples,]
test_movie <- movie_review[-train_samples,]

##### Generate DTM using vocabulary (or dictionary words)
# perform tokenization of train and test data

train_tokens <- train_movie$review %>% tolower %>% word_tokenizer
test_tokens <- test_movie$review %>% tolower %>% word_tokenizer

vocab_train <- create_vocabulary(itoken(train_tokens,ids=train$id,progressbar = FALSE))

# Create train and test DTMs
vocab_train_dtm <- create_dtm(it = itoken(train_tokens,ids=train$id,progressbar = FALSE),
                              vectorizer = vocab_vectorizer(vocab_train))
vocab_test_dtm <- create_dtm(it = itoken(test_tokens,ids=test$id,progressbar = FALSE),
                              vectorizer = vocab_vectorizer(vocab_train))

dim(vocab_train_dtm)
dim(vocab_test_dtm)

# Run LASSO (L1 norm) Logistic Regression
logistic_model(Xtrain = vocab_train_dtm,
               Ytrain = train_movie$sentiment,
               Xtest = vocab_test_dtm,
               Ytest = test_movie$sentiment)

#### PRUNING Vocabulary using stop words
data("stop_words")

vocab_train_prune <- create_vocabulary(itoken(train_tokens,ids=train$id,progressbar = FALSE),
                                       stopwords = stop_words$word)

vocab_train_prune <- prune_vocabulary(vocab_train_prune,term_count_min = 15,
                                      doc_proportion_min = 0.0005,
                                      doc_proportion_max = 0.5)

vocab_train_prune_dtm <- create_dtm(it = itoken(train_tokens,ids=train$id,progressbar = FALSE),
                              vectorizer = vocab_vectorizer(vocab_train_prune))
vocab_test_prune_dtm <- create_dtm(it = itoken(test_tokens,ids=test$id,progressbar = FALSE),
                             vectorizer = vocab_vectorizer(vocab_train_prune))

logistic_model(Xtrain = vocab_train_prune_dtm,
               Ytrain = train_movie$sentiment,
               Xtest = vocab_test_prune_dtm,
               Ytest = test_movie$sentiment)

####### Use N Grams
vocab_train_ngrams <- create_vocabulary(itoken(train_tokens,ids=train$id,progressbar = FALSE),
                                        ngram = c(1L, 2L))

vocab_train_ngrams <- prune_vocabulary(vocab_train_ngrams,term_count_min = 10,
                                       doc_proportion_min = 0.0005,
                                       doc_proportion_max = 0.5)

vocab_train_ngrams_dtm <- create_dtm(it = itoken(train_tokens,ids=train$id,progressbar = FALSE),
                                    vectorizer = vocab_vectorizer(vocab_train_ngrams))
vocab_test_ngrams_dtm <- create_dtm(it = itoken(test_tokens,ids=test$id,progressbar = FALSE),
                                   vectorizer = vocab_vectorizer(vocab_train_ngrams))


logistic_model(Xtrain = vocab_train_ngrams_dtm,
               Ytrain = train_movie$sentiment,
               Xtest = vocab_test_ngrams_dtm,
               Ytest = test_movie$sentiment)

#### FEATURE HASHING

vocab_train_hashing_dtm <- create_dtm(it = itoken(train_tokens,ids=train$id,progressbar = FALSE),
                                      vectorizer = hash_vectorizer(hash_size = 2^14, ngram = c(1L, 2L)))
vocab_test_hashing_dtm <- create_dtm(it = itoken(test_tokens,ids=test$id,progressbar = FALSE),
                                    vectorizer = hash_vectorizer(hash_size = 2^14, ngram = c(1L, 2L)))

logistic_model(Xtrain = vocab_train_hashing_dtm,
               Ytrain = train_movie$sentiment,
               Xtest = vocab_test_hashing_dtm,
               Ytest = test_movie$sentiment)

#### TF-IDF

vocab_train_tfidf <- fit_transform(vocab_train_dtm, TfIdf$new())
vocab_test_tfidf <- fit_transform(vocab_test_dtm, TfIdf$new())

logistic_model(Xtrain = vocab_train_tfidf,
               Ytrain = train_movie$sentiment,
               Xtest = vocab_test_tfidf,
               Ytest = test_movie$sentiment)

