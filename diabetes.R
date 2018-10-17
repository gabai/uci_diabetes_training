library(keras)
library(magrittr)
library(mlbench)
library(dplyr)
library(neuralnet)

temp <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip", temp)
data <- read.csv(unz(temp, "dataset_diabetes/diabetic_data.csv"), header = T, na.strings = c('?','None'))
unlink(temp)

data %<>% mutate_if(is.factor, as.numeric)
data <- data[, c(3:5, 10, 22:24, 48:50)]
data <- na.omit(data)
data <- as.matrix(data)
dimnames(data) <- NULL
summary(data)

data[, 1:9] <- normalize(data[, 1:9])
data[, 10] <- as.numeric(data[, 10]) -1
summary(data)

set.seed(7)
ind <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
training <- data[ind==1, 1:9]
test <- data[ind==2, 1:9]
trainingtarget <- data[ind==1,10]
testtarget <- data[ind==2, 10]

trainlabel <- to_categorical(trainingtarget)
testlabel <- to_categorical(testtarget)
print(trainlabel)

# Model
model <- keras_model_sequential()

model %>%
  layer_dense(units = 50, activation = 'relu', input_shape = c(9)) %>%
  layer_dense(units = 25, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')

# Compile
model %>%
  compile(loss = 'categorical_crossentropy',
          optimizer = 'adam',
          metrics = 'accuracy')

# Fit Model
history <- model %>%
  fit(training, 
      trainlabel,
      epoch = 200,
      batch_size = 32,
      validation_split = 0.2)

plot(history)

#Evaluate model
model %>%
  evaluate(test,
           testlabel)

#Predictions
prob <- model %>%
  predict_proba(test)

pred <- model %>%
  predict_classes(test)

table(Predicted = pred, Actual = testtarget)

cbind(prob, pred, testtarget)
