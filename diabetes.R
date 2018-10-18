library(keras)
library(magrittr)
library(mlbench)
library(dplyr)
library(neuralnet)

#Get data
temp <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip", temp)
data <- read.csv(unz(temp, "dataset_diabetes/diabetic_data.csv"), header = T, na.strings = c('?','None'))
unlink(temp)

#Modify data
data <- data[, c(3:5, 10, 15, 22, 48:50)]

data[,"change"] <- ifelse(data[,"change"] =="No", 0,
                          ifelse(data[,"change"] =="Ch", 1, 99))

data[, "diabetesMed"] <- ifelse(data[,"diabetesMed"] =="No", 0,
                          ifelse(data[,"diabetesMed"] =="Yes", 1, 99))

data[,"readmitted"] <- ifelse(data[,"readmitted"] =="NO", 0,
                              ifelse(data[, "readmitted"] =="<30", 1,
                                     ifelse(data[,"readmitted"] == ">30", 2, 99)))

data[,7] <- as.numeric(data[, 7])
data[,8] <- as.numeric(data[, 8])
data[,9] <- as.numeric(data[, 9])

data %<>% mutate_if(is.factor, as.numeric)
data <- data[complete.cases(data), ]
data <- as.matrix(data)
dimnames(data) <- NULL
summary(data)

data[, 1:8] <- normalize(data[, 1:8])
data[, 9] <- as.numeric(data[, 9])
summary(data)

#Set random seed
set.seed(7)

#Split data 70/30 for training and testing
ind <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
training <- data[ind==1, 1:8]
test <- data[ind==2, 1:8]
trainingtarget <- data[ind==1,9]
testtarget <- data[ind==2, 9]
print(trainingtarget)
print(testtarget)

trainlabel <- to_categorical(trainingtarget)
testlabel <- to_categorical(testtarget)
print(trainlabel)
print(testlabel)

# Model
model <- keras_model_sequential()

model %>%
  layer_dense(units = 50, activation = 'relu', input_shape = c(8)) %>%
  layer_dense(units = 25, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'relu') %>%
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

#plot(history)

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
