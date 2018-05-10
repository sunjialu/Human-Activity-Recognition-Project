# This file contains the code of training the SVM Model.
# The training process is:
#   import the data set
#




library(randomForest)
library(gmodels)
library(neuralnet)
library(RSNNS)
library(Rcpp)
library(lattice)
library(ggplot2)
library(ggfortify)
library(caret)
library(cluster)
library(Rtsne)
library(e1071)
library(kernlab)
library(MASS)
set.seed(123)

# import the data set
# setwd("./Dataset/UCIHARDataset/")
train_data<-read.table("./Dataset/UCIHARDataset/train/X_train.txt")
train_lables<-read.table("./Dataset/UCIHARDataset/train/Y_train.txt")

test_data<-read.table("./Dataset/UCIHARDataset/test/X_test.txt")
test_lables<-read.table("./Dataset/UCIHARDataset/test/Y_test.txt")

# read in the feature names correspond to each column
col_names <- readLines("./Dataset/UCIHARDataset/features.txt")
colnames(train_data)<-make.names(col_names)
colnames(test_data)<-make.names(col_names)
colnames(train_lables)<-"lable"
colnames(test_lables)<-"lable"

# combine the training data and testing data together to form a final data set to be used later
train_final<-cbind(train_lables,train_data)
test_final<-cbind(test_lables,test_data)
final_data<-rbind(train_final,test_final)
final_data$lable<-factor(final_data$lable)



library(stats)

# Do the feature extraction using PCA technique

model_pca <- prcomp(train_final[-1], center=TRUE, tol=0.02)
newfeature_train <- as.matrix(predict(model_pca, train_final))
newfeature_test <- as.matrix(predict(model_pca, test_final))
Train_final <- cbind(train_lables, newfeature_train)
Test_final <- cbind(test_lables, newfeature_test)
Train_final$lable<-factor(Train_final$lable)
Test_final$lable<-factor(Test_final$lable)



# Support vector machine
# 1. linear kernel

# tune.model = tune(svm,
#                   activity~.,
#                   data=train_final,
#                   kernel="linear", # linear kernel
#                   range=list(cost=10^(-1:2)) 
# )
# 
# summary(tune.model)
# plot(tune.model)
# tune.model$best.model

# train the SVM with linear kernel, the parameter has be determined during the search procedure above.
fit_svm_1 <- svm(lable~., data=Train_final, kernel="linear",
               cost = 100, cross=7)

# training set accuracy
train.pd <- predict(fit_svm_1, Train_final[,-1])
table(unlist(train_lables), train.pd)
confus.matrix = table(truth=Train_final$lable, predict=train.pd)
confus.matrix
print(sum(diag(confus.matrix))/sum(confus.matrix))

# testing set accuracy
test.pd <-predict(fit_svm_1, Test_final[,-1])
table(unlist(test_lables), test.pd)
confus.matrix = table(truth=Test_final$lable, predict=test.pd)
confus.matrix
print(sum(diag(confus.matrix))/sum(confus.matrix))




# 2. RBF Gaussian kernel

# tune.model.RBF = tune(svm,
#                   lable~.,
#                   data=Train_final[1:2000,],
#                   kernel="radial", # radial-basis kernel
#                   range=list(cost=10^(0:2), gamma=c(0.0001,0.005,0.001,0.05))
# )
# 
# summary(tune.model.RBF)
# plot(tune.model.RBF)
# tune.model.RBF$best.model

# train the SVM with RBF gaussian kernel, the parameter is given by the tuning procedure shown above.
fit_svm_2 <- svm(lable~., data=Train_final, kernel="radial",
                 cost=10, gamma=0.005 , cross=7)

# training accuracy
train.pd <- predict(fit_svm_2, Train_final[,-1])
table(unlist(train_lables), train.pd)
confus.matrix = table(real=Train_final$lable, predict=train.pd)
print(sum(diag(confus.matrix))/sum(confus.matrix))

# testing accuracy
test.pd <- predict(fit_svm_2, Test_final[,-1])
table(unlist(test_lables), test.pd)
confus.matrix = table(truth=Test_final$lable, predict=test.pd)
confus.matrix
print(sum(diag(confus.matrix))/sum(confus.matrix))


###########################################################################
#
#                         tSNE
#
############################################################################

# perform t-SNE
require(Rtsne)

tsne <- Rtsne(as.matrix(final_data[,-1]), check_duplicates = FALSE, pca = FALSE, perplexity=30, theta=0.5, dims=2)

# add the t-SNE map coordinates as new columns in the full dataset
all <- cbind(final_data[,1], final_data[,-1], tsne$Y)

# now re-split into trn and tst
trn <- all[1:7800,]
colnames(trn)[1] <- "lable"
tst <- all[7801:10299,]
colnames(tst)[1] <- "lable"

# now the target label is in column 1 and columns 2:561 contain the input variables
x1 <- trn[,-1]
y1 <- trn[,1]

x2 <- tst[,-1]
y2 <- tst[,1]

# having unnamed input columns, gives an error while training a random forest
colnames(x1)[562:563] <- c("TSNEx", "TSNEy")
colnames(x2)[562:563] <- c("TSNEx", "TSNEy")


# train the tSNE-SVM model, same as with PCA extracted data set
fit_svm_tsne1 <- svm(lable~., data=trn, kernel="linear",
                 cost = 100, cross=7)

# training accuracy
train.pd <- predict(fit_svm_tsne1, trn[,-1])
table(unlist(trn[,1]), train.pd)
confus.matrix = table(real=trn$lable, predict=train.pd)
print(sum(diag(confus.matrix))/sum(confus.matrix))

# testing accuracy
test.pd <- predict(fit_svm_tsne1, tst[,-1])
table(unlist(tst[,1]), test.pd)
confus.matrix = table(truth=tst$lable, predict=test.pd)
confus.matrix
print(sum(diag(confus.matrix))/sum(confus.matrix))

# train the tSNE-SVM model with RBF kernel
fit_svm_tsne2 <- svm(lable~., data=trn, kernel="radial",
                 cost=10, gamma=0.005 , cross=7)


# train accuracy
train.pd <- predict(fit_svm_tsne2, trn[,-1])
table(unlist(trn[,1]), train.pd)
confus.matrix = table(real=trn$lable, predict=train.pd)
print(sum(diag(confus.matrix))/sum(confus.matrix))

# test accuracy
test.pd <- predict(fit_svm_tsne2, tst[,-1])
table(unlist(tst[,1]), test.pd)
confus.matrix = table(truth=tst$lable, predict=test.pd)
confus.matrix
print(sum(diag(confus.matrix))/sum(confus.matrix))



