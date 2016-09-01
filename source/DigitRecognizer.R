#########################################################################################
#####       # PREPARE FOR PACKAGE EBImage or Imager and MXNet                       #####
#####-------------------------------------------------------------------------------#####
#####       # for library EBImage                                                   #####
#####       ### start R and run following command                                   #####
#####       ### try http:// if https:// URLs are not supported                      #####
#####       ### source("https://bioconductor.org/biocLite.R")                       #####
#####       ### biocLite("EBImage")                                                 #####
#####                                                                               #####
#####       # for library MXNet                                                     #####
#####       ### following instruction in this page:                                 #####
#####       ### https://github.com/dmlc/mxnet/blob/master/docs/how_to/build.md      #####
#####       ### using "R Package Installation" section                              #####
#########################################################################################

# BUILD MODEL
#------------------------------------------------------------------------------
# Clean workspace
rm(list=ls())

# Setup env
#------------------------------------------------------------------------------
# Set configuration
DATA_DIR <- "/run/media/nguyenkien/5f4bf705-1619-444c-83ba-6739b83e1bf7/kaggle/Digit-Recognizer"
IMAGES_TRAIN_DATA <- paste(DATA_DIR, "train.csv", sep="/")
IMAGES_TEST_TRAIN_DATA <- paste(DATA_DIR, "train_train.csv", sep="/")
IMAGES_TEST_TEST_DATA <- paste(DATA_DIR, "train_test.csv", sep="/")
IMAGES_PREDICT_DATA <- paste(DATA_DIR, "test.csv", sep="/")
SOURCE_CODE_DIR <- "/home/nguyenkien/kaggle/Digit-Recognizer"

#########################################################################################
# Set running mode
#------------------------------------------------------------------------------
test_mode = TRUE
#########################################################################################

# Load MXNet
require(mxnet)

# Clearup output dir
#if (length(list.files(IMAGES_OUT_DIR)) > 0){
#  file.remove(paste(IMAGES_OUT_DIR,list.files(IMAGES_OUT_DIR),sep="/"))
#}

# Loading data and set up
#-------------------------------------------------------------------------------
# Data preparation
if (test_mode){
  print("###########################   TEST-MODE   ###########################")
  IMAGES_TRAIN_DATA <- IMAGES_TEST_TRAIN_DATA
  IMAGES_TEST_DATA <- IMAGES_TEST_TEST_DATA
} else {
  print("###########################   PREDICT-MODE   ###########################")
}
train <- read.csv(IMAGES_TRAIN_DATA)
train <- data.matrix(train)
train.x <- train[,-1]
train.y <- train[,1]
train.x <- t(train.x/255)

# Calculate ConvNet
#------------------------------------------------------------------------------
# Build Model
source(paste(SOURCE_CODE_DIR, "ConvNet.R", sep="/"))
# Calculate time collapse
tic <- proc.time()
conv_net_model <- buildConvNetModel(use_fc_layer = TRUE, 
                                    train_data = train.x, 
                                    train_label = train.y)
print(proc.time() - tic)

if (test_mode){
  # Testing
  #-------------------------------------------------------------------------------
  # Prepare testing-data
  test <- read.csv(IMAGES_TEST_DATA)
  test <- data.matrix(test)
  test.y <- test[,1]
  test.x <- test[,-1]
  test.x <- t(test.x/255)
  test.array <- test.x
  dim(test.array) <- c(28, 28, 1, ncol(test.x))
  # Predict labels
  predicted <- predict(conv_net_model, test.array)
  # Assign labels
  predicted_labels <- max.col(t(predicted)) - 1
  
  # Get accuracy 
  compare_label <- test.y == predicted_labels
  print(paste("Accuracy:", length(compare_label[compare_label == TRUE])/length(test.y), sep=" "))
} else{
  # Predict data set
  #------------------------------------------------------------------------------
  # Read data
  data <- read.csv(IMAGES_PREDICT_DATA)
  data.x <- data.matrix(data)
  data.x <- t(data.x/255)
  # Run prediction
  tic <- proc.time()
  predicted <- predictConvNet(model = conv_net_model, data = data.x)
  print(proc.time() - tic)
  write.table(x=predicted, file=paste(DATA_DIR,"predicted.csv",sep="/"), 
            col.names=c("ImageId","Label"), row.names=FALSE, sep=",", 
            dec=".", quote=FALSE)
}