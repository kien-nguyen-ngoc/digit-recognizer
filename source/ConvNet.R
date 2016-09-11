buildConvNetModel <- function(use_fc_layer=TRUE, train_data, train_label){
  # Set up the symbolic model
  #-------------------------------------------------------------------------------
  
  data <- mx.symbol.Variable('data')
  
  # 1st convolutional layer
  conv_1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 60) # 24x24x20
  tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "relu")
  pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2)) # 11x11x20
  dropout_1 <- mx.symbol.Dropout(data = pool_1, p = 0.3)
  # 2nd convolutional layer
  conv_2 <- mx.symbol.Convolution(data = dropout_1, kernel = c(5, 5), num_filter = 200) # 6x6x50
  tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "relu")
  pool_2 <- mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2)) # 2x2x50
  dropout_2 <- mx.symbol.Dropout(data = pool_2, p = 0.3)
  data <- dropout_2
  
  # Use full-connected layer
  if (use_fc_layer){
    # 1st fully connected layer
    flatten <- mx.symbol.Flatten(data = data)
    fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 800) # 1x1x500
    tanh_3 <- mx.symbol.Activation(data = fc_1, act_type = "relu")
    dropout_3 <- mx.symbol.Dropout(data = tanh_3, p = 0.3)
    # 2nd fully connected layer
    fc_2 <- mx.symbol.FullyConnected(data = dropout_3, num_hidden = 10) # 1x1x10
    data <- fc_2
  }
  
  # Output. Softmax output since we'd like to get some probabilities.
  NN_model <- mx.symbol.SoftmaxOutput(data = data)
  
  # Pre-training set up
  #-------------------------------------------------------------------------------
  
  # Set seed for reproducibility
  mx.set.seed(0)
  
  # Device used. CPU in my case.
  devices <- lapply(c(0:3), mx.cpu)
  
  # Training
  #-------------------------------------------------------------------------------
  train.array <- train_data
  dim(train.array) <- c(28, 28, 1, ncol(train_data))
  # Train the model
  model <- mx.model.FeedForward.create(NN_model,
                                       X = train.array,
                                       y = train_label,
                                       ctx = devices,
                                       num.round = 100,
                                       array.batch.size = 32,
                                       learning.rate = 0.001,
                                       momentum = 0.9,
                                       eval.metric = mx.metric.accuracy,
                                       initializer = mx.init.Xavier(factor_type="in", magnitude=1),
                                       epoch.end.callback = mx.callback.log.train.metric(100))
}

predictConvNet <- function(model, data){
  # Testing
  #-------------------------------------------------------------------------------
  dim(data) <- c(28, 28, 1, ncol(data))
  # Predict labels
  predicted <- predict(model, data)
  # Assign labels
  predicted_labels <- max.col(t(predicted)) - 1
  cbind(c(1:length(predicted_labels)), predicted_labels)
}

