library("rstudioapi")
setwd(dirname(getActiveDocumentContext()$path))
source('HP_Discretise.R')
library(splines)
library(hawkes)
library(keras)
library(tensorflow)
library(tfprobability)
K <- keras::backend()
if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()


predict_hawkes_parameters <- function(eta_mu_model, test_processes, 
                                      training_processes, training_parameters){
  
  eta_mu_predict <- predict(eta_mu_model, test_processes)
  eta_est <- median(eta_mu_predict[,1])
  mu_est <- median(eta_mu_predict[,2])
  
  min_eta <- eta_est - .05
  max_eta <- eta_est + .05
  eta <- training_parameters[,1]/training_parameters[,2]
  
  similar_eta_processes <- training_processes[eta > min_eta & eta<max_eta, ]
  similar_eta_alpha <- training_parameters[eta > min_eta & eta<max_eta, 1]
  similar_eta_mu <- training_parameters[eta > min_eta & eta < max_eta, 3]
  
  #similar_eta_avg_events_per_time <- avg_events_per_time[min_eta<eta & eta<max_eta]
  
  max_alpha <- ceiling(max(training_parameters[,1]))
  
  stat_data <- rep(NaN, max_alpha*20)
  for (i in 1:(max_alpha*20)){
    min_v <- (i-1)*.05
    max_v <- i*.05
    
    stat_data[i] <- mean( apply( matrix( similar_eta_processes[
      similar_eta_alpha >= min_v & similar_eta_alpha < max_v, ], ncol = dim(training_processes)[2]), 1, max))
  }
  glm_data <- data.frame(stat_data[(!is.nan(stat_data))],c((1:(max_alpha*20)*0.05 - .025))[(!is.nan(stat_data))])
  colnames(glm_data) <- c("x","y")
  lin_model <- glm(y~x, data = glm_data)
  stat <- median(apply(test_processes, 1, max))

  test_data_point <- data.frame(stat)
  colnames(test_data_point) <- c("x")
  alpha_est <- predict.glm(lin_model, test_data_point)[[1]]
  beta_est <- alpha_est/eta_est
  
  list("alpha_est" = alpha_est, "beta_est" = beta_est, "mu_est" = mu_est, "eta_est" = eta_est)
}





eta_mu_model <- function(number_of_training_processes, horizon, 
                         expected_activity, discretise_step = 1,
                         min_eta = .2, max_eta = .6, 
                         min_beta = 1, max_beta = 3){
# Initiate empty list to store raw event times
raw_training_processes <- list()

# The average number of events for each process is drawn from a N(450,10). This replicates the noise which may be seen in real data.
average_events <- rnorm(number_of_training_processes, expected_activity, 10)
avg_events_per_time <- average_events/horizon

eta <- runif(number_of_training_processes, min_eta, max_eta)
mu <- avg_events_per_time * (1-eta)
beta <- runif(number_of_training_processes, min_beta, max_beta)
alpha <- beta*eta

params <- matrix(c(alpha, beta, mu), nrow = number_of_training_processes)

# Simulate the data using the hawkes package.
for (i in 1:number_of_training_processes) {
  raw_training_processes[i] <- simulateHawkes(mu[i],alpha[i],beta[i],horizon)
}

# Discretise these processes
discretised_process <- discretise(raw_training_processes, discretise_step, horizon)


input <- discretised_process
epochs <- 500L
validation_split <- 0.05
batch_size <- dim(input)[1]*.1
input_dim <- dim(input)[2]
hidden_dim <- horizon


eta_mu_MLP <- keras_model_sequential() %>%
  layer_dense(units = input_dim, input_shape = input_dim) %>%
  layer_dense(units = horizon, activation = "relu", kernel_regularizer=regularizer_l2(l=.001)) %>%
  layer_dense(units = horizon, activation = "relu", kernel_regularizer=regularizer_l2(l=.001)) %>%
  layer_dense(units = horizon, activation = "relu", kernel_regularizer=regularizer_l2(l=.001)) %>%
  layer_dense(units = horizon, activation = "relu", kernel_regularizer=regularizer_l2(l=.001)) %>%
  layer_dense(units = horizon, activation = "relu", kernel_regularizer=regularizer_l2(l=.001)) %>%
  layer_dense(units = horizon, activation = "relu", kernel_regularizer=regularizer_l2(l=.001)) %>%
  layer_dense(units = 2)


# The MLP is called using the MSE and adam optimiser. It predicts the baseline intensity mu and the branching ratio eta.
eta_mu_MLP %>% compile(optimizer = "adam", loss = loss_mean_squared_error)
eta_mu_MLP %>% fit(
  input, cbind(eta, mu),
  epochs = epochs,
  batch_size = batch_size,
  validation_split = validation_split,
  callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 0.01, patience = 25)), view_metrics = FALSE)

list("model" = eta_mu_MLP, "training_procceses" = discretised_process, "training_params" = params)
}