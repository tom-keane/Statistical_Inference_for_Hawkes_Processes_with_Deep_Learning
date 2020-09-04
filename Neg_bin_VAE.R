library("rstudioapi")
setwd(dirname(getActiveDocumentContext()$path))
source('HP_Discretise.R')
source('MLE.R')
library(hawkes)
library(keras)
library(tensorflow)
library(tfprobability)
K <- keras::backend()
if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()
set.seed(0)


number_of_training_processes <- 100000
expected_activity <- 500
min_eta <- .05
max_eta <- .8
min_beta <- 1
max_beta <- 3
horizon <- 100
discretise_step <- 1


raw_training_processes <- list()

average_events <- rnorm(number_of_training_processes, expected_activity, 10)
avg_events_per_time <- average_events/horizon

eta <- runif(number_of_training_processes, min_eta, max_eta)
mu <- avg_events_per_time * (1-eta)
beta <- runif(number_of_training_processes, min_beta, max_beta)
alpha <- beta*eta

params <- matrix(c(alpha, beta, mu), nrow = number_of_training_processes)

for (i in 1:number_of_training_processes) {
  raw_training_processes[i] <- simulateHawkes(mu[i],alpha[i],beta[i],horizon)
}

discretised_process <- discretise(raw_training_processes, discretise_step, horizon)


input <- discretised_process
latent_dim <- 15L 
epochs <- 10000
validation_split <- 0.05
batch_size <- dim(input)[1]*.1
input_dim <- dim(input)[2]
intermediate_dim <- input_dim*.5


# Model definition --------------------------------------------------------
sampling <- function(arg){
  latent_mean <- arg[, 1:(latent_dim)]
  latent_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
  epsilon <- k_random_normal(shape = c(k_shape(latent_mean)[[1]]), mean=0.,stddev=1.0)
  latent_mean + k_exp(latent_log_var/2)*epsilon
}


intermediate_dim_2 <- floor(.5*intermediate_dim)

encoder_input <- layer_input(shape = input_dim,  name = "Input_Layer")
encoder_intermediate_1 <- layer_dense(encoder_input, intermediate_dim, activation = "tanh",  name = "Intermediate_encoder")
encoder_intermediate_2 <- layer_dense(encoder_intermediate_1, intermediate_dim_2, activation = "tanh", name = "Intermediate_encoder_2")
latent_mean <- layer_dense(encoder_intermediate_2, latent_dim, name = "Mean_Layer")
latent_log_var <- layer_dense(encoder_intermediate_2, latent_dim, name = "Std_Layer")
latent <- layer_concatenate(list(latent_mean, latent_log_var)) %>% layer_lambda(sampling, name = "Sampling_Layer")


R_intermediate_1 <- layer_dense(units = intermediate_dim_2, activation = "tanh", name = "R_Intermediate_decoder")
R_intermediate_2 <- layer_dense(units = intermediate_dim, activation = "tanh", name = "R_Intermediate_decoder_2")
R_decoder_output <- layer_dense(units = input_dim, name = "R_Decoder_output")


P_intermediate_1 <- layer_dense(units = intermediate_dim_2, activation = "tanh", name = "P_Intermediate_decoder")
P_intermediate_2 <- layer_dense(units = intermediate_dim, activation = "tanh", name = "P_Intermediate_decoder_2")
P_decoder_output <- layer_dense(units = input_dim, name = "P_Decoder_output")


R_decode_intermediate_1 <- R_intermediate_1(latent)
R_decode_intermediate_2 <- R_intermediate_2(R_decode_intermediate_1)
R_decode_output <- R_decoder_output(R_decode_intermediate_2)

P_decode_intermediate_1 <- P_intermediate_1(latent)
P_decode_intermediate_2 <- P_intermediate_2(P_decode_intermediate_1)
P_decode_output <- P_decoder_output(P_decode_intermediate_2)

vae_decode_output <- layer_concatenate(list(R_decode_output, P_decode_output))


decoder_input <- layer_input(shape = latent_dim,  name = "Decoder_Input_Layer")
p_decoder_intermediate_1 <- P_intermediate_1(decoder_input)
p_decoder_intermediate_2 <- P_intermediate_2(p_decoder_intermediate_1)
p_decoder_output <- P_decoder_output(p_decoder_intermediate_2)

r_decoder_intermediate_1 <- R_intermediate_1(decoder_input)
r_decoder_intermediate_2 <- R_intermediate_2(r_decoder_intermediate_1)
r_decoder_output <- R_decoder_output(r_decoder_intermediate_2)

decoder_output <- layer_concatenate(list(r_decoder_output, p_decoder_output))

vae <- keras_model(encoder_input, vae_decode_output)

encoder <- keras_model(encoder_input, latent_mean)
decoder <- keras_model(decoder_input, decoder_output)




#Loss specification-----------------------------------------------------
weight <- k_variable(0)
kl.start <- 2000
kl.steep <- 1000
anneal_target <- 1
min_cycles <- 8
#Annealing-------------------------------------------------------------------
KL.LinAnn <- R6::R6Class("KL.LinAnn",
                         inherit = KerasCallback,
                         
                         public = list(
                           losses = NULL,
                           params = NULL,
                           model = NULL,
                           weight = NULL,
                           
                           set_context = function(params = NULL, model = NULL) {
                             self$params <- params
                             self$model <- model
                             self$weight <- weight
                           },                        
                           on_epoch_end = function(epoch, logs = NULL) {
                             current_weight <- k_get_value(self$weight)
                             if(epoch>=(kl.start)){
                               new_weight <- min((epoch-kl.start)/kl.steep, anneal_target)
                               if (current_weight >new_weight) {
                                 new_weight <- current_weight + new_weight}
                               k_set_value(self$weight, new_weight)
                               print(paste("     ANNEALING KLD:", k_get_value(self$weight), sep = " "))
                             }}))


KL.CycAnn <- R6::R6Class("KL.CycAnn",
                         inherit = KerasCallback,
                         
                         public = list(
                           losses = NULL,
                           params = NULL,
                           model = NULL,
                           weight = NULL,
                           cycle = NULL,
                           
                           set_context = function(params = NULL, model = NULL) {
                             self$params <- params
                             self$model <- model
                             self$weight <- weight
                             self$cycle <- k_variable(1)
                           },                        
                           on_epoch_end = function(epoch, logs = NULL) {
                             current_cycle <- k_get_value(self$cycle)
                             target_mult <- min(current_cycle/min_cycles, 1)
                             
                             if(epoch>=(kl.start)){
                               new_weight <- target_mult * min(2 * (epoch - kl.start - (current_cycle - 1) * (kl.steep))/kl.steep,
                                                               anneal_target )
                               
                               if ((epoch-kl.start+1)%%kl.steep == 0) {
                                 k_set_value(self$cycle, current_cycle + 1)}
                               
                               k_set_value(self$weight, new_weight)
                               print(paste("     ANNEALING KLD:", k_get_value(self$weight), sep = " "))
                             }}))


loss<- function(weight){
  vae_loss <- function(y_true, y_pred){
    r_pred <- y_pred[,1:input_dim]
    p_pred <- y_pred[,(1:input_dim)+input_dim]
    
    lh <- tfd_negative_binomial(total_count = k_exp(r_pred), logits = p_pred)
    xent_loss <-  -k_sum(tfd_log_prob(lh, y_true), axis=-1)
    kl_loss <- -0.5*k_sum(1 + latent_log_var - k_square(latent_mean) - k_exp(latent_log_var), axis = -1L)
    xent_loss + weight*kl_loss
  }
  return(vae_loss)
}

annealing <- KL.CycAnn$new()

vae %>% compile(optimizer = optimizer_adam(clipvalue = 1000), loss = loss(weight))
vae %>% fit(
  input, input, 
  epochs = epochs,
  batch_size = batch_size,
  validation_split = validation_split,
  callbacks = list(annealing),
  view_metrics = FALSE)

#--------------------------test-----------------------------------
sqrt(test_results)

latent_encoding <- predict(encoder, test_processes)


decoded_parameters <- predict(decoder, latent_encoding)
decoded_process <- exp(decoded_parameters[,1:100])*exp(decoded_parameters[,101:200])

test_results <- rep(0,number_of_tests)


for (i in 1:number_of_tests){
  print(i)
  stepsize <- 1/1000
  n <- horizon/stepsize
  true_intensity <- intensity(0:n*stepsize, c(test_alpha,test_beta,test_mu), raw_test_process[[i]])
  # plot(0:n, true_intensity, type = "l")
  
  f <- function(x){
    true_intensity[x*1000 + 1]
  }
  integrated_intensity <- rep(0,100)
  
  for(k in 1:horizon){
    integrated_intensity[k] <- integrate(f, (k-1), k, subdivisions = 1000, abs.tol = 0.01)$value}
  test_results[i] <- mean(decoded_process[i,] - integrated_intensity)^2/(max(integrated_intensity)-min(integrated_intensity))
}

