library("rstudioapi")
setwd(dirname(getActiveDocumentContext()$path))
source('HP_Discretise.R')
source('MLE.R')
library(hawkes)
library(keras)
library(tensorflow)
library(tfprobability)
library(ggplot2)
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
intermediate_dim <- input_dim*.75


# Model definition --------------------------------------------------------
sampling <- function(arg){
  latent_mean <- arg[, 1:(latent_dim)]
  latent_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
  epsilon <- k_random_normal(shape = c(k_shape(latent_mean)[[1]]), mean=0.,stddev=1.0)
  latent_mean + k_exp(latent_log_var/2)*epsilon
}


intermediate_dim_2 <- floor(.5*intermediate_dim)

encoder_input <- layer_input(shape = input_dim,  name = "Input_Layer")
encoder_intermediate_1 <- layer_dense(encoder_input, intermediate_dim, activation = "relu",  name = "Intermediate_encoder")
encoder_intermediate_2 <- layer_dense(encoder_intermediate_1, intermediate_dim_2, activation = "relu", name = "Intermediate_encoder_2")
latent_mean <- layer_dense(encoder_intermediate_2, latent_dim, name = "Mean_Layer")
latent_log_var <- layer_dense(encoder_intermediate_2, latent_dim, name = "Std_Layer")
latent <- layer_concatenate(list(latent_mean, latent_log_var)) %>% layer_lambda(sampling, name = "Sampling_Layer")


x_intermediate_1 <- layer_dense(units = intermediate_dim_2,activation = "relu", name = "x_Intermediate_decoder_1")
x_intermediate_2 <- layer_dense(units = intermediate_dim, activation = "relu", name = "x_Intermediate_decoder_2")
x_output <- layer_dense(units = input_dim, activation = "softplus", name = "x_Decoder_output")


vae_x_decoder_intermediate_1 <- x_intermediate_1(latent)
vae_x_decoder_intermediate_2 <- x_intermediate_2(vae_x_decoder_intermediate_1)
vae_x_decoder_output <- x_output(vae_x_decoder_intermediate_2)

decoder_input <- layer_input(shape = latent_dim,  name = "Decoder_Input_Layer")
x_decoder_intermediate_1 <- x_intermediate_1(decoder_input)
x_decoder_intermediate_2 <- x_intermediate_2(x_decoder_intermediate_1)
x_decoder_output <- x_output(x_decoder_intermediate_2)

vae <- keras_model(encoder_input, vae_x_decoder_output)
encoder <- keras_model(encoder_input, latent_mean)
decoder <- keras_model(decoder_input, x_decoder_output)




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
    lh <- tfd_poisson(rate=y_pred)
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


#------------------------------------Full Test----------------------------------------
number_of_tests <- 100

test_average_events <- rnorm(number_of_tests, expected_activity, 10)

test_avg_events_per_time <- test_average_events/horizon
test_eta <- runif(number_of_tests, min_eta, max_eta)
test_mu <- test_avg_events_per_time * (1-test_eta)

test_beta <- runif(number_of_tests, min_beta, max_beta)
test_alpha <- test_beta*test_eta
test_params <- matrix(c(test_alpha, test_beta, test_mu), nrow = number_of_tests)


raw_test_process <- list()
for (i in 1:number_of_tests){
raw_test_process[i] <- simulateHawkes(test_mu[i],test_alpha[i],test_beta[i],horizon)}
test_processes <- discretise(raw_test_process, discretise_step, horizon)
latent_encoding <- predict(encoder, test_processes)
decoded_process <- predict(decoder, latent_encoding)


test_results <- rep(0,number_of_tests)
for (i in 1:number_of_tests){
  print(i)
  stepsize <- 1/1000
  n <- horizon/stepsize
  true_intensity <- intensity(0:n*stepsize, c(test_alpha,test_beta,test_mu), raw_test_process[[i]])

  
  f <- function(x){
    true_intensity[x*1000 + 1]
  }
integrated_intensity <- rep(0,100)

for(k in 1:horizon){
integrated_intensity[k] <- integrate(f, (k-1), k, subdivisions = 1000, abs.tol = 0.01)$value}
test_results[i] <- sqrt(mean((decoded_process[i,] - integrated_intensity)^2))/(max(integrated_intensity)-min(integrated_intensity))
}



#------------------------------------Test_low_eta----------------------------------------
number_of_tests <- 1

test_average_events <- 500
test_avg_events_per_time <- 5
test_eta <- .2
test_mu <- test_avg_events_per_time * (1-test_eta)

test_beta <- 1
test_alpha <- test_beta*test_eta
test_params <- matrix(c(test_alpha, test_beta, test_mu), nrow = number_of_tests)


raw_test_process <- list()
for (i in 1:number_of_tests){
  raw_test_process[i] <- simulateHawkes(test_mu[i],test_alpha[i],test_beta[i],horizon)}
test_process <- discretise(raw_test_process, discretise_step, horizon)
latent_encoding <- predict(encoder, test_process)
decoded_process <- predict(decoder, latent_encoding)


stepsize <- 1/1000
n <- horizon/stepsize
true_intensity <- intensity(0:n*stepsize, c(test_alpha,test_beta,test_mu), raw_test_process[[1]])
  
  
f <- function(x){
  true_intensity[x*1000 + 1]
}
integrated_intensity <- rep(0,100)
  
for(k in 1:horizon){
  integrated_intensity[k] <- integrate(f, (k-1), k, subdivisions = 1000, abs.tol = 0.01)$value}

data <- data.frame("x"= 1:100,"intensity" = integrated_intensity, "decoded" = c(decoded_process))

test <- sqrt(mean((decoded_process - integrated_intensity)^2))/(max(integrated_intensity)-min(integrated_intensity))

test <- paste("Normalised RMSE = ",round(test,4))

ggplot(data = data) + geom_line(aes(x=x, y = intensity),size = .6) + 
  geom_line(aes(x=x, y = decoded),size = .6, colour = "red") + 
  annotate(geom="text", x = 60, y = 5.5, label = "Integrated Intensity") +
  annotate(geom="text", x = 25, y = 6, label = "Decoded Intensity", colour = "red") +
  annotate(geom = "text", x = Inf, y = Inf, label = test, hjust = 1.05, vjust = 2, size = 3.25) +
  theme_light() + ylab("") + xlab("Time") + scale_x_continuous(expand = c(0,0))

ggsave("Poisson_VAE_low_eta_low_beta.png", width = 8, height = 4, dpi = 100)

#------------------------------------Test_high_eta----------------------------------------
number_of_tests <- 1

test_average_events <- 500
test_avg_events_per_time <- 5
test_eta <- .7
test_mu <- test_avg_events_per_time * (1-test_eta)

test_beta <- 1
test_alpha <- test_beta*test_eta
test_params <- matrix(c(test_alpha, test_beta, test_mu), nrow = number_of_tests)


raw_test_process <- list()
for (i in 1:number_of_tests){
  raw_test_process[i] <- simulateHawkes(test_mu[i],test_alpha[i],test_beta[i],horizon)}
test_process <- discretise(raw_test_process, discretise_step, horizon)
latent_encoding <- predict(encoder, test_process)
decoded_process <- predict(decoder, latent_encoding)


stepsize <- 1/1000
n <- horizon/stepsize
true_intensity <- intensity(0:n*stepsize, c(test_alpha,test_beta,test_mu), raw_test_process[[1]])


f <- function(x){
  true_intensity[x*1000 + 1]
}
integrated_intensity <- rep(0,100)

for(k in 1:horizon){
  integrated_intensity[k] <- integrate(f, (k-1), k, subdivisions = 1000, abs.tol = 0.01)$value}

data <- data.frame("x"= 1:100,"intensity" = integrated_intensity, "decoded" = c(decoded_process))

test <- sqrt(mean((decoded_process - integrated_intensity)^2))/(max(integrated_intensity)-min(integrated_intensity))

test <- paste("Normalised RMSE = ",round(test,4))

ggplot(data = data) + geom_line(aes(x=x, y = intensity),size = .6) + 
  geom_line(aes(x=x, y = decoded),size = .6, colour = "red") + 
  annotate(geom="text", x = 53, y = 8, label = "Integrated Intensity") +
  annotate(geom="text", x = 82, y = 6, label = "Decoded Intensity", colour = "red") +
  annotate(geom = "text", x = Inf, y = Inf, label = test, hjust = 1.25, vjust = 2) +
  theme_light() + ylab("") + xlab("Time") + scale_x_continuous(expand = c(0,0))

ggsave("Poisson_VAE_high_eta_low_beta.png", width = 8, height = 4, dpi = 100)

#------------------------------------Test_low_eta----------------------------------------
number_of_tests <- 1

test_average_events <- 500
test_avg_events_per_time <- 5
test_eta <- .2
test_mu <- test_avg_events_per_time * (1-test_eta)

test_beta <- 3
test_alpha <- test_beta*test_eta
test_params <- matrix(c(test_alpha, test_beta, test_mu), nrow = number_of_tests)


raw_test_process <- list()
for (i in 1:number_of_tests){
  raw_test_process[i] <- simulateHawkes(test_mu[i],test_alpha[i],test_beta[i],horizon)}
test_process <- discretise(raw_test_process, discretise_step, horizon)
latent_encoding <- predict(encoder, test_process)
decoded_process <- predict(decoder, latent_encoding)


stepsize <- 1/1000
n <- horizon/stepsize
true_intensity <- intensity(0:n*stepsize, c(test_alpha,test_beta,test_mu), raw_test_process[[1]])


f <- function(x){
  true_intensity[x*1000 + 1]
}
integrated_intensity <- rep(0,100)

for(k in 1:horizon){
  integrated_intensity[k] <- integrate(f, (k-1), k, subdivisions = 1000, abs.tol = 0.01)$value}

data <- data.frame("x"= 1:100,"intensity" = integrated_intensity, "decoded" = c(decoded_process))

test <- sqrt(mean((decoded_process - integrated_intensity)^2))/(max(integrated_intensity)-min(integrated_intensity))

test <- paste("Normalised RMSE = ",round(test,4))

ggplot(data = data) + geom_line(aes(x=x, y = intensity),size = .6) + 
  geom_line(aes(x=x, y = decoded),size = .6, colour = "red") + 
  annotate(geom="text", x = 55, y = 6, label = "Integrated Intensity") +
  annotate(geom="text", x = 15, y = 5.75, label = "Decoded Intensity", colour = "red") +
  annotate(geom = "text", x = Inf, y = Inf, label = test, hjust = 1.25, vjust = 2) +
  theme_light() + ylab("") + xlab("Time") + scale_x_continuous(expand = c(0,0))

ggsave("Poisson_VAE_low_eta_high_beta.png", width = 8, height = 4, dpi = 100)

#------------------------------------Test_high_eta----------------------------------------
number_of_tests <- 1

test_average_events <- 500
test_avg_events_per_time <- 5
test_eta <- .7
test_mu <- test_avg_events_per_time * (1-test_eta)

test_beta <- 3
test_alpha <- test_beta*test_eta
test_params <- matrix(c(test_alpha, test_beta, test_mu), nrow = number_of_tests)


raw_test_process <- list()
for (i in 1:number_of_tests){
  raw_test_process[i] <- simulateHawkes(test_mu[i],test_alpha[i],test_beta[i],horizon)}
test_process <- discretise(raw_test_process, discretise_step, horizon)
latent_encoding <- predict(encoder, test_process)
decoded_process <- predict(decoder, latent_encoding)


stepsize <- 1/1000
n <- horizon/stepsize
true_intensity <- intensity(0:n*stepsize, c(test_alpha,test_beta,test_mu), raw_test_process[[1]])


f <- function(x){
  true_intensity[x*1000 + 1]
}
integrated_intensity <- rep(0,100)

for(k in 1:horizon){
  integrated_intensity[k] <- integrate(f, (k-1), k, subdivisions = 1000, abs.tol = 0.01)$value}

data <- data.frame("x"= 1:100,"intensity" = integrated_intensity, "decoded" = c(decoded_process))

test <- sqrt(mean((decoded_process - integrated_intensity)^2))/(max(integrated_intensity)-min(integrated_intensity))

test <- paste("Normalised RMSE = ",round(test,4))

ggplot(data = data) + geom_line(aes(x=x, y = intensity),size = .6) + 
  geom_line(aes(x=x, y = decoded),size = .6, colour = "red") + 
  annotate(geom="text", x = 31, y = 10, label = "Integrated Intensity") +
  annotate(geom="text", x = 82, y = 6.25, label = "Decoded Intensity", colour = "red", size = 3) +
  annotate(geom = "text", x = Inf, y = Inf, label = test, hjust = 1.1, vjust = 2) +
  theme_light() + ylab("") + xlab("Time") + scale_x_continuous(expand = c(0,0))

ggsave("Poisson_VAE_high_eta_high_beta.png", width = 8, height = 4, dpi = 100)

# #----------------------------------------Stan_test-------------------------------------------------
# weights <- get_weights(decoder)
# W1 <- weights[[1]]
# B1 <- weights[[2]]
# W2 <- weights[[3]]
# B2 <- weights[[4]]
# W3 <- weights[[5]]
# B3 <- weights[[6]]
# 
# library(rstan)
# options(mc.cores = parallel::detectCores())
# rstan_options(auto_write = TRUE)
# mcmc_fit <- stan("Stan_test.stan", data = list('p'= latent_dim, 
#                                                'p1'= intermediate_dim_2,
#                                                'p2'= intermediate_dim,
#                                                'n'= input_dim,
#                                                'W1'= W1,
#                                                'B1'= B1,
#                                                'W2'= W2,
#                                                'B2'= B2,
#                                                'W3'= W3,
#                                                'B3'= B3,
#                                                'y'= test_processes[1,]))
#   
# list_of_draws <- extract(mcmc_fit)
# list_of_draws$f
