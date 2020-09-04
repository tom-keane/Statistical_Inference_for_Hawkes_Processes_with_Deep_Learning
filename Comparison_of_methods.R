library("rstudioapi")
setwd(dirname(getActiveDocumentContext()$path))
source('Supervised_method.R')
source('MLE.R')
library(ggplot2)
#----------------------Method Tests----------------------
set.seed(0)
min_eta <- .5
max_eta <- .8
min_beta <- 1
max_beta <- 3
expected_activity <- 1000
horizon <- 1000
discretise_step <- 1

number_of_tests <- 100
number_of_test_processes <- 10

test_average_events <- rnorm(number_of_tests, expected_activity, 100)

test_avg_events_per_time <- test_average_events/horizon
test_eta <- runif(number_of_tests, min_eta, max_eta)
test_mu <- test_avg_events_per_time * (1-test_eta)

test_beta <- runif(number_of_tests, min_beta, max_beta)
test_alpha <- test_beta*test_eta
test_params <- matrix(c(test_alpha, test_beta, test_mu), nrow = number_of_tests)

write.csv(data.frame("alpha" = test_alpha, "beta" = test_beta, "mu" = test_mu), 
          "test_parameters.csv",
          row.names = FALSE)

raw_test_processes <- list()
for (k in 1:number_of_tests){
  raw_test_processes[[k]] <- unlist(simulateHawkes(test_mu[k],test_alpha[k],test_beta[k], horizon = horizon))
}

test_processes_data <- discretise(raw_test_processes, discretise_step, horizon = horizon)

write.table(data.frame(test_processes_data), sep = ",", 
          "test_processes.csv",
          row.names = FALSE, col.names = FALSE)


#---------------------------------Supervised Tests------------------------------------
training <- eta_mu_model(100000, horizon/number_of_test_processes, 
                         expected_activity/number_of_test_processes, discretise_step,
                         min_eta = min_eta, max_eta = max_eta, 
                         min_beta = min_beta, max_beta = max_beta)
raw_results <- list()
errors <- list()
for (k in 1:number_of_tests){
  test_processes <- matrix(test_processes_data[k,], nrow = number_of_test_processes, byrow = T)
  
  raw_results[[k]] <- predict_hawkes_parameters(training$model, test_processes, 
                                                training$training_procceses, training$training_params)
  
  errors[[k]] <-  cbind(raw_results[[k]]$alpha_est, raw_results[[k]]$beta_est, raw_results[[k]]$mu_est) - test_params[k,]
}


supervised_results <- list("raw_results" = matrix(unlist(raw_results),ncol = 4), "errors" = matrix(unlist(errors),ncol = 3))






raw_results <- list()
errors <- list()
mult_factor <- 200/number_of_test_processes
number_of_test_processes <- 200
for (k in 1:number_of_tests){
  raw_test_process <- simulateHawkes(test_mu[k],test_alpha[k],test_beta[k], horizon*mult_factor)
  test_processes <- matrix(discretise(raw_test_process, discretise_step, horizon*mult_factor), 
                           nrow = number_of_test_processes, byrow = T)
  
  raw_results[[k]] <- predict_hawkes_parameters(training$model, test_processes, 
                                                training$training_procceses, training$training_params)
  
  errors[[k]] <-  cbind(raw_results[[k]]$alpha_est, raw_results[[k]]$beta_est, raw_results[[k]]$mu_est) - test_params[k,]
}


supervised_results_high_data <- list("raw_results" = matrix(unlist(raw_results),ncol = 4), 
                                     "errors" = matrix(unlist(errors),ncol = 3))

#----------------------------------MLE Tests---------------------------------------------
raw_results <- list()
errors <- list()
for (k in 1:number_of_tests){
  print(k)
  raw_results[[k]] <- HP_MLE(c(0.5,2,.5), loglik, raw_test_processes[[k]])
  errors[[k]] <-  raw_results[[k]] - test_params[k,]
}

MLE_results <- list("raw_results" = matrix(unlist(raw_results),ncol = 3), "errors" = matrix(unlist(errors),ncol = 3))

#-----------------------------------MCEM Tests------------------------------------------

raw_results <- read.csv("test_results_MCEM.csv",header = F)
mu_est <- raw_results[,1]
raw_results[,1:2] <- raw_results[,2:3]
raw_results[,3] <- mu_est

MCEM_results <- list("raw_results" = raw_results,
                     "errors" = raw_results-test_params)


method_names <- c("Supervised", "Supervised (High Data)", "MC-EM", "MLE")

alpha_error_data <- data.frame("Error" = c(supervised_results$errors[,1],
                                           supervised_results_high_data$errors[,1],
                                           MCEM_results$errors[,1],
                                           MLE_results$errors[,1]),
                               "Methods" = as.factor(rep(method_names, 
                                                             each = number_of_tests)))

beta_error_data <- data.frame("Error" = c(supervised_results$errors[,2],
                                           supervised_results_high_data$errors[,2],
                                           MCEM_results$errors[,2],
                                           MLE_results$errors[,2]),
                               "Methods" = as.factor(rep(method_names, 
                                                         each = number_of_tests)))


mu_error_data <- data.frame("Error" = c(supervised_results$errors[,3],
                                           supervised_results_high_data$errors[,3],
                                           MCEM_results$errors[,3],
                                           MLE_results$errors[,3]),
                               "Methods" = as.factor(rep(method_names, 
                                                         each = number_of_tests)))



for (i in 1:length(method_names)){
  alpha_error_data[which(alpha_error_data$Methods == method_names[i]),1:3] <-
    alpha_error_data[which(alpha_error_data$Methods == method_names[i]), ] %>%
    mutate(outlier = Error > quantile(Error, .75) + IQR(Error)*1.5 | Error < quantile(Error, .25) - IQR(Error)*1.5)
  
  beta_error_data[which(beta_error_data$Methods == method_names[i]),1:3] <-
    beta_error_data[which(beta_error_data$Methods == method_names[i]), ] %>%
    mutate(outlier = Error > quantile(Error, .75) + IQR(Error)*1.5 | Error < quantile(Error, .25) - IQR(Error)*1.5)
  
  mu_error_data[which(mu_error_data$Methods == method_names[i]),1:3] <-
    mu_error_data[which(mu_error_data$Methods == method_names[i]), ] %>%
    mutate(outlier = Error > quantile(Error, .75) + IQR(Error)*1.5 | Error < quantile(Error, .25) - IQR(Error)*1.5)
}



alpha_error_plot <- ggplot(data = alpha_error_data, aes(x = Methods, y = Error)) + geom_boxplot(outlier.shape = NA) + 
  theme_light() +theme(panel.grid.major.x = element_blank(), legend.position="none")+
  geom_point(data = function(x) dplyr::filter(x, outlier), position = position_dodge2(width = .1), aes(alpha = .6))
alpha_error_plot
ggsave("Comparison_of_Methods_Alpha.png", width = 8, height = 4, dpi = 100)

beta_error_plot <- ggplot(data = beta_error_data, aes(x = Methods, y = Error)) + geom_boxplot(outlier.shape = NA) + 
  theme_light() +theme(panel.grid.major.x = element_blank(), legend.position="none")+
  geom_point(data = function(x) dplyr::filter(x, outlier), position = position_dodge2(width = .1), aes(alpha = .6))
beta_error_plot
ggsave("Comparison_of_Methods_Beta.png", width = 8, height = 4, dpi = 100)

mu_error_plot <- ggplot(data = mu_error_data, aes(x = Methods, y = Error)) + geom_boxplot(outlier.shape = NA) + 
  theme_light() +theme(panel.grid.major.x = element_blank(), legend.position="none")+
  geom_point(data = function(x) dplyr::filter(x, outlier), position = position_dodge2(width = .1), aes(alpha = .6))
mu_error_plot
ggsave("Comparison_of_Methods_Mu.png", width = 8, height = 4, dpi = 100)