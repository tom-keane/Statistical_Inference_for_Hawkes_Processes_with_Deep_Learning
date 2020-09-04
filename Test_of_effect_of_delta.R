library("rstudioapi")
setwd(dirname(getActiveDocumentContext()$path))
source('Supervised_method.R')
library(ggplot2)
library(dplyr)

#----------------------Delta tests----------------------
delta_ranges <- list(.25,.5,1,2,5)

set.seed(0)
results <- list()
for (i in 1:length(delta_ranges)){
  print(i)
  min_eta <- .2
  max_eta <- .6
  min_beta <- 1
  max_beta <- 3
  expected_activity <- 500
  horizon <- 100
  discretise_step <- delta_ranges[[i]]
  number_of_tests <- 100
  number_of_test_processes <- 200
  
  test_average_events <- rnorm(number_of_tests,expected_activity,10)
  
  test_avg_events_per_time <- test_average_events/horizon
  test_eta <- runif(number_of_tests, min_eta, max_eta)
  test_mu <- test_avg_events_per_time * (1-test_eta)
  
  test_beta <- runif(number_of_tests, min_beta, max_beta)
  test_alpha <- test_beta*test_eta
  test_params <- matrix(c(test_alpha, test_beta, test_mu), nrow = number_of_tests)
  
  training <- eta_mu_model(100000, horizon, expected_activity, discretise_step,
                           min_eta = min_eta, max_eta = max_eta, 
                           min_beta = min_beta, max_beta = max_beta)
  
  
  raw_results <- list()
  errors <- list()

  
  for (k in 1:number_of_tests){
    raw_test_processes <- list()
    for (j in 1:number_of_test_processes) {
      raw_test_processes[j] <- simulateHawkes(test_mu[k],test_alpha[k],test_beta[k],horizon)
    }
    test_processes <- discretise(raw_test_processes, discretise_step, horizon)
    raw_results[[k]] <- predict_hawkes_parameters(training$model, test_processes, 
                                                  training$training_procceses, training$training_params)
    errors[[k]] <-  cbind(raw_results[[k]]$alpha_est, raw_results[[k]]$beta_est, raw_results[[k]]$mu_est) - test_params[k,]
  }
  results[[i]] <- list("raw_results" = matrix(unlist(raw_results),ncol = 4), "errors" = matrix(unlist(errors),ncol = 3))
}


alpha_error_data <- data.frame("Error" = c(results[[1]]$errors[,1],
                                           results[[2]]$errors[,1],
                                           results[[3]]$errors[,1],
                                           results[[4]]$errors[,1],
                                           results[[5]]$errors[,1]),
                               "Delta" = as.factor(rep(unlist(delta_ranges),each = number_of_tests)))

beta_error_data <-  data.frame("Error" = c(results[[1]]$errors[,2],
                                           results[[2]]$errors[,2],
                                           results[[3]]$errors[,2],
                                           results[[4]]$errors[,2],
                                           results[[5]]$errors[,2]),
                               "Delta" = as.factor(rep(unlist(delta_ranges),each = number_of_tests)))

mu_error_data <-    data.frame("Error" = c(results[[1]]$errors[,3],
                                           results[[2]]$errors[,3],
                                           results[[3]]$errors[,3],
                                           results[[4]]$errors[,3],
                                           results[[5]]$errors[,3]),
                               "Delta" = as.factor(rep(unlist(delta_ranges),each = number_of_tests)))


for (i in 1:length(delta_ranges)){
  alpha_error_data[which(alpha_error_data$Delta == delta_ranges[i]),1:3] <-
    alpha_error_data[which(alpha_error_data$Delta == delta_ranges[i]), ] %>%
    mutate(outlier = Error > quantile(Error, .75) + IQR(Error)*1.5 | Error < quantile(Error, .25) - IQR(Error)*1.5)
  
  beta_error_data[which(beta_error_data$Delta == delta_ranges[i]),1:3] <-
    beta_error_data[which(beta_error_data$Delta == delta_ranges[i]), ] %>%
    mutate(outlier = Error > quantile(Error, .75) + IQR(Error)*1.5 | Error < quantile(Error, .25) - IQR(Error)*1.5)
  
  mu_error_data[which(mu_error_data$Delta == delta_ranges[i]),1:3] <-
    mu_error_data[which(mu_error_data$Delta == delta_ranges[i]), ] %>%
    mutate(outlier = Error > quantile(Error, .75) + IQR(Error)*1.5 | Error < quantile(Error, .25) - IQR(Error)*1.5)
}



alpha_error_plot <- ggplot(data = alpha_error_data, aes(x = Delta, y = Error)) + geom_boxplot(outlier.shape = NA) + 
  theme_light() +theme(panel.grid.major.x = element_blank(), legend.position="none")+
  geom_point(data = function(x) dplyr::filter(x, outlier), position = position_dodge2(width = .1), aes(alpha = .6))
alpha_error_plot
ggsave("Effect_of_Delta_on_Alpha.png", width = 8, height = 4, dpi = 100)

beta_error_plot <- ggplot(data = beta_error_data, aes(x = Delta, y = Error)) + geom_boxplot(outlier.shape = NA) + 
  theme_light() +theme(panel.grid.major.x = element_blank(), legend.position="none")+
  geom_point(data = function(x) dplyr::filter(x, outlier), position = position_dodge2(width = .1), aes(alpha = .6))
beta_error_plot
ggsave("Effect_of_Delta_on_Beta.png", width = 8, height = 4, dpi = 100)

mu_error_plot <- ggplot(data = mu_error_data, aes(x = Delta, y = Error)) + geom_boxplot(outlier.shape = NA) + 
  theme_light() +theme(panel.grid.major.x = element_blank(), legend.position="none")+
  geom_point(data = function(x) dplyr::filter(x, outlier), position = position_dodge2(width = .1), aes(alpha = .6))
mu_error_plot
ggsave("Effect_of_Delta_on_Mu.png", width = 8, height = 4, dpi = 100)
