library(hawkes)
estimated_intensity <- function(params, arrivals){
  alpha_i <- params[1]
  beta_i <- params[2]
  mu_i <- params[3]
  Ai <- c(0, sapply(2:n, function(z) {
    sum(exp( -beta_i * (arrivals[z]- arrivals[1:(z - 1)])))
  }))
  return(mu_i + alpha_i *Ai)
}

loglik <- function(params, arrivals){
  alpha_i <- params[1]
  beta_i <- params[2]
  mu_i <- params[3]
  n <- length(arrivals)
  term_1 <- -mu_i*arrivals[n]
  term_2 <- sum(alpha_i/beta_i*(exp( -beta_i * (arrivals[n] - arrivals)) - 1))
  Ai <- c(0, sapply(2:n, function(z) {
    sum(exp( -beta_i * (arrivals[z]- arrivals[1:(z - 1)])))
  }))
  term_3 <- sum(log( mu_i + alpha_i * Ai))
  return(-term_1- term_2 -term_3)
}

HP_MLE <- function(start_params, log_likelihood, H_process){
  solution <- optim(start_params, log_likelihood, arrivals = H_process)
  #print(paste( c("alpha", "beta", "mu"), round(solution$par,4), sep=" = "))
  round(solution$par,4)
}


intensity <- function(t, params, process){
  alpha <- params[1]
  beta <- params[2]
  mu <- params[3]
  intensity <- rep(0,length(t))
  for (i in 1:length(t)){
    intensity[i] <- sum(exp(beta*(process[which(process<t[i])]-t[i])))
  }
  intensity * alpha + mu
}

#-------------------------Example-----------------------------
# number_of_processes <- 1
# lambda0 <- 3
# alpha <- 3
# beta <- 4
# horizon <- 150
# h <- list()
# for (i in 1:number_of_processes) {
#   h[i] <- simulateHawkes(lambda0,alpha,beta,horizon)
# }
# 
# stepsize <- find_stepsize(h,horizon)
# stepsize <- 5
# discretised_process <- discretise(h, stepsize, horizon)
# jumps <- jump_times(discretised_process,horizon)
# 
# 
# case1_solution1 <- optim(c(1,2,0.1), loglik, arrivals = h[[1]])
# paste( c("alpha", "beta", "mu"), round(case1_solution1$par,2), sep=" = ")
# 
# case1_solution1 <- optim(c(1,2,0.1), loglik, arrivals = jumps)
# paste( c("alpha", "beta", "mu"), round(case1_solution1$par,2), sep=" = ")

