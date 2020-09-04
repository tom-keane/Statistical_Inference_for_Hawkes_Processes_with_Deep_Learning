discretise <- function(jump_times, delta, horizon){
  k <- floor(horizon/delta)
  n <- length(jump_times)
  counts <- matrix(rep(0,k*n), nrow = n)
  for (j in 1:n){
    h <- jump_times[[j]]
    l <- length(h)
    for (i in 1:l){
      w <- ceiling(h[i]/delta)
      counts[j,w] <- counts[j,w] + 1
    }
  }
  counts
}

tempFunct <- function(jump_times, horizon){
  if (length(jump_times)==0){
    stepsize <- horizon
  }
  else{
    times <- c(jump_times, horizon)
    diff <- times[2:length(times)]-times[2:length(times)-1]
    stepsize <- signif(min(diff),1)}
  stepsize
}

find_stepsize <- function(jump_times, horizon){min(unlist(lapply(jump_times, function(x) tempFunct(x, horizon))))}

jump_times <- function(h, horizon){
  stepsize <- horizon / length(h)
  times <- NULL
  idx_1 <- which(h==1)
  idx_2 <- which(h>1)
  if (length(idx_2) > 0) {
    for (i in 1:length(idx_2)){
      times <- c(times,runif(h[idx_2[i]],(idx_2[i]-1)*stepsize, idx_2[i]*stepsize))
    }}
  times <- c(times,(idx_1*stepsize)-(.5*stepsize))
  jump_times <- sort(times, decreasing = F)
  jump_times
}