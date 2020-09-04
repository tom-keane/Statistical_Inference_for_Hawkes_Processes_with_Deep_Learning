library(hawkes)
library(ggplot2)
set.seed(1)
source('C:/Users/Thomas/Dropbox/University/Imperial/Thesis/Code/HP_Discretise.R')
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

xlim = 7.5
params <- c(2, 2.5, 1)

process <- simulateHawkes(params[3], params[1], params[2], xlim)[[1]]
n <- length(process)


cts_x <- 0:(xlim*100000)/100000
cts_y <- intensity(cts_x, params, process)

ylim = ceiling(max(cts_y))

jumps_idx <- which(c(0,discretise(list(process), .00001,xlim), 0)==1)
before_jumps_idx <- jumps_idx-1


intensity_data <- data.frame("x" = cts_x, "y" = cts_y)
jump_data <- data.frame("start_x" = cts_x[jumps_idx], "start_y" = cts_y[jumps_idx], 
                    "end_x" = cts_x[before_jumps_idx], "end_y" = cts_y[before_jumps_idx])
 
x_axis_labels <- c()

for (i in 1:(n)){
  x_axis_labels[i] <- as.expression(bquote("t"[.(i)]))
}

p <- ggplot()
jumps_idx_plot <- c(0,jumps_idx)
before_jumps_idx_plot <- c(before_jumps_idx,length(cts_x))
for (i in 1:length(jumps_idx_plot)) {
  p <- p + geom_line(data = intensity_data[jumps_idx_plot[i]:before_jumps_idx_plot[i],], aes(x = x, y = y), size=.75 )
  
}
p +
  geom_segment(data = jump_data, aes(x=start_x,y=start_y,xend=start_x,yend=-Inf), linetype=2, color="grey") +
  geom_point(data = jump_data, aes(x=start_x,y=start_y)) +  # Solid points to left
  geom_point(data = jump_data, aes(x=end_x, y=end_y), shape=1) +
  theme_light() + labs(x = "t", y = "Conditional Intensity") + scale_y_continuous(breaks=seq(0,ylim,1), limits = c(0,ylim), expand = c(0, 0)) +
  scale_x_continuous(breaks=jump_data$start_x, labels = x_axis_labels, expand = c(0, 0))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))

