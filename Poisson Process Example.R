library(poisson)
library(ggplot2)
set.seed(1)
n <- 7
process <- hpp.sim(1.5, n, num.sims = 1, t0 = 0, prepend.t0 = T)

data <- data.frame("x" = c(process, 5), "y" = c(0:n,n))
data$xend <- c(data$x[2:nrow(data)], NA)
data$yend <- data$y

x_axis_labels <- c()

for (i in 1:n){
  x_axis_labels[i] <- as.expression(bquote("t"[.(i)]))
}

ggplot() +
  geom_segment(data = data[-c(1,2+n),], aes(x=x,y=y,xend=x,yend=-Inf), linetype=2, color="grey") +
  geom_point(data = data[-c(1,2+n),], aes(x,y)) +  # Solid points to left
  geom_point(data = data[-(1:2+n),], aes(x=xend, y=y), shape=1) +  # Open points to right
  geom_segment(data=data, aes(x=x,y=y,xend=xend,yend=yend), size=.75) + 
  theme_light() + labs(x = "t", y = expression(N[t])) + scale_y_continuous(breaks=seq(0,7,1)) +
  scale_x_continuous(breaks=data$x[-c(1,n+2)], labels = x_axis_labels, expand = c(0, 0))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))
        