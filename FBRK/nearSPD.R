library(Matrix)
setwd("***") # enter the working folder

A=read.csv('nearSPD.csv')
A=A$x
A=matrix(data=A,sqrt(length(A)),sqrt(length(A)))
B=nearPD(A)
B=B$mat
B=as.vector(B)
write.csv(B,'nearSPD.csv')
