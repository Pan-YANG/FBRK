library(sp)
library(gstat)
setwd("***") # enter the working folder
data.err=read.csv('bayes_krig.csv')
coordinates(data.err)=~x+y
err.vgm=variogram(obs~1,data.err)
coeff_out=tryCatch({
err.fit=fit.variogram(err.vgm,model=vgm("Exp"))
tao=err.fit[[2]][1]
sigma=err.fit[[2]][2]
theta=err.fit[[3]][2]
c(tao,sigma,theta)
},warning=function(w){
err.fit=fit.variogram(err.vgm,model=vgm(1,"Exp",1))
tao=0
sigma=err.fit[[2]][1]
theta=err.fit[[3]][1]
c(tao,sigma,theta)
}
)

write.csv(coeff_out,'variogram_coeff.csv')
