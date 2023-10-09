library(ade4)
library(FactoMineR)
temperature <-read.table("http://factominer.free.fr/book/temperature.csv",header=TRUE,sep=";",dec=".",row.names=1)

class(temperature)
names(temperature)
rownames(temperature)
dim(temperature)
plot(as.numeric(temperature[1,1:12]), ylim= range(temperature[,1:12]))
lines(as.numeric(temperature[2,1:12]))
lines(as.numeric(temperature[3,1:12]), col="red")
lines(as.numeric(temperature[4,1:12]), col="blue")

res <- PCA(temperature,ind.sup=24:35,quanti.sup=13:16,quali.sup=17)

?PCA
names(res)

res$var

res$call

plot.PCA(res,choix="ind")

plot.PCA(res,choix="ind",habillage=17)

dimdesc(res)

res$eig

plot.PCA(res, choix = c("ind"),invisible=c("ind.sup", "quali", "quanti.sup"))

plot.PCA(res, choix = c("ind"), invisible = c("ind"))
res$ind.sup

res$quali.sup
plot.PCA(res, choix = "ind", invisible = c("ind", "ind.sup"))








data(JO)
?JO

apply(JO, 1, sum)

resJO <- CA(JO)
summary(resJO)

## profils lignes
rowprof <- JO / apply(JO, 1, sum)
apply(rowprof,1,sum)
## profils colonnes
colprof <- t(t(JO) / apply(JO, 2, sum)) #t() : transposee
apply(colprof,2,sum)

round(resJO$eig,1)

