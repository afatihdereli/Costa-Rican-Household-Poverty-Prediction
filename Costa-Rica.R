library("data.table")
library("xgboost")

setwd("C:/Users/fatih.dereli/Desktop/fatih/Personal/Costa Rican Household Poverty Prediction")

train = fread("train.csv")
test  = fread("test.csv")
sample = fread("sample_submission.csv")


#Data Prep
target<-train$Target

train$Target <- NULL

all<-rbind(train[,-c("Id")],test[,-c("Id")])

all$v18q1<- NULL
all$rez_esc<- NULL

all$SQBmeaned[is.na(all$SQBmeaned)]<-mean(all$SQBmeaned,na.rm=T)

all$meaneduc[is.na(all$meaneduc)]<-mean(all$meaneduc,na.rm=T)

all$v2a1[is.na(all$v2a1)]<-mean(all$v2a1,na.rm=T)


#PCA
str(all)

str(all[,100:139])

all$idhogar<-as.numeric(as.factor(all$idhogar))

all$dependency<-as.numeric(as.factor(all$dependency))

all$edjefe<-as.numeric(as.factor(all$edjefe))

all$edjefa<-as.numeric(as.factor(all$edjefa))


pca<-princomp(all)

as.numeric(as.factor(all$idhogar))
target


#XGB
tri <- 1:nrow(train)
dtest <- xgb.DMatrix(data = as.matrix(all[-tri, ]))
tr_te <- all[tri, ]

dtrain <- xgb.DMatrix(data = as.matrix(all[tri, ]), label = target-1)

p <- list(objective = "multi:softmax",
          booster = "gbtree",
          eval_metric = "mlogloss",
           num_class = length(unique(target)),
          nthread = 4,
          eta = 0.05,
          max_depth = 8,
          min_child_weight = 6,
          gamma = 0,
          subsample = 0.7,
          colsample_bytree = 0.7,
          colsample_bylevel = 0.7,
          alpha = 0,
          lambda = 0,
          nrounds = 10000)

dval <- xgb.DMatrix(data = as.matrix(all[-tri, ]), label = target[-tri] - 1)

watchlist<-list(train=dtrain, test=dtest)


#m_xgb <- xgb.train(p, dtrain, p$nrounds, #list(val = dval), 
                   print_every_n = 200, early_stopping_rounds = 400, watchlist=watchlist)

bst <- xgboost(data = dtrain, max_depth = p$max_depth, eta = p$eta, nrounds = p$nrounds, nthread = p$nthread, 
               objective = p$objective, print_every_n = 100,num_class=p$num_class)


predtest<-predict(bst, dtrain)

predtest[predtest<0]<-0

rmsle(new_train$target,predtest)

##1.34 on train, 2.42 on test
pred <- predict(bst, dtest)

results<-data.table(cbind(ID=test$Id,target=data.table(pred)$pred))

results$target<-as.numeric(results$target)+1

results[results$target<0,]$target<-0

write.csv(results,"costa_rica_xgboost_nrounds_10000.csv",row.names=FALSE)







tri <- createDataPartition(y, p = 0.9, list = F) %>% c()
dtrain <- xgb.DMatrix(data = tr_te[tri, ], label = y[tri] - 1)
dval <- xgb.DMatrix(data = tr_te[-tri, ], label = y[-tri] - 1)
cols <- colnames(tr_te)
rm(tr, te, tr_te, tri)
gc()