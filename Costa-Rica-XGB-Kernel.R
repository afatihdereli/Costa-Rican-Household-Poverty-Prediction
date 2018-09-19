library("data.table")
library("xgboost")

train = fread("../input/train.csv")
test  = fread("../input/test.csv")

target<-train$Target

train$Target <- NULL

all<-rbind(train[,-c("Id")],test[,-c("Id")])

all$v18q1<- NULL
all$rez_esc<- NULL

all$SQBmeaned[is.na(all$SQBmeaned)]<-mean(all$SQBmeaned,na.rm=T)

all$meaneduc[is.na(all$meaneduc)]<-mean(all$meaneduc,na.rm=T)

all$v2a1[is.na(all$v2a1)]<-mean(all$v2a1,na.rm=T)

all$idhogar<-as.numeric(as.factor(all$idhogar))

all$dependency<-as.numeric(as.factor(all$dependency))

all$edjefe<-as.numeric(as.factor(all$edjefe))

all$edjefa<-as.numeric(as.factor(all$edjefa))


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
          nrounds = 2000)



bst <- xgboost(data = dtrain, max_depth = p$max_depth, eta = p$eta, nrounds = p$nrounds, nthread = p$nthread, 
               objective = p$objective, print_every_n = 100,num_class=p$num_class)

pred <- predict(bst, dtest)

results<-data.table(cbind(ID=test$Id,target=data.table(pred)$pred))

results$target<-as.numeric(results$target)+1

write.csv(results,"costa_rica_xgboost_nrounds_2000.csv",row.names=FALSE)


genel = aggregate(genel[, -1], genel[, 1], mean)