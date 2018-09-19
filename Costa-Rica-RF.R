library("data.table")
library("randomForest")

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



rf <- randomForest(target ~ ., data=all[1:nrow(train),-c("Id")], ntree=20)

pred<-predict(rf,all[(nrow(train)+1):nrow(all),-c("Id")])

results<-data.table(cbind(ID=test$Id,target=data.table(pred)$pred))

results$target<-round(as.numeric(as.character(results$target)))

write.csv(results,"costa_rica_rf_ntree_20.csv",row.names=FALSE)
