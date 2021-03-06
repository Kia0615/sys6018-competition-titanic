library(readr)  
library(dplyr)

#Read in the datasets
train <- read_csv("train.csv")
test <- read_csv("test.csv")
test
#Factorize the categorical variables
train$Pclass <- factor(train$Pclass)
train$Sex <- factor(train$Sex)
train$Embarked <- factor(train$Embarked)
train$Parch <- factor(train$Parch)
test$Pclass <- factor(test$Pclass)
test$Sex <- factor(test$Sex)
test$Embarked <- factor(test$Embarked)
test$Parch <- factor(test$Parch)

# Subset to cross validate
sub <- sample(1:891,size=446)
subtrain <- train[sub,]     # Select subset for cross-validation
valid <- train[-sub,]

#Logistic regression model
lg1 <- glm(Survived~.-PassengerId-Name-Ticket-Cabin, data=subtrain, family = "binomial")
summary(lg1)

# Test the full model on the training set
probs<-as.vector(predict(lg1, type="response"))
preds <- rep(0,446)  # Initialize prediction vector
preds[probs>0.5] <- 1 # p>0.5 -> 1
preds
table(preds,subtrain$Survived)
(159+68)/446  # Proportion of predictions that are correct:50.9%

anova(lg1,test="Chisq")

lg2 <- glm(Survived~Sex+Age+Pclass+Parch, data=subtrain, family = "binomial")
summary(lg2)

# Test the full model on the training set
probs<-as.vector(predict(lg2, type="response"))
preds <- rep(0,446)  # Initialize prediction vector
preds[probs>0.5] <- 1 # p>0.5 -> 1
preds
table(preds,subtrain$Survived)
(160+70)/446  # Proportion of predictions that are correct:51.6%

# Try the full model on the validation set
probs<-as.vector(predict(lg2,newdata=valid, type="response"))
preds <- rep(0,445)# Initialize prediction vector
preds[probs>0.5] <- 1 # p>0.5 -> 1
preds
table(preds,valid$Survived)
(245+108)/445  # Not bad:79.3%

lg3 <- glm(Survived~Sex+Age+Pclass, data=subtrain, family = "binomial")
summary(lg3)

# Test the full model on the training set
probs<-as.vector(predict(lg3, type="response"))
preds <- rep(0,446)  # Initialize prediction vector
preds[probs>0.5] <- 1 # p>0.5 -> 1
preds
table(preds,subtrain$Survived)
(166+83)/446  # Proportion of predictions that are correct:56%

# Try the full model on the validation set
probs<-as.vector(predict(lg3,newdata=valid, type="response"))
preds <- rep(0,445)# Initialize prediction vector
preds[probs>0.5] <- 1 # p>0.5 -> 1
preds
table(preds,valid$Survived)
(247+108)/445  # Not bad:79.8%

# Develop the final model with the whole training set
#======================Final Model=========================
lg <- glm(Survived~Sex+Age+Pclass, data=train, family = "binomial")
summary(lg)

lg_noage <- glm(Survived~Sex+Pclass, data=train, family = "binomial")
summary(lg_noage)

# Use the final model to make predictions on test set
probs<- predict(lg, newdata = test)
probs <- as.data.frame(probs)

#Some passengers have missing ages, make a seperate model without age as an explanatory variable
probs_noage <- predict(lg_noage,newdata = test)
probs_noage <- as.data.frame(probs_noage)

for (i in 1:nrow(probs)){
  if (is.na(probs[i,])){
    probs[i,] = probs_noage[i,]
  }
}
probs
mypreds <- rep(0,418)  # Initialize prediction vector
mypreds[probs>0.5] <- 1 # p>0.5 -> 1

mypreds <- cbind(test$PassengerId,mypreds)
mypreds

#Write out the predictions to a csv file
write.table(mypreds, file = "wd3fg_submissions.csv", row.names=F, col.names=c("PassengerId","Survived"), sep=",")

