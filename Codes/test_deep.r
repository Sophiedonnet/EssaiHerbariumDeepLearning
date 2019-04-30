
#############################################################################
# first test on herbarium data                                              #
#############################################################################

res_all_INC <- read.csv("data/res_all_INC.csv")


res_all_INC$train = (is.na(res_all_INC$LC)==F & res_all_INC$RL_STATUT!="DD")

library(tidyverse)
X_all = select(res_all_INC, c(1, 11:114)) %>% select( c(4,22:25, 27:59, 61:66, 68:91, 94, 105))
# str(X_all)


# IMPUTE Manually for categorical variables (9 as a new modality) and mean/median for qualitative
library(Hmisc)
index_quali=c(1,39:43,45:69)
X_allquali = select(X_all, index_quali) %>% mutate_all(funs(impute(., "9")))%>% mutate_all(funs(as.factor(.))) 
# str(X_allquali)
# summary(X_allquali)
X_allquanti = select(X_all, -index_quali)%>% mutate_all(funs(impute(.)))
# summary(X_allquanti)

X_all_noNAv1 <- bind_cols(X_allquanti,X_allquali) %>% select(names(X_all))
# summary(X_all_noNAv1)
str(X_all_noNAv1)
for (i in 1:(ncol(X_all_noNAv1)-1)){X_all_noNAv1[,i]= as.numeric(X_all_noNAv1[,i])}

# conversion of categorical to dummies
index_quali=c(1,39:43,45:69)
names_quali=colnames(X_all[, index_quali]) 

library(dummies)
X_dum <-  bind_cols(X_all_noNAv1, dummy(X_all_noNAv1$SUPER_ORDER, sep = "_"))%>% select(-SUPER_ORDER)

library(sjmisc)

X_dum<- (X_all_noNAv1  %>% 
  to_dummy(SUPER_ORDER, suffix = "label") %>% 
  bind_cols(X_all_noNAv1,.) %>% 
  select(-SUPER_ORDER))
# head(X_dum)

for (j in 2:length(names_quali)){
X_dum<- (X_dum  %>%
           to_dummy(names_quali[j], suffix = "label") %>%
           bind_cols(X_dum,.) %>%
           select(-!!names_quali[j]))}

#X_dum <- select(X_dum, -!!names_quali[2:length(names_quali)])

library(keras)
library(tensorflow)

# test set vs training set
N=sum(res_all_INC$train)
train_index =sample(1:N, 12000, replace =F)
test_index = setdiff(1:N, train_index)


# normalizing training sample
library(caret)
train_param <- filter(X_dum, train==T) %>% slice(train_index) %>% select(-train) %>% preProcess("range")
train_X <- predict(train_param, filter(X_dum, train==T) %>% slice(train_index)%>% select(-train)) %>% as.matrix()

# normalizing the test set with the train parameters
test_X <- predict(train_param, filter(X_dum, train==T) %>% slice(test_index)%>% select(-train)) %>% as.matrix()

# preparing the response data (3 responses either 7 modalities, 3 modalities or binary that is LC vs not LC) TRAIN data and TEST data
train_Y7m <- filter(res_all_INC, train==T) %>% slice(train_index) %>% select(RL_NUM)%>% transmute(RL_NUM=RL_NUM-1)%>% 
  pull()%>% to_categorical(7)
train_Y3m <- filter(res_all_INC, train==T) %>% slice(train_index) %>% select(RL_V3) %>% transmute(RL_V3=as.numeric(droplevels(RL_V3))-1) %>% pull()%>% to_categorical(3)
train_Ybin <- filter(res_all_INC, train==T) %>% slice(train_index) %>% select(LC) %>% transmute(LC=as.numeric(LC)) %>% pull()

test_Y7m <- filter(res_all_INC, train==T) %>% slice(test_index) %>% select(RL_NUM)%>% transmute(RL_NUM=RL_NUM-1)%>% 
  pull()%>% to_categorical(7)
test_Y3m <- filter(res_all_INC, train==T) %>% slice(test_index) %>% select(RL_V3) %>% transmute(RL_V3=as.numeric(droplevels(RL_V3))-1) %>% pull()%>% to_categorical(3)
test_Ybin <- filter(res_all_INC, train==T) %>% slice(test_index) %>% select(LC) %>% transmute(LC=as.numeric(LC)) %>% pull()

# library(pryr)
# object_size(train_Xdum)
# class(train_X)
# object_size(train_X)
# object.size(train_X)
# 
# class(train_x)
# object_size(X_all_noNAv1)

# define a model (here linear stack of layers)
model_herbier_7 <- keras_model_sequential() %>% 
  layer_dense(units = ncol(train_X), input_shape = ncol(train_X)) %>% 
  layer_dropout(rate=0.4)%>%
  layer_activation(activation = 'relu') %>% 
  layer_dense(units = 7) %>% 
  layer_activation(activation = 'softmax')%>% 
  compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam', 
  metrics = c('accuracy')
)

model_herbier_3 <- keras_model_sequential() %>% 
  layer_dense(units = ncol(train_X), input_shape = ncol(train_X)) %>% 
  layer_dropout(rate=0.4)%>%
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dropout(rate=0.4)%>%
  layer_dense(units = 3, activation = 'softmax')%>% 
  compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam', 
    metrics = c('accuracy')
  )

model_herbier_bin <- keras_model_sequential() %>% 
  layer_dense(units = ncol(train_X), input_shape = ncol(train_X), activation = "relu") %>% 
  layer_dropout(rate=0.4)%>%
  layer_dense(units = 1, activation = 'sigmoid')%>% 
  compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam', 
    metrics = c('accuracy')
  )


#rm(list=setdiff(ls(), c("model_herbier", "train_Xdum", "train_X", "train_Y")))

lh7 <- model_herbier_7 %>% fit(train_X, train_Y7m, epochs = 100, batch_size = 512, validation_data=list(test_X,test_Y7m ))
lm7 <- model_herbier_7 %>% evaluate(test_X, test_Y7m, batch_size = 128)

lh3 <- model_herbier_3 %>% fit(train_X, train_Y3m, epochs = 100, batch_size = 512, validation_data=list(test_X,test_Y3m ))
lm3 <- model_herbier_3 %>% evaluate(test_X, test_Y3m, batch_size = 128)

lhbin <- model_herbier_bin %>% fit(train_X, train_Ybin, epochs = 100, batch_size = 120, validation_data=list(test_X,test_Ybin ))
lmbin <- model_herbier_bin %>% evaluate(test_X, test_Ybin, batch_size = 128)

res7m <- predict_classes(model_herbier_7, test_X)
table(res7m)
head(res7m)
head(test_Y7m)
