rm(list = ls())
#############################################################################
# first test on herbarium data                                              #
#############################################################################

#-----------------------------------------------------
#------------ packages
#-----------------------------------------------------
library(tidyverse)
library(Hmisc)
library(dummies)
library(sjmisc)
library(keras)
library(tensorflow)
library(caret)

#-----------------------------------------------------
#---------------- datasets ---------------------------
#-----------------------------------------------------


if (!exists('res_all_INC')) { res_all_INC <- read.csv("data/res_all_INC.csv")}

res_all_INC$train = (is.na(res_all_INC$LC) == F & res_all_INC$RL_STATUT != "DD")
X_all <- select(res_all_INC, c(1, 11:114))
#X_all <- X_all %>% select( c(4,7,8,22:25, 27:59, 61:66, 68:91, 94, 105))
X_all <- X_all %>% select( c(4,7,10,22:25, 27:59, 61:66, 68:91, 94, 105))
names(X_all)
# str(X_all)

index_text <- c(2,3)
#---------------- About NA
# IMPUTE Manually for categorical variables (9 as a new modality) and mean/median for qualitative
index_quali <- c(1,41:45,47:71)
index_quali_text <- c(1,2,3,41:45,47:71)


X_allquali = select(X_all, index_quali_text) %>% mutate_all(funs(impute(., "9"))) %>% mutate_all(funs(as.factor(.)))
# str(X_allquali)
# summary(X_allquali)
X_allquanti = select(X_all, -index_quali_text) %>%  mutate_all(funs(impute(.)))
# summary(X_allquanti)

X_all_noNAv1 <- bind_cols(X_allquanti,X_allquali) %>% select(names(X_all))
#summary(X_all_noNAv1)
#str(X_all_noNAv1)
for (i in (1:(ncol(X_all_noNAv1) - 1))[-c(1,2,3)]) {X_all_noNAv1[,i] = as.numeric(X_all_noNAv1[,i])}

head(X_all_noNAv1)
#---------------- Conversion of categorical variables to dummies
index_quali <- c(1,41:45,47:71)
#names_quali = colnames(X_all[, index_quali])
#X_dum <- dummy.data.frame(X_all_noNAv1,names = names_quali,sep = "_")

names_quali_text = colnames(X_all[, c(index_quali)])
X_dum <- dummy.data.frame(X_all_noNAv1,names = names_quali_text,sep = "_")
X_dum$FAMILY_CLEAN <- as.numeric(X_dum$FAMILY_CLEAN)
X_dum$GENUS <- as.numeric(X_dum$GENUS)
#X_dum <- select(X_dum,-FULLNAME)
length(unique(X_all$FAMILY_CLEAN))
length(unique(X_all$GENUS))

X_dum_2 <- select(X_dum,GENUS,FAMILY_CLEAN,train)
#----------------  test set vs training set
N = sum(res_all_INC$train)
train_index = sample(1:N, 12000, replace = F)
test_index = setdiff(1:N, train_index)

#--------------------------------------------------------
#--------------------- normalizing training sample for numeric variables only
#--------------------------------------------------------
#decal  <- 424-8
#train_param <- select(X_dum, decal + c(11:47,154)) %>% filter( train == T) %>% slice(train_index) %>% select(-train) %>% preProcess("range")
#train_X0 <- predict(train_param, filter(select(X_dum, c(11:47,154)), train == T) %>% slice(train_index) %>% select(-train))

train_param <- filter(X_dum, train == T) %>% slice(train_index) %>% select(-train) %>% preProcess("range")
train_X <- predict(train_param, filter(X_dum, train == T) %>% slice(train_index) %>% select(-train)) %>% as.matrix()


#train_X1 <- select(X_dum, c(1:10, 48:154)) %>% filter( train == T)  %>% slice(train_index) %>% select(-train)
#train_X <- bind_cols(train_X0, train_X1)

###########" ON NORMALISE les variables qualitatives aussi?????

#-----------------------------------------------------------------------
#-------------------  normalizing the test set with the train parameters
#-----------------------------------------------------------------------


test_X <- predict(train_param, filter(X_dum, train == T) %>% slice(test_index) %>%  select(-train))%>% as.matrix()


#---------------- preparing the response data (3 responses either 7 modalities, 3 modalities or binary that is LC vs not LC) TRAIN data and TEST data
#train_Y7m  <- filter(res_all_INC, train == T) %>% slice(train_index) %>% select(RL_NUM) %>% transmute(RL_NUM = RL_NUM - 1)                       %>% pull() %>% to_categorical(7)
#train_Y3m  <- filter(res_all_INC, train == T) %>% slice(train_index) %>% select(RL_V3)  %>% transmute(RL_V3 = as.numeric(droplevels(RL_V3)) - 1) %>% pull() %>% to_categorical(3)
train_Ybin <- filter(res_all_INC, train == T) %>% slice(train_index) %>% select(LC)     %>% transmute(LC = as.numeric(LC))                       %>% pull()
#test_Y7m   <- filter(res_all_INC, train == T) %>% slice(test_index)  %>% select(RL_NUM) %>% transmute(RL_NUM = RL_NUM - 1)                       %>% pull() %>% to_categorical(7)
#test_Y3m   <- filter(res_all_INC, train == T) %>% slice(test_index)  %>% select(RL_V3)  %>% transmute(RL_V3 = as.numeric(droplevels(RL_V3)) - 1) %>% pull() %>% to_categorical(3)
test_Ybin  <- filter(res_all_INC, train == T) %>% slice(test_index)  %>% select(LC)     %>% transmute(LC = as.numeric(LC))                       %>% pull()

# library(pryr)
# object_size(train_Xdum)
# class(train_X)
# object_size(train_X)
# object.size(train_X)
#
# class(train_x)
# object_size(X_all_noNAv1)
#
#
# #-----------------------------------------------------------------------
# # Define neural networs  (here linear stack of layers)
# #-----------------------------------------------------------------------
# model_herbier_7 <- keras_model_sequential() %>%
#   layer_dense(units = ncol(train_X), input_shape = ncol(train_X)) %>%
#   layer_dropout(rate = 0.4) %>%
#   layer_activation(activation = 'relu') %>%
#   layer_dense(units = 7) %>%
#   layer_activation(activation = 'softmax') %>%
#   compile(
#     loss = 'categorical_crossentropy',
#     optimizer = 'adam',
#     metrics = c('accuracy')
#   )
#
# model_herbier_3 <- keras_model_sequential() %>%
#   layer_dense(units = ncol(train_X), input_shape = ncol(train_X)) %>%
#   layer_dropout(rate = 0.4) %>%
#   layer_dense(units = 64, activation = 'relu') %>%
#   layer_dropout(rate = 0.4) %>%
#   layer_dense(units = 3, activation = 'softmax') %>%
#   compile(
#     loss = 'categorical_crossentropy',
#     optimizer = 'adam',
#     metrics = c('accuracy')
#   )

model_herbier_bin <- keras_model_sequential() %>%
  layer_dense(units = ncol(train_X), input_shape = ncol(train_X), activation = "relu") %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 100, activation = "relu") %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 1, activation = 'sigmoid') %>%
  compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = c('accuracy')
  )


#rm(list=setdiff(ls(), c("model_herbier", "train_Xdum", "train_X", "train_Y")))


#lh7 <- model_herbier_7 %>% fit(train_X, train_Y7m, epochs = 100, batch_size = 512, validation_data = list(test_X,test_Y7m ))
#lm7 <- model_herbier_7 %>% evaluate(test_X, test_Y7m, batch_size = 128)

#lh3 <- model_herbier_3 %>% fit(train_X, train_Y3m, epochs = 100, batch_size = 512, validation_data = list(test_X,test_Y3m ))
#lm3 <- model_herbier_3 %>% evaluate(test_X, test_Y3m, batch_size = 128)

lhbin <- model_herbier_bin %>% fit(train_X, train_Ybin, epochs = 100, batch_size = 120, validation_data = list(test_X,test_Ybin ))
lmbin <- model_herbier_bin %>% evaluate(test_X, test_Ybin, batch_size = 128)

resbin <- predict_classes(model_herbier_bin, test_X)
table(resbin,test_Ybin)




####################################################################"
######################## TEST de EMBEDDING
##################################################################""

X_dum_2 <- select(X_dum,GENUS,train)
#----------------  test set vs training set
N = sum(res_all_INC$train)
train_index = sample(1:N, 12000, replace = F)
test_index = setdiff(1:N, train_index)

train_X <- slice(X_dum_2,train_index) %>% select(-train) %>% as.matrix()
dim(train_X)
test_X <- slice(X_dum_2,test_index) %>% select(-train) %>% as.matrix()



train_Ybin <- filter(res_all_INC, train == T) %>% slice(train_index) %>% select(LC)     %>% transmute(LC = as.numeric(LC))                       %>% pull()
test_Ybin  <- filter(res_all_INC, train == T) %>% slice(test_index)  %>% select(LC)     %>% transmute(LC = as.numeric(LC))                       %>% pull()



model_embedding_ <- keras_model_sequential()
model_embedding %>% 
  layer_embedding(input_dim = 16444, output_dim = 16) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model_embedding %>% summary()


model_embedding %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)

lhbin <- model_embedding %>% fit(train_X, train_Ybin, epochs = 40, batch_size = 512, validation_data = list(test_X,test_Ybin ))
lmbin <- model_embedding %>% evaluate(test_X, test_Ybin, batch_size = 128)
