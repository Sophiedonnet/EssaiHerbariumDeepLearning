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

 



####################################################################"
######################## TEST de EMBEDDING
##################################################################""

#X_dum <- select(X_dum,GENUS,FAMILY_CLEAN,N_LINE,N_CB,train)
#----------------  test set vs training set
N = sum(res_all_INC$train)
train_index = sample(1:N, 12000, replace = F)
test_index = setdiff(1:N, train_index)


train_param <- select(X_dum,-FAMILY_CLEAN,-GENUS) %>% filter(train == T) %>% slice(train_index) %>% select(-train) %>% preProcess("range")
train_X_3 <- predict(train_param, filter(select(X_dum,-FAMILY_CLEAN,-GENUS), train == T) %>% slice(train_index) %>% select(-train)) %>% as.matrix()
dim(train_X_3)



train_X_1 <-  select(X_dum,FAMILY_CLEAN,train)  %>% slice(train_index) %>% select(-train) %>% as.matrix()
train_X_2 <-  select(X_dum,GENUS,train)  %>% slice(train_index) %>% select(-train) %>% as.matrix()


test_X_3 <- predict(train_param, filter(select(X_dum,-FAMILY_CLEAN,-GENUS), train == T) %>% slice(test_index) %>%  select(-train)) %>% as.matrix()
test_X_1 <- select(X_dum,FAMILY_CLEAN,train)  %>% slice(test_index) %>% select(-train) %>% as.matrix()
test_X_2 <- select(X_dum,GENUS,train)  %>% slice(test_index) %>% select(-train) %>% as.matrix()


#------------------ Y 
train_Ybin <- filter(res_all_INC, train == T) %>% slice(train_index) %>% select(LC)   %>% transmute(LC = as.numeric(LC))       %>% pull()
test_Ybin  <- filter(res_all_INC, train == T) %>% slice(test_index)  %>% select(LC)   %>% transmute(LC = as.numeric(LC))       %>% pull()

#----------------------------- KERAS model
embedding_size_FAMILY_CLEAN = 20
embedding_size_GENUS = 100



inp1 <- layer_input(shape = c(1), name = 'inp_FAMILY_CLEAN')
inp2 <- layer_input(shape = c(1), name = 'inp_GENUS')
inp3 <- layer_input(shape = c(151), name = 'inp_otherVars')

embedding_out1 <- inp1 %>% layer_embedding(input_dim = 417 + 1, output_dim = embedding_size_FAMILY_CLEAN, input_length = 1, name="embedding_FAMILY_CLEAN") %>%  layer_flatten()

embedding_out2 <- inp2 %>% layer_embedding(input_dim = 16444 + 1, output_dim = embedding_size_GENUS, input_length = 1, name="embedding_GENUS") %>%  layer_flatten()

combined_model <- layer_concatenate(c(embedding_out1, embedding_out2, inp3)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 100, activation = "relu") %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 4, activation = "relu") %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 1, activation = 'sigmoid') 

model <- keras::keras_model(inputs = c(inp1, inp2, inp3), outputs = combined_model)

model %>% compile(loss = "binary_crossentropy", optimizer = "adam", metric="accuracy") 

summary(model)

inputVariables <- list(train_X_1,train_X_2, train_X_3)
testVariables <- list(test_X_1,test_X_2,test_X_3)
model %>% fit(x = inputVariables, y = train_Ybin, epochs = 100, batch_size = 512,validation_data = list(testVariables,test_Ybin ))

