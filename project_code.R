library(FSelector)
library(Boruta)
library(RWeka)
library(caret)
library(rsample)
library(corrr)
library(ggplot2)
library(e1071)  # For SVM
library(nnet)   # For neural networks
library(rpart)  # For decision trees
library(gbm)    # For gradient boosting

getwd()
df<- read.csv('C:/Users/shrad/Desktop/BU/CS699_Dataminning/Project/preprocessing_data.csv')
str(df)
summary(df)
df$Class <- factor(df$Class) 
levels(df$Class) <- c("N", "Y")


############## Splitting the data into training and testing data ############################
set.seed(31)
split <- initial_split(df, prop = 0.66, strata = Class)
train <- training(split)
test <- testing(split)

# Writing sampled data to CSV
write.csv(train, "train.csv", row.names = FALSE)
write.csv(test, "test.csv", row.names = FALSE)
table(train$Class)


###########################################################################################################################

###############  Synthetic Minority Over-sampling Technique (SMOTE)  ############

###########################################################################################################################

library(ROSE)
set.seed(31)
smote_data <- ROSE(Class ~ ., data = train, N = 2555 * 2, seed = 1)$data
table(smote_data$Class)


###########################################################################################################################

# feature selection technique 1 (Info gain)

###########################################################################################################################

info_gain_result <- information.gain(Class ~ ., data = smote_data)
info_gain_df <- as.data.frame(info_gain_result)
info_gain_df$Feature = rownames(info_gain_df)
sorted_info_gain_df <- info_gain_df[order(-info_gain_df$attr_importance), ]
print(sorted_info_gain_df)
feature_names <- sorted_info_gain_df$Feature
print(feature_names)
feature_1 = c(
  "DECIDE", "CHCCOPD2", "PHYSHLTH", "CHCKDNY2", "BLIND",
  "X_CASTHM1", "X_RFHLTH", "X_ASTHMS1", "ACETTHEM", "X_LTASTH1",
  "HIVRISK5", "ACEHVSEX", "ACETOUCH", "MEDCOST", "ACEPRISN",
  "ASTHMA3", "GENHLTH", "ACEDEPRS", "SLEPTIM1", "X_PHYS14D",
  "DIABETE4", "CVDINFR4", "CVDSTRK3", "X_DRDXAR2", "ACEDRUGS"
)

feature_1



###########################################################################################################################

# feature selection technique 2 (CFS)

###########################################################################################################################

cfs_subset <- cfs(Class ~ ., data = smote_data)
print(cfs_subset)
feature_2 = c(
  "ACEDEPRS", "ACEDIVRC", "ACEDRINK", "ACEDRUGS", "ACEHVSEX", "ACEPRISN", "ACETOUCH",
  "ACETTHEM", "ASTHMA3", "BLIND", "CHCCOPD2", "CHCKDNY2", "CVDINFR4", "CVDSTRK3",
  "DECIDE", "DIABETE4", "GENHLTH", "HIVRISK5", "MEDCOST", "PHYSHLTH", "SLEPTIM1",
  "X_ASTHMS1", "X_CASTHM1", "X_DRDXAR2", "X_LTASTH1", "X_PHYS14D", "X_RFHLTH"
)

###########################################################################################################################

# feature selection technique 3 (Recursive Feature Elimination (RFE))

###########################################################################################################################

library(randomForest) 
library(caret)

control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
rfe_results <- rfe(smote_data[, -which(names(smote_data) == "Class")], 
                   smote_data$Class, 
                   sizes = c(1:20), 
                   rfeControl = control)
print(rfe_results)
top_features <- predictors(rfe_results)
print(top_features)
feature_3 <- c(
  "DECIDE", "ACETTHEM", "HIVRISK5", "X_RFHLTH", "X_CASTHM1", "CHCKDNY2", "PHYSHLTH",
  "CHCCOPD2", "BLIND", "ACEHVSEX", "ACEPRISN", "X_ASTHMS1", "ACETOUCH", "MEDCOST",
  "X_LTASTH1", "ACEDEPRS", "GENHLTH", "X_PHYS14D", "DIABETE4", "ASTHMA3", "SLEPTIM1",
  "CVDSTRK3", "ACEDRINK", "X_SEX", "ACEDIVRC", "ACEDRUGS", "X_DRDXAR2", "CVDINFR4",
  "X_MICHD", "X_SMOKER3", "INCOME2"
)

###########################################################################################################################

# feature selection technique 4 (Boruta)

###########################################################################################################################
library(Boruta)
set.seed(123)
boruta_output <- Boruta(Class ~ ., data = smote_data, doTrace = 2, maxRuns = 100)
print(boruta_output)
final_features <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(final_features)
feature_4 <- c(
  "ACEDEPRS", "ACEDIVRC", "ACEDRINK", "ACEDRUGS", "ACEHURT1", "ACEHVSEX", "ACEPRISN", "ACEPUNCH", "ACESWEAR", "ACETOUCH", 
  "ACETTHEM", "ASTHMA3", "BLIND", "CHCCOPD2", "CHCKDNY2", "CHECKUP1", "CVDINFR4", "CVDSTRK3", "DECIDE", "DIABETE4", 
  "DRNKANY5", "EMPLOY1", "FLUSHOT7", "GENHLTH", "HAVARTH4", "HIVRISK5", "HLTHPLN1", "HTIN4", "INCOME2", "MARITAL", 
  "MEDCOST", "PHYSHLTH", "RENTHOM1", "SEXVAR", "SLEPTIM1", "SMOKE100", "X_AGE_G", "X_AIDTST4", "X_ASTHMS1", "X_BMI5CAT", 
  "X_CASTHM1", "X_CHLDCNT", "X_DRDXAR2", "X_HCVU651", "X_HISPANC", "X_INCOMG", "X_LTASTH1", "X_METSTAT", "X_MICHD", "X_PHYS14D", 
  "X_RACE", "X_RFBING5", "X_RFDRHV7", "X_RFHLTH", "X_SEX", "X_SMOKER3", "X_TOTINDA"
)



###########################################################################################################################
###########################################################################################################################

# Model training using feature_1

###########################################################################################################################
###########################################################################################################################
# J48 and Rcart training
features_formula <- as.formula(paste("Class ~", paste(feature_1, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)

# J48 Model Training with feature selection and corrected target variable using smote_data
model_J48_tuneLength <- train(features_formula, data = smote_data, method = "J48", trControl = train_control, tuneLength = 4)
print(model_J48_tuneLength)
plot(model_J48_tuneLength)
test_pred_J48_tuneLength <- predict(model_J48_tuneLength, newdata = test[, c(feature_1, "Class")])
confusionMatrix(test_pred_J48_tuneLength, test$Class)

# Training J48 with custom tuneGrid using smote_data
model_J48_tuneGrid <- train(features_formula, data = smote_data, method = "J48", trControl = train_control, tuneGrid = expand.grid(C = c(0.01, 0.25, 0.5), M = 1:4))
print(model_J48_tuneGrid)
plot(model_J48_tuneGrid)
test_pred_J48_tuneGrid <- predict(model_J48_tuneGrid, newdata = test[, c(feature_1, "Class")])
confusionMatrix(test_pred_J48_tuneGrid, test$Class)


###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_1, "Class")]
predict_J48 <- predict(model_J48_tuneGrid, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_J48, reference = as.factor(TestInfo$Class))
performance_measures
conf_matrix <- confusionMatrix(predict_J48, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)

library(pROC)
predict_J48_probs <- predict(model_J48_tuneGrid, newdata = TestInfo, type = "prob")
predict_J48_Y_probs <- predict_J48_probs[, "Y"]
roc_result <- roc(TestInfo$Class, predict_J48_Y_probs, levels=c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve")

###########################################################################################################################
###########################################################################################################################
## Logistic Regression
features_formula_logistic <- as.formula(paste("Class ~", paste(feature_1, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)
logisticModel <- train(features_formula_logistic, data = smote_data, method = "glm",
                       family = "binomial",
                       trControl = train_control,
                       preProcess = c("center", "scale"))
print(logisticModel)
test_pred_model_logisticModel<- predict(logisticModel, newdata = test[, c(feature_1, "Class")])
confusionMatrix(test_pred_model_logisticModel, test$Class)



###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_1, "Class")]
predict_logisticModel <- predict(logisticModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_logisticModel, reference = as.factor(TestInfo$Class))
performance_measures
conf_matrix <- confusionMatrix(predict_logisticModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



#ROC
ROC <- predict(logisticModel, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(TestInfo$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for LOGREGX")
###########################################################################################################################
###########################################################################################################################
## KNN
features_formula_knn <- as.formula(paste("Class ~", paste(feature_1, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)

# First KNN Model with tuneLength
knnModel <- train(features_formula_knn, data = smote_data, method = "knn",
                  trControl = train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 200)

knnModel
plot(knnModel)

# Predicting and evaluating performance
test_pred <- predict(knnModel, newdata = test)
confusionMatrix(test_pred, test$Class)

# Second KNN Model with custom tuneGrid
knnGrid <- expand.grid(k = seq(1, 100, 2))
knnModel <- train(features_formula_knn, data = smote_data, method = "knn",
                  trControl = train_control,
                  preProcess = c("center", "scale"),
                  tuneGrid = knnGrid)

knnModel
plot(knnModel)


###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_1, "Class")]
predict_knnModel <- predict(knnModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_knnModel, reference = as.factor(TestInfo$Class))
performance_measures

predict_knnModel_probs <- predict(knnModel, newdata = TestInfo, type = "prob")
predict_Y_probs <- predict_knnModel_probs[, "Y"]
roc_result <- roc(TestInfo$Class, predict_Y_probs, levels=c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve")

conf_matrix <- confusionMatrix(predict_knnModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)


###########################################################################################################################
###########################################################################################################################
## XGBOOST
xgb_trcontrol = trainControl(
  method = "cv", number = 10,
  summaryFunction = defaultSummary
)

xgbGrid <- expand.grid(
  nrounds = seq(from = 50, to = 300, by = 50),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2, 3),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)
library(xgboost)
set.seed(123)
xgbModel <- caret::train(x = smote_data[, feature_1], 
                         y = smote_data$Class,
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         metric = "Spec",
                         verbose = FALSE,
                         trControl = xgb_trcontrol)
print(xgbModel)

###########################################################################################################################
###########################################################################################################################

TestInfo = test[,c(feature_1, "Class")]
predict_xgbModel <- predict(xgbModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_xgbModel, reference = as.factor(TestInfo$Class))
performance_measures
conf_matrix <- confusionMatrix(predict_xgbModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)


#ROC
ROC <- predict(xgbModel, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(TestInfo$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for xgboost")

###########################################################################################################################
###########################################################################################################################
# Gradient Boosting
ctrl <- trainControl(method = "cv",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (2:10) * 10,
                       shrinkage = c(.001, .01),
                       n.minobsinnode = 3)

set.seed(31)
gbmFit <- caret::train(x = smote_data[, feature_1], 
                       y = smote_data$Class,
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)
print(gbmFit)

###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_1, "Class")]
predict_gbmFit <- predict(gbmFit, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_gbmFit, reference = as.factor(TestInfo$Class))
performance_measures
conf_matrix <- confusionMatrix(predict_gbmFit, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)

#ROC
predict_gbmFit_probs <- predict(gbmFit, newdata = TestInfo, type = "prob")
predict_Y_probs <- predict_gbmFit_probs[, "Y"]
roc_result <- roc(response = as.factor(TestInfo$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for GBDT")


###########################################################################################################################
###########################################################################################################################
# Random forest
ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     summaryFunction = twoClassSummary, 
                     classProbs = TRUE, 
                     savePredictions = TRUE)

mtryValues <- seq(2, length(feature_1), by = 1)

set.seed(31)
rfFit <- caret::train(features_formula, 
                      data = smote_data, 
                      method = "rf",
                      tuneGrid = expand.grid(mtry = mtryValues),
                      importance = TRUE,
                      metric = "ROC",
                      trControl = ctrl,
                      ntree = 500)
print(rfFit)



###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_1, "Class")]
predict_rfFit <- predict(rfFit, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_rfFit, reference = as.factor(TestInfo$Class))
performance_measures

#ROC
ROC <- predict(rfFit, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(TestInfo$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for rANDOM FOREST")


conf_matrix <- confusionMatrix(predict_rfFit, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)


###########################################################################################################################
###########################################################################################################################

# Model training using feature_2

###########################################################################################################################
###########################################################################################################################
# J48 and Rcart training
features_formula <- as.formula(paste("Class ~", paste(feature_2, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)

# J48 Model Training with feature selection and corrected target variable using smote_data
model_J48_tuneLength <- train(features_formula, data = smote_data, method = "J48", trControl = train_control, tuneLength = 4)
print(model_J48_tuneLength)
plot(model_J48_tuneLength)
test_pred_J48_tuneLength <- predict(model_J48_tuneLength, newdata = test[, c(feature_2, "Class")])
confusionMatrix(test_pred_J48_tuneLength, test$Class)

# Training J48 with custom tuneGrid using smote_data
model_J48_tuneGrid <- train(features_formula, data = smote_data, method = "J48", trControl = train_control, tuneGrid = expand.grid(C = c(0.01, 0.25, 0.5), M = 1:4))
print(model_J48_tuneGrid)
plot(model_J48_tuneGrid)
test_pred_J48_tuneGrid <- predict(model_J48_tuneGrid, newdata = test[, c(feature_2, "Class")])
confusionMatrix(test_pred_J48_tuneGrid, test$Class)

###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_2, "Class")]
predict_model_J48_tuneLength <- predict(model_J48_tuneLength, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_model_J48_tuneLength, reference = as.factor(TestInfo$Class))
performance_measures
conf_matrix <- confusionMatrix(predict_model_J48_tuneLength, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)


###########################################################################################################################
###########################################################################################################################
## Logistic Regression
features_formula_logistic <- as.formula(paste("Class ~", paste(feature_2, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)
logisticModel <- train(features_formula_logistic, data = smote_data, method = "glm",
                       family = "binomial",
                       trControl = train_control,
                       preProcess = c("center", "scale"))
print(logisticModel)
test_pred_model_logisticModel<- predict(logisticModel, newdata = test[, c(feature_2, "Class")])
confusionMatrix(test_pred_model_logisticModel, test$Class)



###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_2, "Class")]
predict_logisticModel <- predict(logisticModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_logisticModel, reference = as.factor(TestInfo$Class))
performance_measures

#ROC
ROC <- predict(logisticModel, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(TestInfo$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for LoGReg")

conf_matrix <- confusionMatrix(predict_logisticModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)


###########################################################################################################################
###########################################################################################################################
## KNN
features_formula_knn <- as.formula(paste("Class ~", paste(feature_2, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)

# First KNN Model with tuneLength
knnModel <- train(features_formula_knn, data = smote_data, method = "knn",
                  trControl = train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 200)

knnModel
plot(knnModel)

# Predicting and evaluating performance
test_pred <- predict(knnModel, newdata = test)
confusionMatrix(test_pred, test$Class)

# Second KNN Model with custom tuneGrid
knnGrid <- expand.grid(k = seq(1, 100, 2))
knnModel <- train(features_formula_knn, data = smote_data, method = "knn",
                  trControl = train_control,
                  preProcess = c("center", "scale"),
                  tuneGrid = knnGrid)

knnModel
plot(knnModel)


###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_2, "Class")]
predict_knnModel <- predict(knnModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_knnModel, reference = as.factor(TestInfo$Class))
performance_measures
conf_matrix <- confusionMatrix(predict_knnModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)


#ROC
ROC <- predict(knnmodel2, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(TestInfo$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for KNN")
###########################################################################################################################
###########################################################################################################################
# Gradient Boosting
set.seed(31)
ctrl <- trainControl(method = "cv",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (2:10) * 10,
                       shrinkage = c(.001, .01),
                       n.minobsinnode = 3)

set.seed(31)
gbmFit <- caret::train(x = smote_data[, feature_2], 
                       y = smote_data$Class,
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)
print(gbmFit)

###########################################################################################################################
###########################################################################################################################
tscfs = test[,c(feature_2, "Class")]
test_pred <- predict(gbmFit, newdata = tscfs)
test_pred = as.factor(test_pred)
levels(test_pred)
test_pred
performance_measures = confusionMatrix(test_pred, as.factor(tscfs$Class))
performance_measures

#ROC
ROC <- predict(gbmFit, newdata = tscfs, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for GBDT")

cm <- performance_measures$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################
# Random forest
features_formula <- as.formula(paste("Class ~", paste(feature_2, collapse = " + ")))
set.seed(31)
ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     summaryFunction = twoClassSummary, 
                     classProbs = TRUE, 
                     savePredictions = TRUE)

mtryValues <- seq(2, length(feature_2), by = 1)

set.seed(31)
rfFit <- caret::train(features_formula, 
                      data = smote_data, 
                      method = "rf",
                      tuneGrid = expand.grid(mtry = mtryValues),
                      importance = TRUE,
                      metric = "ROC",
                      trControl = ctrl,
                      ntree = 500)
print(rfFit)


###########################################################################################################################
###########################################################################################################################
tscfs = test[,c(feature_2, "Class")]
test_pred <- predict(rfFit, newdata = tscfs)
test_pred = as.factor(test_pred)
levels(test_pred)
test_pred
performance_measures = confusionMatrix(test_pred, as.factor(tscfs$Class))
performance_measures
cm <- performance_measures$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



#ROC
ROC <- predict(rfFit, newdata = tscfs, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for RandomForest")

###########################################################################################################################
###########################################################################################################################
## XGBOOST
set.seed(31)
xgb_trcontrol = trainControl(
  method = "cv", number = 10,
  summaryFunction = defaultSummary
)

xgbGrid <- expand.grid(
  nrounds = seq(from = 50, to = 300, by = 50),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2, 3),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)
library(xgboost)
set.seed(123)
xgbModel <- caret::train(x = smote_data[, feature_2], 
                         y = smote_data$Class,
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         metric = "Spec",
                         verbose = FALSE,
                         trControl = xgb_trcontrol)
print(xgbModel)


###########################################################################################################################
###########################################################################################################################
tscfs = test[,c(feature_2, "Class")]
test_pred <- predict(xgbModel, newdata = tscfs)
test_pred = as.factor(test_pred)
levels(test_pred)
test_pred


#ROC
ROC <- predict(xgbModel, newdata = tscfs, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for XGBOOST")


performance_measures = confusionMatrix(test_pred, as.factor(tscfs$Class))
performance_measures
cm <- performance_measures$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################

# Model training using feature_3

###########################################################################################################################
###########################################################################################################################
## XGBOOST
set.seed(31)
xgb_trcontrol = trainControl(
  method = "cv", 
  number = 10,
  summaryFunction = defaultSummary,
  classProbs = TRUE,  
  savePredictions = "final"  
)

xgbGrid <- expand.grid(
  nrounds = seq(from = 50, to = 300, by = 50),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2, 3),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)

library(xgboost)
set.seed(123)
xgbModel <- caret::train(x = smote_data[, feature_3], 
                         y = smote_data$Class,
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         metric = "Spec",
                         verbose = FALSE,
                         trControl = xgb_trcontrol)
print(xgbModel)


###########################################################################################################################
###########################################################################################################################
tscfs = test[,c(feature_3, "Class")]
test_pred <- predict(xgbModel, newdata = tscfs)
test_pred = as.factor(test_pred)
levels(test_pred)
test_pred


#ROC
ROC <- predict(xgbModel, newdata = tscfs, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for XGBOOST")


performance_measures = confusionMatrix(test_pred, as.factor(tscfs$Class))
performance_measures
cm <- performance_measures$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################
# Random forest
features_formula <- as.formula(paste("Class ~", paste(feature_3, collapse = " + ")))
set.seed(31)
ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     summaryFunction = twoClassSummary, 
                     classProbs = TRUE, 
                     savePredictions = TRUE)

mtryValues <- seq(2, length(feature_3), by = 1)

set.seed(31)
rfFit <- caret::train(features_formula, 
                      data = smote_data, 
                      method = "rf",
                      tuneGrid = expand.grid(mtry = mtryValues),
                      importance = TRUE,
                      metric = "ROC",
                      trControl = ctrl,
                      ntree = 500)
print(rfFit)


###########################################################################################################################
###########################################################################################################################
tscfs = test[,c(feature_3, "Class")]
test_pred <- predict(rfFit, newdata = tscfs)
test_pred = as.factor(test_pred)
levels(test_pred)
test_pred
performance_measures = confusionMatrix(test_pred, as.factor(tscfs$Class))
performance_measures

#ROC
ROC <- predict(rfFit, newdata = tscfs, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for RF")


cm <- performance_measures$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################
## KNN
features_formula_knn <- as.formula(paste("Class ~", paste(feature_3, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)

# First KNN Model with tuneLength
knnModel <- train(features_formula_knn, data = smote_data, method = "knn",
                  trControl = train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 200)

knnModel
plot(knnModel)

# Predicting and evaluating performance
test_pred <- predict(knnModel, newdata = test)
confusionMatrix(test_pred, test$Class)

# Second KNN Model with custom tuneGrid
knnGrid2 <- expand.grid(k = seq(1, 100, 2))
knnModel2 <- train(features_formula_knn, data = smote_data, method = "knn",
                  trControl = train_control,
                  preProcess = c("center", "scale"),
                  tuneGrid = knnGrid2)

knnModel2
plot(knnModel2)


###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_3, "Class")]
predict_knnModel <- predict(knnModel2, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_knnModel, reference = as.factor(TestInfo$Class))
performance_measures


#ROC
ROC <- predict(knnModel2, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for KNN")


conf_matrix <- confusionMatrix(predict_knnModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)




###########################################################################################################################
###########################################################################################################################
# Gradient Boosting
set.seed(31)
ctrl <- trainControl(method = "cv",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (2:10) * 10,
                       shrinkage = c(.001, .01),
                       n.minobsinnode = 3)

set.seed(31)
gbmFit <- caret::train(x = smote_data[, feature_3], 
                       y = smote_data$Class,
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)
print(gbmFit)

###########################################################################################################################
###########################################################################################################################
tscfs = test[,c(feature_3, "Class")]
test_pred <- predict(gbmFit, newdata = tscfs)
test_pred = as.factor(test_pred)
levels(test_pred)
test_pred
performance_measures = confusionMatrix(test_pred, as.factor(tscfs$Class))
performance_measures

#ROC
ROC <- predict(gbmFit, newdata = tscfs, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for GBDT")


cm <- performance_measures$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################
# J48 and Rcart training
features_formula <- as.formula(paste("Class ~", paste(feature_3, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)

# J48 Model Training with feature selection and corrected target variable using smote_data
model_J48_tuneLength <- train(features_formula, data = smote_data, method = "J48", trControl = train_control, tuneLength = 4)
print(model_J48_tuneLength)
plot(model_J48_tuneLength)
test_pred_J48_tuneLength <- predict(model_J48_tuneLength, newdata = test[, c(feature_3, "Class")])
confusionMatrix(test_pred_J48_tuneLength, test$Class)

# Training J48 with custom tuneGrid using smote_data
model_J48_tuneGrid <- train(features_formula, data = smote_data, method = "J48", trControl = train_control, tuneGrid = expand.grid(C = c(0.01, 0.25, 0.5), M = 1:4))
print(model_J48_tuneGrid)
plot(model_J48_tuneGrid)
test_pred_J48_tuneGrid <- predict(model_J48_tuneGrid, newdata = test[, c(feature_3, "Class")])
confusionMatrix(test_pred_J48_tuneGrid, test$Class)

###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_3, "Class")]
predict_J48 <- predict(model_J48_tuneGrid, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_J48, reference = as.factor(TestInfo$Class))
performance_measures


#ROC
ROC <- predict(model_J48_tuneGrid, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(TestInfo$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for J48")


conf_matrix <- confusionMatrix(predict_J48, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################
## Logistic Regression
features_formula_logistic <- as.formula(paste("Class ~", paste(feature_3, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)
logisticModel <- train(features_formula_logistic, data = smote_data, method = "glm",
                       family = "binomial",
                       trControl = train_control,
                       preProcess = c("center", "scale"))
print(logisticModel)
test_pred_model_logisticModel<- predict(logisticModel, newdata = test[, c(feature_3, "Class")])
confusionMatrix(test_pred_model_logisticModel, test$Class)



###########################################################################################################################
###########################################################################################################################

TestInfo = test[,c(feature_3, "Class")]
predict_logisticModel <- predict(logisticModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_logisticModel, reference = as.factor(TestInfo$Class))
performance_measures

#ROC
ROC <- predict(logisticModel, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(TestInfo$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for Logistic Regression")


conf_matrix <- confusionMatrix(predict_logisticModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)


###########################################################################################################################
###########################################################################################################################
## KNN
features_formula_knn <- as.formula(paste("Class ~", paste(feature_3, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)

# First KNN Model with tuneLength
knnModel <- train(features_formula_knn, data = smote_data, method = "knn",
                  trControl = train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 200)

knnModel
plot(knnModel)

# Predicting and evaluating performance
test_pred <- predict(knnModel, newdata = test)
confusionMatrix(test_pred, test$Class)

# Second KNN Model with custom tuneGrid
knnGrid <- expand.grid(k = seq(1, 100, 2))
knnModel <- train(features_formula_knn, data = smote_data, method = "knn",
                  trControl = train_control,
                  preProcess = c("center", "scale"),
                  tuneGrid = knnGrid)

knnModel
plot(knnModel)


###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_3, "Class")]
predict_knnModel <- predict(knnModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_knnModel, reference = as.factor(TestInfo$Class))
performance_measures
conf_matrix <- confusionMatrix(predict_knnModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)


###########################################################################################################################
###########################################################################################################################





############################################################################################################################
############################################################################################################################
############################################################################################################################

# Sampling using up and feature selection and model training 


#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

library(FSelector)
library(Boruta)
library(RWeka)
library(caret)
library(rsample)
library(corrr)
library(ggplot2)
library(e1071)  # For SVM
library(nnet)   # For neural networks
library(rpart)  # For decision trees
library(gbm)    # For gradient boosting

getwd()
df<- read.csv('C:/Users/shrad/Desktop/BU/CS699_Dataminning/Project/preprocessing_data.csv')
summary(df)
df$Class <- factor(df$Class) 
levels(df$Class) <- c("N", "Y")


############## Splitting the data into training and testing data ############################
set.seed(31)
split <- initial_split(df, prop = 0.66, strata = Class)
train <- training(split)
test <- testing(split)
table(train$Class)


set.seed(31)
# Splitting the data into two parts based on class
data_N <- subset(train, Class == "N")
data_Y <- subset(train, Class == "Y")
data_Y_upsampled <- data_Y[sample(nrow(data_Y), size = nrow(data_N), replace = TRUE), ]
train_upsampled <- rbind(data_N, data_Y_upsampled)
table(train_upsampled$Class)
table(train$Class)


# plot number of cases in undersampled dataset
ggplot(data = train_upsampled, aes(fill = Class))+
  geom_bar(aes(x = Class))+
  ggtitle("Number of samples in each class after undersampling", 
          subtitle="Total samples: 984")+
  xlab("")+
  ylab("Samples")+
  scale_y_continuous(expand = c(0,0))+
  scale_x_discrete(expand = c(0,0))+
  theme(legend.position = "none", 
        legend.title = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank())



###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

# feature selection technique 1 (Info gain)

###########################################################################################################################

info_gain_result <- information.gain(Class ~ ., data = train_upsampled)
info_gain_df <- as.data.frame(info_gain_result)
info_gain_df$Feature = rownames(info_gain_df)
sorted_info_gain_df <- info_gain_df[order(-info_gain_df$attr_importance), ]
print(sorted_info_gain_df)
feature_names <- sorted_info_gain_df$Feature
print(feature_names)
feature_1 <- c(
  "DECIDE", "GENHLTH", "X_PHYS14D", "PHYSHLTH", "X_RFHLTH", "SLEPTIM1", "EMPLOY1", "HAVARTH4", "X_DRDXAR2",
  "ASTHMA3", "X_ASTHMS1", "X_LTASTH1", "HTIN4", "INCOME2", "X_INCOMG", "X_AIDTST4", "X_CASTHM1", "X_SMOKER3",
  "SMOKE100", "CHCCOPD2", "RENTHOM1", "MARITAL", "SEXVAR", "X_SEX", "ACEDEPRS", "X_AGE_G", "ACESWEAR", "ACEHVSEX",
  "MEDCOST", "ACETOUCH", "X_RFSMOK3", "PNEUVAC4", "BLIND", "X_TOTINDA", "X_HCVU651", "ACEPUNCH", "ACETTHEM",
  "ACEDRINK", "ACEPRISN", "DIABETE4", "EDUCA", "X_EDUCAG"
)

feature_1



###########################################################################################################################

# feature selection technique 2 (CFS)

###########################################################################################################################

cfs_subset <- cfs(Class ~ ., data = train_upsampled)
print(cfs_subset)
feature_2 <- c("ACEDEPRS", "ACEHVSEX", "ASTHMA3", "BLIND", "CHCCOPD2", "DECIDE", "EMPLOY1", "GENHLTH", "HAVARTH4", "HTIN4", "MARITAL", "MEDCOST", 
               "PHYSHLTH", "RENTHOM1", "SEXVAR", "SLEPTIM1", "X_AIDTST4", "X_CASTHM1", "X_PHYS14D", "X_RFHLTH", "X_SMOKER3")
feature_2

###########################################################################################################################

# feature selection technique 3 (Recursive Feature Elimination (RFE))

###########################################################################################################################

library(randomForest) 
library(caret)

control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
rfe_results <- rfe(train_upsampled[, -which(names(train_upsampled) == "Class")], 
                   train_upsampled$Class, 
                   sizes = c(1:20), 
                   rfeControl = control)
print(rfe_results)
top_features <- predictors(rfe_results)
print(top_features)
feature_3 <- c("X_STATE", "DECIDE", "HTIN4", "X_DUALUSE", "SLEPTIM1", "X_BMI5CAT", "GENHLTH", "PERSDOC2", "X_AGE_G", "EMPLOY1", "X_PHYS14D", "PHYSHLTH",
               "DRNKANY5", "CHECKUP1", "X_CHLDCNT", "X_EXTETH3", "X_RACE", "MARITAL", "DIABETE4", "INCOME2", "X_RFBMI5", "X_IMPRACE", "FLUSHOT7", "PNEUVAC4",
               "X_HCVU651", "X_METSTAT", "HIVRISK5", "LASTDEN4", "ACEDEPRS", "DEAF", "CHCOCNCR", "EDUCA", "X_RFDRHV7", "X_EDUCAG", "X_RFHLTH", "DIFFWALK",
               "X_AIDTST4", "SEXVAR", "X_RFBING5", "X_URBSTAT", "X_SEX", "MEDCOST", "CHCKDNY2", "ACEDIVRC", "X_SMOKER3")
feature_3

###########################################################################################################################

# feature selection technique 4 (Boruta)

###########################################################################################################################
library(Boruta)
set.seed(123)
boruta_output <- Boruta(Class ~ ., data = train_upsampled, doTrace = 2, maxRuns = 100)
print(boruta_output)
final_features <- getSelectedAttributes(boruta_output, withTentative = TRUE)
feature_4 <- c("ACEDEPRS", "ACEDIVRC", "ACEDRINK", "ACEDRUGS", "ACEHURT1", "ACEHVSEX", "ACEPRISN", "ACEPUNCH", "ACESWEAR", "ACETOUCH", "ACETTHEM", "ASTHMA3",
               "BLIND", "CHCCOPD2", "CHCKDNY2", "CHCOCNCR", "CHCSCNCR", "CHECKUP1", "CVDINFR4", "CVDSTRK3", "DEAF", "DECIDE", "DIABETE4", "DRNKANY5",
               "EDUCA", "EMPLOY1", "FLUSHOT7", "GENHLTH", "HAVARTH4", "HIVRISK5", "HLTHPLN1", "HTIN4", "INCOME2", "LASTDEN4", "MARITAL", "MEDCOST",
               "PERSDOC2", "PHYSHLTH")
feature_4



###########################################################################################################################
###########################################################################################################################
library(FactoMineR)
library(factoextra)
famd_result <- FAMD(train_upsampled, graph = FALSE)
fviz_eig(famd_result, addlabels = TRUE, ylim = c(0, 50))
fviz_famd_var(famd_result, col.var = "contrib", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), repel = TRUE)
fviz_famd_ind(famd_result, 
              geom = "point", 
              habillage = train_upsampled$Class, 
              palette = c("#00AFBB", "#FC4E07"), 
              repel = TRUE)
var_contributions <- as.data.frame(famd_result$var$contrib)
print(var_contributions[, 1])
important_features <- rowSums(var_contributions[, 1:3])
important_features_sorted <- sort(important_features, decreasing = TRUE)
print(important_features_sorted)
top_features <- names(important_features_sorted)[1:25]
feature_5 <- top_features[top_features != "Class"]
print(feature_5)


feature_5 <- c("X_AGE_G", "ACEPRISN", "ACEHVSEX", "ACETTHEM", 
               "ACEDRUGS", "X_HCVU651", "ACETOUCH", "ACEDEPRS", 
               "ACEHURT1", "X_ASTHMS1", "ACEDRINK", "GENHLTH", 
               "ASTHMA3", "X_LTASTH1", "ACESWEAR", "ACEPUNCH", 
               "HAVARTH4", "X_DRDXAR2", "EMPLOY1", "X_CASTHM1",
               "X_RFHLTH", "ACEDIVRC", "X_MICHD", "INCOME2",  
               "EDUCA")
feature_5

###########################################################################################################################
###########################################################################################################################

# Model training using feature_1


###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
## XGBOOST
xgb_trcontrol = trainControl(
  method = "cv", number = 10,
  summaryFunction = defaultSummary
)

xgbGrid <- expand.grid(
  nrounds = seq(from = 50, to = 300, by = 50),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2, 3),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)
library(xgboost)
set.seed(123)
xgbModel <- caret::train(x = train_upsampled[, feature_1], 
                         y = train_upsampled$Class,
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         metric = "Spec",
                         verbose = FALSE,
                         trControl = xgb_trcontrol)
print(xgbModel)

###########################################################################################################################
###########################################################################################################################

TestInfo = test[,c(feature_1, "Class")]
predict_xgbModel <- predict(xgbModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_xgbModel, reference = as.factor(TestInfo$Class))
performance_measures
conf_matrix <- confusionMatrix(predict_xgbModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)


#ROC
ROC <- predict(xgbModel, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(TestInfo$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for xgboost")

###########################################################################################################################
###########################################################################################################################
# Gradient Boosting
set.seed(31)
ctrl <- trainControl(method = "cv",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (2:10) * 10,
                       shrinkage = c(.001, .01),
                       n.minobsinnode = 3)

set.seed(31)
gbmFit <- caret::train(x = train_upsampled[, feature_1], 
                       y = train_upsampled$Class,
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)
print(gbmFit)

###########################################################################################################################
###########################################################################################################################
tscfs = test[,c(feature_1, "Class")]
test_pred <- predict(gbmFit, newdata = tscfs)
test_pred = as.factor(test_pred)
levels(test_pred)
test_pred
performance_measures = confusionMatrix(test_pred, as.factor(tscfs$Class))
performance_measures


#ROC
ROC <- predict(gbmFit, newdata = tscfs, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for RF")


cm <- performance_measures$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################
# Random forest
features_formula <- as.formula(paste("Class ~", paste(feature_1, collapse = " + ")))
set.seed(31)
ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     summaryFunction = twoClassSummary, 
                     classProbs = TRUE, 
                     savePredictions = TRUE)

mtryValues <- seq(2, length(feature_1), by = 1)

set.seed(31)
rfFit <- caret::train(features_formula, 
                      data = train_upsampled, 
                      method = "rf",
                      tuneLength = 10,
                      importance = TRUE,
                      metric = "ROC",
                      trControl = ctrl,
                      ntree = 500)
print(rfFit)


###########################################################################################################################
###########################################################################################################################
tscfs = test[,c(feature_1, "Class")]
test_pred <- predict(rfFit, newdata = tscfs)
test_pred = as.factor(test_pred)
levels(test_pred)
test_pred
performance_measures = confusionMatrix(test_pred, as.factor(tscfs$Class))
performance_measures

#ROC
ROC <- predict(rfFit, newdata = tscfs, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for RF")


cm <- performance_measures$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################
## KNN
features_formula_knn <- as.formula(paste("Class ~", paste(feature_1, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)

# First KNN Model with tuneLength
knnModel <- train(features_formula_knn, data = train_upsampled, method = "knn",
                  trControl = train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 200)

knnModel
plot(knnModel)

# Predicting and evaluating performance
test_pred <- predict(knnModel, newdata = test)
confusionMatrix(test_pred, test$Class)

# Second KNN Model with custom tuneGrid
knnGrid2 <- expand.grid(k = seq(1, 100, 2))
knnModel2 <- train(features_formula_knn, data = train_upsampled, method = "knn",
                   trControl = train_control,
                   preProcess = c("center", "scale"),
                   tuneGrid = knnGrid2)

knnModel2
plot(knnModel2)


###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_1, "Class")]
predict_knnModel <- predict(knnModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_knnModel, reference = as.factor(TestInfo$Class))
performance_measures

#ROC
ROC <- predict(knnModel, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for RF")


conf_matrix <- confusionMatrix(predict_knnModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)


###########################################################################################################################
###########################################################################################################################
# J48 and Rcart training
features_formula <- as.formula(paste("Class ~", paste(feature_1, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)

# J48 Model Training with feature selection and corrected target variable using smote_data
model_J48_tuneLength <- train(features_formula, data = train_upsampled, method = "J48", trControl = train_control, tuneLength = 4)
print(model_J48_tuneLength)
plot(model_J48_tuneLength)
test_pred_J48_tuneLength <- predict(model_J48_tuneLength, newdata = test[, c(feature_1, "Class")])
confusionMatrix(test_pred_J48_tuneLength, test$Class)

# Training J48 with custom tuneGrid using smote_data
model_J48_tuneGrid <- train(features_formula, data = train_upsampled, method = "J48", trControl = train_control, tuneGrid = expand.grid(C = c(0.01, 0.25, 0.5), M = 1:4))
print(model_J48_tuneGrid)
plot(model_J48_tuneGrid)
test_pred_J48_tuneGrid <- predict(model_J48_tuneGrid, newdata = test[, c(feature_1, "Class")])
confusionMatrix(test_pred_J48_tuneGrid, test$Class)


###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_1, "Class")]
predict_J48 <- predict(model_J48_tuneGrid, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_J48, reference = as.factor(TestInfo$Class))
performance_measures
conf_matrix <- confusionMatrix(predict_J48, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)

library(pROC)
predict_J48_probs <- predict(model_J48_tuneGrid, newdata = TestInfo, type = "prob")
predict_J48_Y_probs <- predict_J48_probs[, "Y"]
roc_result <- roc(TestInfo$Class, predict_J48_Y_probs, levels=c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve")

###########################################################################################################################
###########################################################################################################################
## Logistic Regression
features_formula_logistic <- as.formula(paste("Class ~", paste(feature_1, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)
logisticModel <- train(features_formula_logistic, data = train_upsampled, method = "glm",
                       family = "binomial",
                       trControl = train_control,
                       preProcess = c("center", "scale"))
print(logisticModel)
test_pred_model_logisticModel<- predict(logisticModel, newdata = test[, c(feature_1, "Class")])
confusionMatrix(test_pred_model_logisticModel, test$Class)



###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_1, "Class")]
predict_logisticModel <- predict(logisticModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_logisticModel, reference = as.factor(TestInfo$Class))
performance_measures
conf_matrix <- confusionMatrix(predict_logisticModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)


#ROC
ROC <- predict(logisticModel, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(TestInfo$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for LOGREGX")

###########################################################################################################################
###########################################################################################################################

# Model training using feature_2(CFS)


###########################################################################################################################
###########################################################################################################################


###########################################################################################################################
###########################################################################################################################
# J48 and Rcart training
features_formula <- as.formula(paste("Class ~", paste(feature_2, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)

# J48 Model Training with feature selection and corrected target variable using smote_data
model_J48_tuneLength <- train(features_formula, data = train_upsampled, method = "J48", trControl = train_control, tuneLength = 4)
print(model_J48_tuneLength)
plot(model_J48_tuneLength)
test_pred_J48_tuneLength <- predict(model_J48_tuneLength, newdata = test[, c(feature_1, "Class")])
confusionMatrix(test_pred_J48_tuneLength, test$Class)

# Training J48 with custom tuneGrid using smote_data
model_J48_tuneGrid <- train(features_formula, data = train_upsampled, method = "J48", trControl = train_control, tuneGrid = expand.grid(C = c(0.01, 0.25, 0.5), M = 1:4))
print(model_J48_tuneGrid)
plot(model_J48_tuneGrid)
test_pred_J48_tuneGrid <- predict(model_J48_tuneGrid, newdata = test[, c(feature_1, "Class")])
confusionMatrix(test_pred_J48_tuneGrid, test$Class)


###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_2, "Class")]
predict_J48 <- predict(model_J48_tuneGrid, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_J48, reference = as.factor(TestInfo$Class))
performance_measures
conf_matrix <- confusionMatrix(predict_J48, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)

library(pROC)
predict_J48_probs <- predict(model_J48_tuneGrid, newdata = TestInfo, type = "prob")
predict_J48_Y_probs <- predict_J48_probs[, "Y"]
roc_result <- roc(TestInfo$Class, predict_J48_Y_probs, levels=c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve")

###########################################################################################################################
###########################################################################################################################
## Logistic Regression
features_formula_logistic <- as.formula(paste("Class ~", paste(feature_2, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)
logisticModel <- train(features_formula_logistic, data = train_upsampled, method = "glm",
                       family = "binomial",
                       trControl = train_control,
                       preProcess = c("center", "scale"))
print(logisticModel)
test_pred_model_logisticModel<- predict(logisticModel, newdata = test[, c(feature_2, "Class")])
confusionMatrix(test_pred_model_logisticModel, test$Class)



###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_2, "Class")]
predict_logisticModel <- predict(logisticModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_logisticModel, reference = as.factor(TestInfo$Class))
performance_measures
conf_matrix <- confusionMatrix(predict_logisticModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



#ROC
ROC <- predict(logisticModel, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(TestInfo$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for LOGREGX")

###########################################################################################################################
###########################################################################################################################


# Gradient Boosting
set.seed(31)
ctrl <- trainControl(method = "cv",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = seq(50, 500, by = 50),
                       shrinkage = c(0.001, 0.01, 0.1),
                       n.minobsinnode = c(3, 10, 20))
set.seed(31)
gbmFit <- caret::train(x = train_upsampled[, feature_2], 
                       y = train_upsampled$Class,
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)
print(gbmFit)

###########################################################################################################################
###########################################################################################################################
tscfs = test[,c(feature_2, "Class")]
test_pred <- predict(gbmFit, newdata = tscfs)
test_pred = as.factor(test_pred)
levels(test_pred)
test_pred
performance_measures = confusionMatrix(test_pred, as.factor(tscfs$Class))
performance_measures
cm <- performance_measures$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################
# Random forest
features_formula <- as.formula(paste("Class ~", paste(feature_2, collapse = " + ")))
set.seed(31)
ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     summaryFunction = twoClassSummary, 
                     classProbs = TRUE, 
                     savePredictions = TRUE)

set.seed(31)
rfFit <- caret::train(features_formula, 
                      data = train_upsampled, 
                      method = "rf",
                      tuneLength = 10, 
                      importance = TRUE,
                      metric = "ROC",
                      trControl = ctrl,
                      ntree = 500)
print(rfFit)


###########################################################################################################################
###########################################################################################################################
tscfs = test[,c(feature_2, "Class")]
test_pred <- predict(rfFit, newdata = tscfs)
test_pred = as.factor(test_pred)
levels(test_pred)
test_pred
performance_measures = confusionMatrix(test_pred, as.factor(tscfs$Class))
performance_measures

#ROC
ROC <- predict(rfFit, newdata = tscfs, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for RF")

cm <- performance_measures$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################
## KNN
features_formula_knn <- as.formula(paste("Class ~", paste(feature_2, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)

# First KNN Model with tuneLength
knnModel <- train(features_formula_knn, data = train_upsampled, method = "knn",
                  trControl = train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 200)

knnModel
plot(knnModel)

# Predicting and evaluating performance
test_pred <- predict(knnModel, newdata = test)
confusionMatrix(test_pred, test$Class)

# Second KNN Model with custom tuneGrid
knnGrid2 <- expand.grid(k = seq(1, 100, 2))
knnModel2 <- train(features_formula_knn, data = train_upsampled, method = "knn",
                   trControl = train_control,
                   preProcess = c("center", "scale"),
                   tuneGrid = knnGrid2)

knnModel2


###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_2, "Class")]
predict_knnModel <- predict(knnModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_knnModel, reference = as.factor(TestInfo$Class))
performance_measures

#ROC
ROC <- predict(knnModel, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for KNN")


conf_matrix <- confusionMatrix(predict_knnModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################
## XGBOOST
xgb_trcontrol = trainControl(
  method = "cv", number = 10,
  summaryFunction = defaultSummary
)

xgbGrid <- expand.grid(
  nrounds = seq(from = 50, to = 300, by = 50),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2, 3),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)
library(xgboost)
set.seed(123)
xgbModel <- caret::train(x = train_upsampled[, feature_3], 
                         y = train_upsampled$Class,
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         metric = "Spec",
                         verbose = FALSE,
                         trControl = xgb_trcontrol)
print(xgbModel)

###########################################################################################################################
###########################################################################################################################

TestInfo = test[,c(feature_3, "Class")]
predict_xgbModel <- predict(xgbModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_xgbModel, reference = as.factor(TestInfo$Class))
performance_measures
conf_matrix <- confusionMatrix(predict_xgbModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)


#ROC
ROC <- predict(xgbModel, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(TestInfo$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for xgboost")

###########################################################################################################################
###########################################################################################################################


#######################################################################################################################
###########################################################################################################################
###########################################################################################################################

# Feature _ 3 model training

###########################################################################################################################
###########################################################################################################################

###########################################################################################################################
###########################################################################################################################
## XGBOOST
xgb_trcontrol = trainControl(
  method = "cv", number = 10,
  summaryFunction = defaultSummary
)

xgbGrid <- expand.grid(
  nrounds = seq(from = 50, to = 300, by = 50),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2, 3),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)
library(xgboost)
set.seed(123)
xgbModel <- caret::train(x = train_upsampled[, feature_3], 
                         y = train_upsampled$Class,
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         metric = "Spec",
                         verbose = FALSE,
                         trControl = xgb_trcontrol)
print(xgbModel)

###########################################################################################################################
###########################################################################################################################

TestInfo = test[,c(feature_3, "Class")]
predict_xgbModel <- predict(xgbModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_xgbModel, reference = as.factor(TestInfo$Class))
performance_measures
conf_matrix <- confusionMatrix(predict_xgbModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)


#ROC
ROC <- predict(xgbModel, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(TestInfo$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for xgboost")



###########################################################################################################################
###########################################################################################################################

# J48 and Rcart training
features_formula <- as.formula(paste("Class ~", paste(feature_3, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)

# J48 Model Training with feature selection and corrected target variable using smote_data
model_J48_tuneLength <- train(features_formula, data = train_upsampled, method = "J48", trControl = train_control, tuneLength = 4)
print(model_J48_tuneLength)
plot(model_J48_tuneLength)
test_pred_J48_tuneLength <- predict(model_J48_tuneLength, newdata = test[, c(feature_3, "Class")])
confusionMatrix(test_pred_J48_tuneLength, test$Class)

# Training J48 with custom tuneGrid using smote_data
model_J48_tuneGrid <- train(features_formula, data = train_upsampled, method = "J48", trControl = train_control, tuneGrid = expand.grid(C = c(0.01, 0.25, 0.5), M = 1:4))
print(model_J48_tuneGrid)
plot(model_J48_tuneGrid)
test_pred_J48_tuneGrid <- predict(model_J48_tuneGrid, newdata = test[, c(feature_3, "Class")])
confusionMatrix(test_pred_J48_tuneGrid, test$Class)



###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_3, "Class")]
predict_J48 <- predict(model_J48_tuneGrid, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_J48, reference = as.factor(TestInfo$Class))
performance_measures
conf_matrix <- confusionMatrix(predict_J48, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)

library(pROC)
predict_J48_probs <- predict(model_J48_tuneGrid, newdata = TestInfo, type = "prob")
predict_J48_Y_probs <- predict_J48_probs[, "Y"]
roc_result <- roc(TestInfo$Class, predict_J48_Y_probs, levels=c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve")



###########################################################################################################################
###########################################################################################################################
## Logistic Regression
features_formula_logistic <- as.formula(paste("Class ~", paste(feature_3, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)
logisticModel <- train(features_formula_logistic, data = train_upsampled, method = "glm",
                       family = "binomial",
                       trControl = train_control,
                       preProcess = c("center", "scale"))
print(logisticModel)
test_pred_model_logisticModel<- predict(logisticModel, newdata = test[, c(feature_3, "Class")])
confusionMatrix(test_pred_model_logisticModel, test$Class)



###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_3, "Class")]
predict_logisticModel <- predict(logisticModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_logisticModel, reference = as.factor(TestInfo$Class))
performance_measures
conf_matrix <- confusionMatrix(predict_logisticModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



#ROC
ROC <- predict(logisticModel, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(TestInfo$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for LOGREGX")


###########################################################################################################################
###########################################################################################################################
## KNN
features_formula_knn <- as.formula(paste("Class ~", paste(feature_3, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)

# First KNN Model with tuneLength
knnModel <- train(features_formula_knn, data = train_upsampled, method = "knn",
                  trControl = train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 200)

knnModel
plot(knnModel)

# Predicting and evaluating performance
test_pred <- predict(knnModel, newdata = test)
confusionMatrix(test_pred, test$Class)

# Second KNN Model with custom tuneGrid
knnGrid <- expand.grid(k = seq(1, 100, 2))
knnModel <- train(features_formula_knn, data = train_upsampled, method = "knn",
                  trControl = train_control,
                  preProcess = c("center", "scale"),
                  tuneGrid = knnGrid)

knnModel
plot(knnModel)


###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_3, "Class")]
predict_knnModel <- predict(knnModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_knnModel, reference = as.factor(TestInfo$Class))
performance_measures

predict_knnModel_probs <- predict(knnModel, newdata = TestInfo, type = "prob")
predict_Y_probs <- predict_knnModel_probs[, "Y"]
roc_result <- roc(TestInfo$Class, predict_Y_probs, levels=c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve")

conf_matrix <- confusionMatrix(predict_knnModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)


###########################################################################################################################
###########################################################################################################################
## XGBOOST
xgb_trcontrol = trainControl(
  method = "cv", number = 10,
  summaryFunction = defaultSummary
)

xgbGrid <- expand.grid(
  nrounds = seq(from = 50, to = 300, by = 50),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2, 3),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)
library(xgboost)
set.seed(123)
xgbModel <- caret::train(x = train_upsampled[, feature_3], 
                         y = train_upsampled$Class,
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         metric = "Spec",
                         verbose = FALSE,
                         trControl = xgb_trcontrol)
print(xgbModel)

###########################################################################################################################
###########################################################################################################################

TestInfo = test[,c(feature_3, "Class")]
predict_xgbModel <- predict(xgbModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_xgbModel, reference = as.factor(TestInfo$Class))
performance_measures
conf_matrix <- confusionMatrix(predict_xgbModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)


#ROC
ROC <- predict(xgbModel, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(TestInfo$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for xgboost")

###########################################################################################################################
###########################################################################################################################
# Gradient Boosting
ctrl <- trainControl(method = "cv",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (2:10) * 10,
                       shrinkage = c(.001, .01),
                       n.minobsinnode = 3)

set.seed(31)
gbmFit <- caret::train(x = train_upsampled[, feature_3], 
                       y = train_upsampled$Class,
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)
print(gbmFit)

###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_3, "Class")]
predict_gbmFit <- predict(gbmFit, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_gbmFit, reference = as.factor(TestInfo$Class))
performance_measures
conf_matrix <- confusionMatrix(predict_gbmFit, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)

#ROC
predict_gbmFit_probs <- predict(gbmFit, newdata = TestInfo, type = "prob")
predict_Y_probs <- predict_gbmFit_probs[, "Y"]
roc_result <- roc(response = as.factor(TestInfo$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for GBDT")


###########################################################################################################################
###########################################################################################################################
# Random forest
features_formula <- as.formula(paste("Class ~", paste(feature_3, collapse = " + ")))
ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     summaryFunction = twoClassSummary, 
                     classProbs = TRUE, 
                     savePredictions = TRUE)

mtryValues <- seq(2, length(feature_3), by = 1)

set.seed(31)
rfFit <- caret::train(features_formula, 
                      data = train_upsampled, 
                      method = "rf",
                      tuneGrid = expand.grid(mtry = mtryValues),
                      importance = TRUE,
                      metric = "ROC",
                      trControl = ctrl,
                      ntree = 500)
print(rfFit)



###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_3, "Class")]
predict_rfFit <- predict(rfFit, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_rfFit, reference = as.factor(TestInfo$Class))
performance_measures

#ROC
ROC <- predict(rfFit, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(TestInfo$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for rANDOM FOREST")


conf_matrix <- confusionMatrix(predict_rfFit, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)
###########################################################################################################################
###########################################################################################################################














###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

# Applying Under and Over both 

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################


library(rsample)
library(ROSE)
set.seed(31)
# Sampling using ovun.sample
both_sample <- ovun.sample(Class ~ ., data = train, method = "both",
                           p = 0.5,
                           seed = 226,
                           N = 3500)$data

# Printing the distribution of both classes
table(both_sample$Class)

# Writing sampled data to CSV
write.csv(both_sample, "BOTH.csv", row.names = FALSE)





# Info Gain on BOTH sampling 
info_gain_result <- information.gain(Class ~ ., data = both_sample)
info_gain_df <- as.data.frame(info_gain_result)
info_gain_df$Feature = rownames(info_gain_df)
sorted_info_gain_df <- info_gain_df[order(-info_gain_df$attr_importance), ]
print(sorted_info_gain_df)
feature_names <- sorted_info_gain_df$Feature
print(feature_names)
feature_4 <- c("DECIDE", "GENHLTH", "PHYSHLTH", "X_PHYS14D", "X_RFHLTH", "EMPLOY1", 
               "HTIN4", "SLEPTIM1", "HAVARTH4", "X_DRDXAR2", "INCOME2", "X_INCOMG", 
               "RENTHOM1", "X_SMOKER3", "SMOKE100", "ASTHMA3", "X_ASTHMS1", "X_LTASTH1", 
               "CHCCOPD2", "X_TOTINDA", "MARITAL", "X_CASTHM1", "X_AIDTST4", "ACEPUNCH", 
               "SEXVAR", "X_SEX", "ACEDEPRS", "ACESWEAR", "DIABETE4")
feature_4 
###########################################################################################################################
###########################################################################################################################

# Feature _ 1 model training

###########################################################################################################################
###########################################################################################################################
# Gradient Boosting
set.seed(31)
ctrl <- trainControl(method = "cv",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (2:10) * 10,
                       shrinkage = c(.001, .01),
                       n.minobsinnode = 3)

set.seed(31)
gbmFit <- caret::train(x = both_sample[, feature_4], 
                       y = both_sample$Class,
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)
print(gbmFit)

###########################################################################################################################
###########################################################################################################################
tscfs = test[,c(feature_4, "Class")]
test_pred <- predict(gbmFit, newdata = tscfs)
test_pred = as.factor(test_pred)
levels(test_pred)
test_pred
performance_measures = confusionMatrix(test_pred, as.factor(tscfs$Class))
performance_measures


#ROC
ROC <- predict(gbmFit, newdata = tscfs, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for GBDT")


cm <- performance_measures$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################
# Random forest
features_formula <- as.formula(paste("Class ~", paste(feature_4, collapse = " + ")))
set.seed(31)
ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     summaryFunction = twoClassSummary, 
                     classProbs = TRUE, 
                     savePredictions = TRUE)

mtryValues <- seq(2, length(feature_4), by = 1)

set.seed(31)
rfFit <- caret::train(features_formula, 
                      data = both_sample, 
                      method = "rf",
                      tuneLength = 10,
                      importance = TRUE,
                      metric = "ROC",
                      trControl = ctrl,
                      ntree = 500)
print(rfFit)


###########################################################################################################################
###########################################################################################################################
tscfs = test[,c(feature_4, "Class")]
test_pred <- predict(rfFit, newdata = tscfs)
test_pred = as.factor(test_pred)
levels(test_pred)
test_pred
performance_measures = confusionMatrix(test_pred, as.factor(tscfs$Class))
performance_measures

#ROC
ROC <- predict(rfFit, newdata = tscfs, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for RF")


cm <- performance_measures$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################
## KNN
features_formula_knn <- as.formula(paste("Class ~", paste(feature_4, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)

# First KNN Model with tuneLength
knnModel <- train(features_formula_knn, data = both_sample, method = "knn",
                  trControl = train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 200)

knnModel
plot(knnModel)

# Predicting and evaluating performance
test_pred <- predict(knnModel, newdata = test)
confusionMatrix(test_pred, test$Class)

# Second KNN Model with custom tuneGrid
knnGrid2 <- expand.grid(k = seq(1, 100, 2))
knnModel2 <- train(features_formula_knn, data = both_sample, method = "knn",
                   trControl = train_control,
                   preProcess = c("center", "scale"),
                   tuneGrid = knnGrid2)

knnModel2
plot(knnModel2)


###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_4, "Class")]
predict_knnModel <- predict(knnModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_knnModel, reference = as.factor(TestInfo$Class))
performance_measures

#ROC
ROC <- predict(knnModel, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for KNN")


conf_matrix <- confusionMatrix(predict_knnModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################

# Model training using feature_5


###########################################################################################################################
###########################################################################################################################
cfs_subset <- cfs(Class ~ ., data = both_sample)
print(cfs_subset)
feature_5 <- c("ACEDEPRS", "ACEPUNCH", "ASTHMA3", "BLIND", "CHCCOPD2", "DECIDE", "EMPLOY1",
               "GENHLTH", "HAVARTH4", "HTIN4", "INCOME2", "MARITAL", "MEDCOST", "PHYSHLTH",
               "RENTHOM1", "SLEPTIM1", "X_AIDTST4", "X_PHYS14D", "X_RFHLTH", "X_SMOKER3")
feature_5


# Gradient Boosting
set.seed(31)
ctrl <- trainControl(method = "cv",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (2:10) * 10,
                       shrinkage = c(.001, .01),
                       n.minobsinnode = 3)

set.seed(31)
gbmFit <- caret::train(x = both_sample[, feature_5], 
                       y = both_sample$Class,
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)
print(gbmFit)

###########################################################################################################################
###########################################################################################################################
tscfs = test[,c(feature_5, "Class")]
test_pred <- predict(gbmFit, newdata = tscfs)
test_pred = as.factor(test_pred)
levels(test_pred)
test_pred
performance_measures = confusionMatrix(test_pred, as.factor(tscfs$Class))
performance_measures


#ROC
ROC <- predict(gbmFit, newdata = tscfs, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for GBDT")


cm <- performance_measures$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################
# Random forest
features_formula <- as.formula(paste("Class ~", paste(feature_5, collapse = " + ")))
set.seed(31)
ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     summaryFunction = twoClassSummary, 
                     classProbs = TRUE, 
                     savePredictions = TRUE)

mtryValues <- seq(2, length(feature_5), by = 1)

set.seed(31)
rfFit <- caret::train(features_formula, 
                      data = both_sample, 
                      method = "rf",
                      tuneLength = 10,
                      importance = TRUE,
                      metric = "ROC",
                      trControl = ctrl,
                      ntree = 500)
print(rfFit)


###########################################################################################################################
###########################################################################################################################
tscfs = test[,c(feature_5, "Class")]
test_pred <- predict(rfFit, newdata = tscfs)
test_pred = as.factor(test_pred)
levels(test_pred)
test_pred
performance_measures = confusionMatrix(test_pred, as.factor(tscfs$Class))
performance_measures

#ROC
ROC <- predict(rfFit, newdata = tscfs, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for RF")


cm <- performance_measures$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################
## KNN
features_formula_knn <- as.formula(paste("Class ~", paste(feature_5, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)

# First KNN Model with tuneLength
knnModel <- train(features_formula_knn, data = both_sample, method = "knn",
                  trControl = train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 200)

knnModel
plot(knnModel)

# Predicting and evaluating performance
test_pred <- predict(knnModel, newdata = test)
confusionMatrix(test_pred, test$Class)

# Second KNN Model with custom tuneGrid
knnGrid2 <- expand.grid(k = seq(1, 100, 2))
knnModel2 <- train(features_formula_knn, data = both_sample, method = "knn",
                   trControl = train_control,
                   preProcess = c("center", "scale"),
                   tuneGrid = knnGrid2)

knnModel2
plot(knnModel2)


###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_5, "Class")]
predict_knnModel <- predict(knnModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_knnModel, reference = as.factor(TestInfo$Class))
performance_measures

#ROC
ROC <- predict(knnModel, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for KNN")


conf_matrix <- confusionMatrix(predict_knnModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################
library(randomForest) 
library(caret)

control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
rfe_results <- rfe(both_sample[, -which(names(both_sample) == "Class")], 
                   both_sample$Class, 
                   sizes = c(1:20), 
                   rfeControl = control)
print(rfe_results)
top_features <- predictors(rfe_results)
print(top_features)
feature_6 <- c("DECIDE", "ACETTHEM", "HIVRISK5", "X_RFHLTH", "X_CASTHM1", "CHCKDNY2", "PHYSHLTH",
               "CHCCOPD2", "BLIND", "ACEHVSEX", "ACEPRISN", "X_ASTHMS1", "ACETOUCH", "MEDCOST",
               "X_LTASTH1", "ACEDEPRS", "GENHLTH", "X_PHYS14D", "DIABETE4", "ASTHMA3", "SLEPTIM1",
               "CVDSTRK3", "ACEDRINK", "X_SEX", "ACEDIVRC", "ACEDRUGS", "X_DRDXAR2", "CVDINFR4",
               "X_MICHD", "X_SMOKER3", "INCOME2")
feature_6

##########################################################################################################################
##########################################################################################################################
# Gradient Boosting
set.seed(31)
ctrl <- trainControl(method = "cv",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (2:10) * 10,
                       shrinkage = c(.001, .01),
                       n.minobsinnode = 3)

set.seed(31)
gbmFit <- caret::train(x = both_sample[, feature_6], 
                       y = both_sample$Class,
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)
print(gbmFit)

###########################################################################################################################
###########################################################################################################################
tscfs = test[,c(feature_6, "Class")]
test_pred <- predict(gbmFit, newdata = tscfs)
test_pred = as.factor(test_pred)
levels(test_pred)
test_pred
performance_measures = confusionMatrix(test_pred, as.factor(tscfs$Class))
performance_measures


#ROC
ROC <- predict(gbmFit, newdata = tscfs, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for GBDT")


cm <- performance_measures$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################
# Random forest
features_formula <- as.formula(paste("Class ~", paste(feature_6, collapse = " + ")))
set.seed(31)
ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     summaryFunction = twoClassSummary, 
                     classProbs = TRUE, 
                     savePredictions = TRUE)

mtryValues <- seq(2, length(feature_6), by = 1)

set.seed(31)
rfFit <- caret::train(features_formula, 
                      data = both_sample, 
                      method = "rf",
                      tuneLength = 10,
                      importance = TRUE,
                      metric = "ROC",
                      trControl = ctrl,
                      ntree = 500)
print(rfFit)


###########################################################################################################################
###########################################################################################################################
tscfs = test[,c(feature_6, "Class")]
test_pred <- predict(rfFit, newdata = tscfs)
test_pred = as.factor(test_pred)
levels(test_pred)
test_pred
performance_measures = confusionMatrix(test_pred, as.factor(tscfs$Class))
performance_measures

#ROC
ROC <- predict(rfFit, newdata = tscfs, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for RF")


cm <- performance_measures$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################
## KNN
features_formula_knn <- as.formula(paste("Class ~", paste(feature_6, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)

# First KNN Model with tuneLength
knnModel <- train(features_formula_knn, data = both_sample, method = "knn",
                  trControl = train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 200)

knnModel
plot(knnModel)

# Predicting and evaluating performance
test_pred <- predict(knnModel, newdata = test)
confusionMatrix(test_pred, test$Class)

# Second KNN Model with custom tuneGrid
knnGrid2 <- expand.grid(k = seq(1, 100, 2))
knnModel2 <- train(features_formula_knn, data = both_sample, method = "knn",
                   trControl = train_control,
                   preProcess = c("center", "scale"),
                   tuneGrid = knnGrid2)

knnModel2
plot(knnModel2)


###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_6, "Class")]
predict_knnModel <- predict(knnModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_knnModel, reference = as.factor(TestInfo$Class))
performance_measures

#ROC
ROC <- predict(knnModel, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for KNN")


conf_matrix <- confusionMatrix(predict_knnModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



############################################################################################################################
library(Boruta)
set.seed(123)
boruta_output <- Boruta(Class ~ ., data = both_sample, doTrace = 2, maxRuns = 100)
print(boruta_output)
final_features <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(final_features)
feature_7 <- c("ACEDEPRS", "ACEDIVRC", "ACEDRINK", "ACEDRUGS", "ACEHURT1", "ACEHVSEX",
            "ACEPRISN", "ACEPUNCH", "ACESWEAR", "ACETOUCH", "ACETTHEM", "ASTHMA3",
            "BLIND", "CHCCOPD2", "CHCKDNY2", "CHCOCNCR", "CHCSCNCR", "CHECKUP1",
            "CVDINFR4", "CVDSTRK3", "DEAF", "DECIDE", "DIABETE4", "DRNKANY5",
            "EDUCA", "EMPLOY1", "FLUSHOT7", "GENHLTH")
feature_7

##########################################################################################################################
##########################################################################################################################
# Gradient Boosting
set.seed(31)
ctrl <- trainControl(method = "cv",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (2:10) * 10,
                       shrinkage = c(.001, .01),
                       n.minobsinnode = 3)

set.seed(31)
gbmFit <- caret::train(x = both_sample[, feature_7], 
                       y = both_sample$Class,
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)
print(gbmFit)

###########################################################################################################################
###########################################################################################################################
tscfs = test[,c(feature_7, "Class")]
test_pred <- predict(gbmFit, newdata = tscfs)
test_pred = as.factor(test_pred)
levels(test_pred)
test_pred
performance_measures = confusionMatrix(test_pred, as.factor(tscfs$Class))
performance_measures


#ROC
ROC <- predict(gbmFit, newdata = tscfs, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for GBDT")


cm <- performance_measures$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################
# Random forest
features_formula <- as.formula(paste("Class ~", paste(feature_7, collapse = " + ")))
set.seed(31)
ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     summaryFunction = twoClassSummary, 
                     classProbs = TRUE, 
                     savePredictions = TRUE)

mtryValues <- seq(2, length(feature_7), by = 1)

set.seed(31)
rfFit <- caret::train(features_formula, 
                      data = both_sample, 
                      method = "rf",
                      tuneLength = 10,
                      importance = TRUE,
                      metric = "ROC",
                      trControl = ctrl,
                      ntree = 500)
print(rfFit)


###########################################################################################################################
###########################################################################################################################
tscfs = test[,c(feature_7, "Class")]
test_pred <- predict(rfFit, newdata = tscfs)
test_pred = as.factor(test_pred)
levels(test_pred)
test_pred
performance_measures = confusionMatrix(test_pred, as.factor(tscfs$Class))
performance_measures

#ROC
ROC <- predict(rfFit, newdata = tscfs, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for RF")


cm <- performance_measures$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)



###########################################################################################################################
###########################################################################################################################
## KNN
features_formula_knn <- as.formula(paste("Class ~", paste(feature_6, collapse = " + ")))
set.seed(31)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)

# First KNN Model with tuneLength
knnModel <- train(features_formula_knn, data = both_sample, method = "knn",
                  trControl = train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 200)

knnModel
plot(knnModel)

# Predicting and evaluating performance
test_pred <- predict(knnModel, newdata = test)
confusionMatrix(test_pred, test$Class)

# Second KNN Model with custom tuneGrid
knnGrid2 <- expand.grid(k = seq(1, 100, 2))
knnModel2 <- train(features_formula_knn, data = both_sample, method = "knn",
                   trControl = train_control,
                   preProcess = c("center", "scale"),
                   tuneGrid = knnGrid2)

knnModel2
plot(knnModel2)


###########################################################################################################################
###########################################################################################################################
TestInfo = test[,c(feature_6, "Class")]
predict_knnModel <- predict(knnModel, newdata = TestInfo, type = "raw")
performance_measures  <- confusionMatrix(data = predict_knnModel, reference = as.factor(TestInfo$Class))
performance_measures

#ROC
ROC <- predict(knnModel, newdata = TestInfo, type = "prob")
predict_Y_probs <- ROC[, "Y"]
roc_result <- roc(response = as.factor(tscfs$Class), predictor = predict_Y_probs, levels = c("N", "Y"))
cat("Area Under the ROC Curve (AUC):", auc(roc_result), "\n")
plot(roc_result, main="ROC Curve for KNN")


conf_matrix <- confusionMatrix(predict_knnModel, test$Class)
cm <- conf_matrix$table
tp <- cm["Y", "Y"]
fp <- cm["N", "Y"]
tn <- cm["N", "N"]
fn <- cm["Y", "N"]
performance_measures_Y = calculate_measures(tp, fp, tn, fn)
cat("Performance Measures for Class 'Y':\n")
print(performance_measures_Y)

performance_measures_N = calculate_measures(tn, fn, tp, fp)
cat("Performance Measures for Class 'N':\n")
print(performance_measures_N)
