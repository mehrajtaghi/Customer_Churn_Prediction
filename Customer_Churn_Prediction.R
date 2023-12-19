library(tidyverse)
library(skimr)
library(inspectdf)
library(caret)
library(glue)
library(highcharter)
library(h2o)
library(scorecard)
library(data.table)
library(ggsci)
library(yardstick)

df <- fread('/Users/mehrac/Downloads/bank-full (1).csv', sep = ';') 
df %>%view()

view(df)

glimpse (df)

df$y %>% factor (levels=c('no', 'yes'), labels=c(0,1)) -> 
  df$y 

lapply (df,class) 
class(df$age)
prop.table(table(df$y))
skimr::skim(df)
view(df)

#az olan target make as 1

# 26 - 72 empty
# Excluding unimportant variables

ivars <- df %>% 
  iv(y='y') %>%
  as_tibble() %>%
  mutate(info_value=round(info_value, 3)) %>% 
  arrange (desc(info_value))

ivars <- ivars %>% filter(info_value>0.02)
ivars <-  ivars[[1]]

df <- df %>% select(y, ivars)
# 80 - 86 empty

df_list <- df %>% split_df('y', ratio = 0.8, seed = 123)

 # Applying binning according to weight of Evidence principle
#df %% woebin ('y')->bins

df %>% filter(job=='student')

df %>% woebin ('y')->bins 
bins$job %>% woebin_plot()
bins$pdays %>% as_tibble() %>% view()
bins$duration %>% woebin_plot()

train_woe <- df_list$train %>% woebin_ply(bins)
test_woe <- df_list$test %>% woebin_ply(bins)

test_woe %>% view()

names<-names (train_woe)
names<- gsub(' _woe', '',names)

names(train_woe)<-names 
names(test_woe)<-names


#Standardize features
#we have use woe method, so there is no need to standardize

# Finding multicollinearity by applying VIF

target<-'y'
features<-train_woe %>% select(-y) %>% names()

f <- as.formula(paste(target, paste(features, collapse = ' + '), sep = ' ~ '))
glm <- glm(f, data = train_woe, family = 'binomial')

summary(glm)

coef_na <- attributes(alias(glm)$complete)$dimnames[[1]]
features <- features[!features%in%coef_na]

f <- as.formula(paste(target, paste(features, collapse = ' + '), sep = ' ~ '))
glm <- glm(f, data = train_woe, family = 'binomial')


while(glm %>% vif() %>% arrange(desc(gvif)) %>% .[1,2] >= 3){
  afterVIF <- glm %>% vif() %>% arrange(desc(gvif)) %>% .[-1,"variable"]
  afterVIF <- afterVIF$variable
  f <- as.formula(paste(target, paste(afterVIF, collapse = " + "), sep = " ~ "))
  glm <- glm(f, data = train_woe, family = "binomial")
}

features <- glm %>% vif() %>% arrange(desc(gvif)) %>% pull(variable) 
features

h2o.init()

train_h2o <- train_woe %>% select(target,features) %>% as.h2o()
test_h2o <- test_woe %>% select(target,features) %>% as.h2o()

model <- h2o.glm(
  x = features, y = target, family = "binomial", 
  training_frame = train_h2o, validation_frame = test_h2o,
  nfolds = 10, seed = 123, remove_collinear_columns = T,
  balance_classes = T, lambda = 0, compute_p_values = T)


while(model@model$coefficients_table %>%
      as.data.frame() %>%
      select(names,p_value) %>%
      mutate(p_value = round(p_value,3)) %>%
      .[-1,] %>%
      arrange(desc(p_value)) %>%
      .[1,2] >= 0.05){
  model@model$coefficients_table %>%
    as.data.frame() %>%
    select(names,p_value) %>%
    mutate(p_value = round(p_value,3)) %>%
    filter(!is.nan(p_value)) %>%
    .[-1,] %>%
    arrange(desc(p_value)) %>%
    .[1,1] -> v
  features <- features[features!=v]
  
  train_h2o <- train_woe %>% select(target,features) %>% as.h2o()
  test_h2o <- test_woe %>% select(target,features) %>% as.h2o()
  
  
  model <- h2o.glm(
    x = features, y = target, family = "binomial", 
    training_frame = train_h2o, validation_frame = test_h2o,
    nfolds = 10, seed = 123, remove_collinear_columns = T,
    balance_classes = T, lambda = 0, compute_p_values = T)
}

model@model$coefficients_table %>%
  as.data.frame() %>%
  select(names,p_value) %>%
  mutate(p_value = round(p_value,3))


model@model$coefficients %>%
  as.data.frame() %>%
  mutate(names = rownames(model@model$coefficients %>% as.data.frame())) %>%
  `colnames<-`(c('coefficients','names')) %>%
  select(names,coefficients)

h2o.varimp(model) %>% as.data.frame() %>%
  select(variable, percentage) %>%
  hchart("pie", hcaes(x = variable, y = percentage)) %>%
  hc_colors(colors = 'orange') %>%
  hc_xAxis(visible=T) %>%
  hc_yAxis(visible=T)

pred <- model %>% h2o.predict(newdata = test_h2o) %>% 
  as.data.frame() %>% select(p1,predict)

model %>% h2o.performance(newdata = test_h2o) %>%
  h2o.find_threshold_by_max_metric('f1')

eva <- perf_eva(
  pred = pred %>% pull(p1),
  label = df_list$test$y %>% as.character() %>% as.numeric(),
  binomial_metric = c("auc","gini"),
  show_plot = "roc")


# Check overfitting ----
model %>%
  h2o.auc(train = T,
          valid = T,
          xval = T) %>%
  as_tibble() %>%
  round(2) %>%
  mutate(data = c('train','test','cross_val')) %>%
  mutate(gini = 2*value-1) %>%
  select(data,auc=value,gini)


model %>% h2o.performance(test_h2o) %>% 
  h2o.find_threshold_by_max_metric('f1') -> threshold

#confusion matrix

cm <- pred %>%
  bind_cols (test_h2o %>% as.data.frame () %>% select (y)) %>% 
  conf_mat(y, predict) %>% pluck (1) %>% as_tibble() %>% 
  mutate (true_pred = ifelse (Prediction == Truth, 1,0))

cm$n [cm$true_pred == 1] %>% sum() / cm$n %>% sum() -> accuracy
  

cm %>%
  as_tibble %>%
  ggplot (aes(x=Prediction, y=Truth, fill=n)) + 
  geom_tile(show.legend=F, alpha=0.5) +
  geom_text(aes(label=n), color="black", alpha-1, size=8) +
  labs (
    title = glue('Accuracy = {round (enexpr(accuracy), 2)}%') +
      scale_fill_gsea()
    
    
model %>% 
  h2o.performance(test_h2o) %>% h2o.precision()

model %>% 
  h2o.confusionMatrix(test_h2o) %>% 
  as_tibble() %>% 
  select("0","1") %>%
  .[1:2,] %>% t() %>%
  fourfoldplot (conf.level = 0, color = c("red", "darkgreen"),
                                  main = paste ("Accuracy =",
                                                round (sum(diag (.))/sum(.)*100,1) , "%"))
