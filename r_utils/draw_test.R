library(data.table)
library(ggplot2)
library(stringr)
library(pROC)
library(parallel)
library(scales)
library(ROCR)

f_names <- list.files(path = ".", 
                      pattern = "testing",
                      full.names = T)
out_dir <- "plots/"
if (dir.exists(out_dir) ){
  print('nothing')
} else {
  dir.create(out_dir)
}

result <- data.table(model_name = character(),
                     thres= numeric(0), 
                     accuracy= numeric(0), 
                     fpr = numeric(0),
                     fnr = numeric(0),
                     recall = numeric(0),
                     precision = numeric(0),
                     f1sc = numeric(0)
                     #auc = numeric(0)
                     )

optimistic = FALSE
for (fs in f_names){
  main_name = basename(fs)
  main_name = gsub('testing', 'resnet', fs)
  main_name = gsub('.csv', '', main_name)
  main_name = gsub('stage', 'block', main_name)
  
  pd_table <- fread(fs, header = T, sep = ',')
  pd_table[, y_pred := as.double(y_pred)]
  
  #score.auc <- auc(pd_table$y_true, pd_table$y_pred)
  #main_name  <- paste0(main_name, '_auc:', round(score.auc, 3))
  
  thres_all = unique(sort(pd_table[y_true == 1,]$y_pred))
  thres_all <- thres_all[thres_all <= 0.99]
  if (optimistic) {
    print('OPTIMISTIC!')
  } else {
    print('BAD!')
    thres_all <- c(thres_all[1], thres_all[2:length(thres_all)]+0.0000001)
  }
  
  #
  
  for (thres in thres_all)  {
    pd_table[, adjust_ := ifelse(y_pred >= thres, 1, 0)]
    conf <- table(pd_table[, .(y_true, y_p = pd_table$adjust_ )]) # tmp$V1 should change (iter over epoch)
    
    if (length(conf) == 2) {
      conf[3] = 0
      conf[4] = 0
    }
    
    acc = sum(diag(conf)) / sum(conf)
    fpr = conf[1,2] / sum(conf[1,])
    fnr = conf[2,1] / sum(conf[2,])
    
    recall = conf[2,2] / sum(conf[2, ])
    precision = conf[2,2] / sum(conf[, 2])
    f1sc = 2 * recall * precision / (recall + precision)
    
    tmp_out <- data.table(model_name = main_name, 
                          thres= thres, 
                          accuracy= acc, 
                          fpr = fpr,
                          fnr = fnr,
                          recall = recall,
                          precision = precision,
                          f1sc = f1sc
                          #auc = round(score.auc, 5)
                          )
    
    result <- rbind(result, tmp_out)
    print(paste0('judge level: ', thres))
  }
}


  tt <- result
  tt[, model_name := basename(model_name)]
  tt[, model_name := gsub(pattern = "block", replacement = "stage", x = model_name)]
  tt[, flag := abs(fpr - 0.03) ]
  
  y <- tt[, .(flag = min(flag)), .(model_name)]
  y <- tt[model_name %in% y$model_name & flag %in% y$flag,][!duplicated(model_name),]
  y <- y[order(fnr, decreasing = F),]
  tt$model_name <- factor(tt$model_name, levels = unique(y$model_name))
  
  ggplot(tt, aes(fnr, fpr, color = model_name)) + 
    geom_line(lty = 2) +
    geom_point(size = 1, alpha = 0.25) +
    # geom_text(data = tt[thres == max(thres) | thres == min(thres), ], 
    #           aes(label = paste0(thres, '\n',
    #                              sprintf("%.3f", round(fnr, 4) * 100), "%")), color = 'tomato3') + 
    coord_cartesian(xlim = c(0,0.05),ylim = c(0, 0.05)) +
    theme_bw() + 
    scale_x_continuous(breaks = seq(0, 0.3, 0.01), labels = percent) +
    scale_y_continuous(breaks = seq(0, 0.1, 0.002), labels = percent) +
    ggtitle("fpr-fnr curve")+ 
    theme(plot.title =  element_text(hjust = 0.5)) 
    #stat_smooth()
  #scale_x_log10() + scale_y_log10()
  
  ggsave(paste0(out_dir, 'fnr_fpr_aggregate.png'), 
         width = 24, height = 17, units = 'cm', dpi = 500)
  
  ggplot(tt, aes(fpr, fnr, color = model_name)) + 
    geom_line(lty = 2) +
    geom_point(size = 1, alpha = 0.25) +
    # geom_text(data = tt[thres == max(thres) | thres == min(thres), ], 
    #           aes(label = paste0(thres, '\n',
    #                              sprintf("%.3f", round(fnr, 4) * 100), "%")), color = 'tomato3') + 
    coord_cartesian(xlim = c(0,0.15),ylim = c(0, 0.05)) +
    theme_bw() + 
    scale_x_continuous(breaks = seq(0, 0.3, 0.01), labels = percent) +
    scale_y_continuous(breaks = seq(0, 0.1, 0.002), labels = percent) +
    ggtitle("fpr-fnr curve")+ 
    theme(plot.title =  element_text(hjust = 0.5)) 
  #stat_smooth()
  #scale_x_log10() + scale_y_log10()
  
ggsave(paste0(out_dir, 'fpr_fnr_aggregate.png'), 
         width = 24, height = 17, units = 'cm', dpi = 500)

ggplot(tt, aes(fpr, fnr)) + 
  geom_line(lty = 2, alpha = 0.1) +
  geom_point(size = 1, alpha = 0.1) +
  coord_cartesian(xlim = c(0,0.15),ylim = c(0, 0.05)) +
  theme_bw() + 
  scale_x_continuous(breaks = seq(0, 0.3, 0.01), labels = percent) +
  scale_y_continuous(breaks = seq(0, 0.1, 0.002), labels = percent) +
  ggtitle("fpr-fnr curve")+ 
  theme(plot.title =  element_text(hjust = 0.5)) +
  stat_smooth()

# generate tables
# fpr base
get_table <- tt[, .(fnr_under_fp_.03 = approx(x = fpr, y = fnr, xout = 0.03)$y,
                    fnr_under_fp_.01 = approx(x = fpr, y = fnr, xout = 0.01)$y,
                    fnr_under_fp_.005 = approx(x = fpr, y = fnr, xout = 0.005)$y,
                    fnr_under_fp_.001 = approx(x = fpr, y = fnr, xout = 0.001)$y,
                    thres_under_fp_.03 = approx(x = fpr, y = thres, xout = 0.03)$y,
                    thres_under_fp_.01 = approx(x = fpr, y = thres, xout = 0.01)$y,
                    thres_under_fp_.005 = approx(x = fpr, y = thres, xout = 0.005)$y,
                    thres_under_fp_.001 = approx(x = fpr, y = thres, xout = 0.001)$y
), .(model_name)]

# fnr base
get_table <- tt[, .(fpr_under_fn_.03 = approx(x = fnr, y = fpr, xout = 0.03)$y,
                    fpr_under_fn_.01 = approx(x = fnr, y = fpr, xout = 0.01)$y,
                    fpr_under_fn_.005 = approx(x = fnr, y = fpr, xout = 0.005)$y,
                    fpr_under_fn_.001 = approx(x = fnr, y = fpr, xout = 0.001)$y,
                    fpr_under_fn_.0005 = approx(x = fnr, y = fpr, xout = 0.0005)$y,
                    fpr_under_fn_.0001 = approx(x = fnr, y = fpr, xout = 0.0001)$y,
                    fpr_under_fn_.0 = approx(x = fnr, y = fpr, xout = 0)$y,
                    thres_under_fn_.03 = approx(x = fnr, y = thres, xout = 0.03)$y,
                    thres_under_fn_.01 = approx(x = fnr, y = thres, xout = 0.01)$y,
                    thres_under_fn_.005 = approx(x = fnr, y = thres, xout = 0.005)$y,
                    thres_under_fn_.001 = approx(x = fnr, y = thres, xout = 0.001)$y,
                    thres_under_fn_.0005 = approx(x = fnr, y = thres, xout = 0.0005)$y,
                    thres_under_fn_.0001 = approx(x = fnr, y = thres, xout = 0.0001)$y,
                    thres_under_fn_.0 = approx(x = fnr, y = thres, xout = 0)$y
), .(model_name)]
get_table_out <- get_table[, .(model_name,
                               fpr_under_fn_.03 = paste0( sprintf("%.2f", round(fpr_under_fn_.03 * 100,3)), "%" ),
                               thres_under_fn_.03 = round(thres_under_fn_.03, 3),
                               fpr_under_fn_.01 = paste0( sprintf("%.2f", round(fpr_under_fn_.01 * 100,3)), "%" ),
                               thres_under_fn_.01 = round(thres_under_fn_.01, 3),
                               fpr_under_fn_.005 = paste0( sprintf("%.2f", round(fpr_under_fn_.005 * 100,3)), "%" ),
                               thres_under_fn_.005 = round(thres_under_fn_.005, 3),
                               fpr_under_fn_.001 = paste0( sprintf("%.2f", round(fpr_under_fn_.001 * 100,3)), "%" ),
                               thres_under_fn_.001 = round(thres_under_fn_.001, 3),
                               fpr_under_fn_.0005 = paste0( sprintf("%.2f", round(fpr_under_fn_.0005 * 100,3)), "%" ),
                               thres_under_fn_.0005 = round(thres_under_fn_.0005, 3),
                               fpr_under_fn_.0001 = paste0( sprintf("%.2f", round(fpr_under_fn_.0001 * 100,3)), "%" ),
                               thres_under_fn_.0001 = round(thres_under_fn_.0001, 3),
                               fpr_under_fn_.0 = paste0( sprintf("%.2f", round(fpr_under_fn_.0 * 100,3)), "%" ),
                               thres_under_fn_.0 = round(thres_under_fn_.0, 3)
                               )]
write.csv(get_table, file = "result_table_fnr_base.csv")
write.csv(get_table_out, file = "result_table_fnr_base_pretty.csv")
  
##############
# plot scatter plot (hitter for )
all_pred <- lapply(f_names, function(x) {
  y <- fread(x)
  y$model_name = basename(x)
  colnames(y) <- c("png_name", "y_pred", "y_true", "model_name")
  return(y)
  })
all_pred <- do.call(rbind, all_pred)  

all_pred[, model_name := gsub(pattern = "testing_|\\.csv", replacement = "", x = model_name)]
all_pred[, model_name := gsub(pattern = "block", replacement = "stage", x = model_name)]

ggplot(all_pred, aes(x = factor(y_true), y = y_pred)) +
  geom_point(alpha = 0.25, position = 'jitter') + 
  theme_bw() + 
  facet_wrap(~model_name) +
  theme(strip.background = element_rect(fill = 'papayawhip')) +
  labs(x = 'truth', y = 'probability of predicting as copper defect') +
  ggtitle('defect prediction probability')

ggsave(paste0(out_dir, 'prediction_scatters_aggregate.png'), 
       width = 24, height = 17, units = 'cm', dpi = 500)

### ROC curve plot
tt_all <- tt[, .(fpr = mean(fpr, na.rm = T)), .(tpr = recall)]
ggplot(tt_all, aes(fpr, tpr)) + geom_point()
saveRDS(tt_all, file = 'roc_curve_table.rds')


# aggregate many data
r_data1 <- readRDS("resnet/roc_curve_table.rds")
r_data2 <- readRDS("self/roc_curve_table.rds")

r_data1$version <- '-restnet_mean'
r_data2$version <- '-dataset_mean'

roc_data <- rbind(r_data1, r_data2)
ggplot(roc_data, aes(fpr, tpr, color = version)) +
  geom_point(size = 1) +
  geom_line(lty = 2) +
  theme_bw() +
  coord_cartesian(ylim = c(0.95, 1)) +
  scale_x_continuous(breaks = seq(0,1,0.005), label = percent) +
  scale_y_continuous(breaks = seq(0,1,0.01), label = percent) +
  labs(x = 'False positive rate', y = 'True positive rate') +
  ggtitle("ROC curve") +
  theme(plot.title = element_text(hjust = 0.5))
ggsave("ROC_cuve_plot.png", dpi = 500, width = 24, height = 17, units = "cm")  
  


############# 

tt[, `:=`(model_name = sapply(model_name, function(x) str_extract(string = x, pattern = ".*_k"))) ]
x_range <- seq(0,0.05, 0.0005)
x_interp <- data.table()
for (ix in x_range) {
  x_tmp <- tt[, .(fnr = ix, 
                  fpr = approx(x = fnr, y = fpr, xout = ix)$y ), .(model_name)]
  x_interp <- rbind(x_interp, x_tmp)
}


x_interp_avg <- x_interp[, .(fpr.avg = mean(fpr, na.rm = T),
                             fpr.se = sd(fpr, na.rm = T)/sqrt(.N)), .(fnr, model_name)]


ggplot(x_interp_avg, aes(x = fnr, y = fpr.avg, color = model_name)) +
  geom_point(size = 1) + 
  geom_line(lty = 2) +
  geom_errorbar(aes(ymin = fpr.avg - 1.96 * fpr.se,
                    ymax = fpr.avg + 1.96 * fpr.se), width = 0.00025) +
  theme_bw() +
  scale_x_continuous(breaks = seq(0,0.01, 0.001), labels = percent) +
  scale_y_continuous(breaks = seq(0, 0.2, 0.005), labels = percent) +
  coord_cartesian(xlim = c(0, 0.01)) + 
  labs(x = "false negative rate", y = "average false positive rate (with 1.96 se)") +
  ggtitle("Average false-positve rate under fixed false-negative rate")
  
ggsave(filename = paste0(out_dir, "/estimate_avgfpr_fixfnr.png"), dpi = 500,
       width = 24, height = 17, units = "cm")


##########################
# read in all data (for k fold use)
print(f_names)
all_pred <- all_pred[!duplicated(png_name), ]

thres_all <- unique(round(all_pred$y_pred, 3))
res_all <- data.table(model_name = character(),
                     thres= numeric(0), 
                     accuracy= numeric(0), 
                     fpr = numeric(0),
                     fnr = numeric(0),
                     recall = numeric(0),
                     precision = numeric(0),
                     f1sc = numeric(0)
                     #auc = numeric(0)
)


for (thres in thres_all)  {
  
  all_pred[, adjust_ := ifelse(y_pred > thres, 1, 0)]
  conf <- table(all_pred$y_true, all_pred$adjust_ ) # tmp$V1 should change (iter over epoch)
  
  if (length(conf) == 2) {
    next
  }
  
  acc = sum(diag(conf)) / sum(conf)
  fpr = conf[1,2] / sum(conf[1,])
  fnr = conf[2,1] / sum(conf[2,])
  
  recall = conf[2,2] / sum(conf[2, ])
  precision = conf[2,2] / sum(conf[, 2])
  f1sc = 2 * recall * precision / (recall + precision)
  
  tmp_out <- data.table(model_name = main_name, 
                        thres= thres, 
                        accuracy= acc, 
                        fpr = fpr,
                        fnr = fnr,
                        recall = recall,
                        precision = precision,
                        f1sc = f1sc
                        #auc = round(score.auc, 5)
  )
  
  res_all <- rbind(res_all, tmp_out)
  print(paste0('judge level: ', thres))
}
ggplot(res_all, aes(fpr, fnr)) +
  geom_point(size = 1, alpha = 0.2) + 
  geom_line(lty = 2) + 
  theme_bw() +
  scale_x_continuous(breaks = seq(0, 1, 0.01), labels = percent) +
  scale_y_continuous(breaks = seq(0, 1, 0.005), labels = percent) +
  coord_cartesian(xlim = c(0, 0.04), ylim = c(0, 0.1)) +
  labs(x = 'false-positive rate', y = 'false-negative rate') +
  ggtitle('fpr~fnr, all data point')
ggsave(filename = paste0(out_dir, '/all_data_fpr_fnr.png'),
       dpi = 500, width = 24, height = 17, units = "cm")
saveRDS(res_all, file = 'all_data_thres.rds')

# catch fp / fns
fp_cri <- 0.99
fn_cri <- 0.10
fps <- all_pred[y_pred > fp_cri & y_true == 0, ]
fns <- all_pred[y_pred < fn_cri & y_true == 1, ]

write.csv(fps, file = paste0('fps_catch_', fp_cri, '.csv'))
write.csv(fns, file = paste0('fns_catch_', fn_cri, '.csv'))



## plot merged version
# tmp use
ori1 <- readRDS("self/dataset_mean.rds")
ori2 <- readRDS("resnet/imagenet_mean.rds")
#rev2 <- readRDS("../0713/fpr_fnr_review2.rds")
#rev2 <- readRDS("../rev_by_vote.rds")
ori1$version <- "-dataset_mean"
ori2$version <- "-imagenet_mean"
#rev2$version <- "review-by-vote"
x_interp_avg <- rbind(ori1, ori2)



ggplot(x_interp_avg, aes(x = fnr, y = fpr.avg, color = version)) +
  geom_point(size = 1) + 
  geom_line(lty = 2) +
  geom_errorbar(aes(ymin = fpr.avg - 1.96 * fpr.se,
                    ymax = fpr.avg + 1.96 * fpr.se), width = 0.00025) +
  theme_bw() +
  scale_x_continuous(breaks = seq(0,0.01, 0.001), labels = percent) +
  scale_y_continuous(breaks = seq(0, 0.2, 0.005), labels = percent) +
  coord_cartesian(xlim = c(0, 0.01)) + 
  labs(x = "false negative rate", y = "average false positive rate (with 1.96 se)") +
  ggtitle("Average false-positve rate under fixed false-negative rate") +
  scale_color_manual(values = c("tomato3", "royalblue2")) 

ggsave(filename = paste0(out_dir, "/estimate_avgfpr_fixfnr_dataset_imagenet_mean.png"), dpi = 500,
       width = 24, height = 17, units = "cm")
