library(plyr)

library(tidyverse)
library(latex2exp)
library(viridis)
fsep = .Platform$file.sep

chi_elev_levels <- c('chi_elev_1', 'chi_elev_2', 'chi_elev_3', 'chi_elev_4', 'chi_elev_5',
                'chi_elev_6', 'chi_elev_7', 'chi_elev_8', 'chi_elev_9', 'chi_elev_10',
                'chi_elev_11', 'chi_elev_12', 'chi_elev_13', 'chi_elev_14', 'chi_elev_15',
                'chi_elev_16', 'chi_elev_17', 'chi_elev_18', 'chi_elev_19', 'chi_elev_20')
colors = c('#000000',
           '#ffff33',  # yellow
           '#fb9a99', # pink
           '#e31a1c', # red
           '#fdbf6f', # orange
           '#ff7f00', # dk orange
           '#e7298a', # fuscia
           '#a6cee3', # lt blu
           '#1f78b4', # dk blue
           #'#b2df8a', # green
           '#33a02c', # # dk green
           '#6a3d9a', # dk purple
           '#cab2d6', # lt purple
           '#b15928') # brown
elev_levels <- cbind(rep(c('elev_1', 'elev_2', 'elev_3', 'elev_4', 'elev_5'), 4))
chi_levels <- cbind(rep('chi_1', 5), rep('chi_2', 5), rep('chi_3', 5), rep('chi_4', 5))                 

model_key_order <- c('Basic', 
                     'Vm','Ss','Th','Dd','Hy', 'Fi',
                     'St','Vs',
                     'Ch',  
                     'Rt', 'Sa', 
                     'Cv')

model_key_labels <- c('Basic', 
                      'Variable M Exponent', 'Shear Stress', 
                      'Threshold', 'Depth Dependent Thresh',
                      'Erosion-Deposition', 'Presence of Fines',
                      'Stochastic Hydrology','Variable Source Area',
                      'Non Linear Hillslope', 
                      'Rock Till Lithology', 'Explicit Regolith', 
                      'Variable Paleoclimate')

names(colors) <- model_key_labels

match_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'auxillary_inputs')
match_file <- dir(match_fp, pattern='model_name_element_match.csv', full.names=TRUE)
match_df <- read.table(match_file,header=TRUE,sep=",",row.names=NULL, na.strings = "")
names(match_df)[names(match_df) == 'ID'] <- 'model_name'

match_df$First.Element <- ordered(mapvalues(match_df$First.Element, 
                                            from=model_key_order,
                                            to=model_key_labels),
                                  levels=model_key_labels)

match_df$Second.Element <- ordered(mapvalues(match_df$Second.Element, 
                                             from=model_key_order,
                                             to=model_key_labels),
                                   levels=model_key_labels)

match_df$Third.Element <- ordered(mapvalues(match_df$Third.Element, 
                                            from=model_key_order,
                                            to=model_key_labels),
                                  levels=model_key_labels)
for (loc in c('sew', 'gully')) {
  fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'calibration', loc)
  
  df_list <- list()
  
  ego_summary_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'calibration', loc, paste('ego2.', loc, '.calibration.summary.csv', sep=''))
  summary_df <- read.table(ego_summary_fp, header=TRUE,sep=",",row.names=NULL)

  
  
  for(infile in dir(fp, pattern='ego2.*.residuals.full.*', full.names=TRUE)) {
    
    
    file_list  <- unlist(strsplit(infile, fsep))
    file_name <- file_list[8]
    model_key <- unlist(strsplit(file_name, "\\."))[5]
    if(nchar(model_key)<10){
      print(model_key)
      
      itter <- length(df_list) + 1
      
      model_name <- unlist(strsplit(unlist(strsplit(file_name, "\\."))[5], '_'))[2]
      
      data <- read.table(infile,header=TRUE,sep=",",row.names=NULL)
      data$model_name <- model_name
      data$objective_function <- sum(data$best_residual_values^2)
      
      # add 95% confidence values
      data$ego_only_of <- summary_df$ego_objective_function[summary_df$X == model_key]
      data$objective_function_95_t_statistic <- summary_df$objective_function_95_t_statistic[summary_df$X == model_key]
      data$objective_function_95_f_statistic <- summary_df$objective_function_95_f_statistic[summary_df$X == model_key]
      df_list[[itter]] <- data
    }
  }
  
  df <- df_list %>% reduce(bind_rows)
  df$chi_elev <- ordered(df$X, levels = chi_elev_levels)
  df$elevation <- ordered(mapvalues(df$chi_elev, 
                                    from = chi_elev_levels,
                                    to = elev_levels),
                          levels=c('elev_5', 'elev_4', 'elev_3', 'elev_2', 'elev_1'))
  df$chi <- ordered(mapvalues(df$chi_elev, 
                                    from = chi_elev_levels,
                                    to = chi_levels))
  
  of_dat <- data.frame(unique(select(df, model_name, objective_function)))
  of_dat <- arrange(of_dat, objective_function)
  model_order <- of_dat$model_name
                                     
  df <- merge(x=df, y=match_df, by="model_name",all.x=TRUE)
  df$model_name <- ordered(df$model_name, levels=model_order)
  
  # compare EGO and combined
  p<- df %>%
    select(model_name, Second.Element, First.Element, Third.Element, Model.Type, ego_only_of, objective_function) %>%
    distinct() %>%
    ggplot(aes(ego_only_of, objective_function, colour=Second.Element, fill=First.Element, shape=Model.Type)) +
    geom_point(size=1.5, stroke=1.5) +
    geom_point(size=1, stroke=0, aes(fill=Third.Element))+
    geom_abline(intercept = 0, slope = 1) +
    theme_bw(base_size=9) +  ylim(0, NA) +
    xlab('EGO Only Objective Function')+xlim(0, NA)+
    ylab('EGO+NL2SOL Objective Function')+ylim(0, NA)+
    scale_colour_manual(values=colors, name = "Process Element", labels=names(colors), limits=names(colors)) +
    scale_fill_manual(values=colors) +
    scale_shape_manual(values = c(21, 22, 23, 24), name = "Model Type") +
    guides(fill=FALSE, size=FALSE,  
           colour = guide_legend(order = 2), 
           shape = guide_legend(order = 1)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))#+
  #ggtitle(paste(loc, 'Calibration Summary'))
  figure_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'calibration', loc, 'EGO2', paste(loc, 'EGO_EGONL2_comparison.pdf', sep='.'))
  ggsave(figure_fp, width=6.5, height=4, units='in')
  
  #stochastic
  #Basic - BasicSt
  #BasicVs - BasicStVs
  #BasicHy - BasicHySt
  #BasicDd - BasicDdSt
  #BasicTh - BasicThSt
  #BasicSs - BasicSsSt
  
  
  
  dstoch<- df %>%
    select(Model.Name, Second.Element, First.Element, Third.Element, Model.Type, objective_function) %>%
    distinct() %>%
    filter(Model.Name %in% c('Basic', 'BasicSt', 
                            'BasicVs', 'BasicStVs',
                            'BasicHy', 'BasicHySt',
                            'BasicDd', 'BasicDdSt',
                            'BasicTh', 'BasicThSt',
                            'BasicSs', 'BasicSsSt')) %>%
    mutate(stochastic = case_when(Model.Name %in% c('BasicSt','BasicStVs','BasicHySt','BasicDdSt','BasicThSt','BasicSsSt') ~ "stochastic",
                                  Model.Name %in% c('Basic','BasicVs', 'BasicHy', 'BasicDd', 'BasicTh', 'BasicSs')  ~ "non_stochastic"),
           base_model = case_when(Model.Name %in% c('Basic', 'BasicSt') ~ 'Basic',
                                  Model.Name %in% c('BasicVs', 'BasicStVs') ~ 'BasicVs',
                                  Model.Name %in% c('BasicHy', 'BasicHySt') ~ 'BasicHy',
                                  Model.Name %in% c('BasicDd', 'BasicDdSt') ~ 'BasicDd',
                                  Model.Name %in% c('BasicTh', 'BasicThSt') ~ 'BasicTh',
                                  Model.Name %in% c('BasicSs', 'BasicSsSt') ~ 'BasicSs'))
  dstoch_stoch <- dstoch %>% filter(stochastic=='stochastic') %>% select(base_model, objective_function) %>% rename(stochastic_objective_function = objective_function)
  p <- dstoch %>% filter(stochastic=='non_stochastic') %>%
    left_join(., dstoch_stoch, by='base_model') %>%
    ggplot(aes(objective_function, stochastic_objective_function, colour=Second.Element, fill=First.Element)) +
    geom_point(size=1.5, stroke=1.5) +
    geom_abline(intercept = 0, slope = 1) +
    theme_bw(base_size=9) +
    ylab('Stochastic Model Objective Function')+xlim(0, NA)+
    xlab('Base Model Objective Function')+ylim(0, NA)+
    scale_colour_manual(values=colors, name = "Process Element", labels=names(colors), limits=names(colors)) +
    scale_fill_manual(values=colors) +
    guides(fill=FALSE, size=FALSE,  
           colour = guide_legend(order = 2)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))#+
  #ggtitle(paste(loc, 'Calibration Summary'))
  
  print(p)
  figure_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'calibration', loc, 'EGO2', paste(loc, 'stochastic_comparison.pdf', sep='.'))
  ggsave(figure_fp, width=6.5, height=4, units='in')
    

  
  # first plot each residual element
  p <- ggplot(df, aes(model_name, best_residual_values, colour=Second.Element, fill=First.Element, shape=Model.Type)) +
    geom_point(size=1.5, stroke=1.5) +
    geom_point(size=1, stroke=0, aes(fill=Third.Element))+
    theme_bw(base_size=9) +  ylim(0, NA) +
    xlab('Model Name')+
    ylab('Calibrated Residual Value')+
    scale_colour_manual(values=colors, name = "Process Element", labels=names(colors), limits=names(colors)) +
    scale_fill_manual(values=colors) +
    scale_shape_manual(values = c(21, 22, 23, 24), name = "Model Type") +
    facet_grid(elevation~chi)+
    guides(fill=FALSE, size=FALSE,  
           colour = guide_legend(order = 2), 
           shape = guide_legend(order = 1)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))#+
    #ggtitle(paste(loc, 'Calibration Summary'))
  
  print(p)
  figure_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'calibration', loc, 'EGO2', paste(loc, 'calibration_summary_figure.pdf', sep='.'))
  ggsave(figure_fp, width=6.5, height=8, units='in')
  
  of_df <- merge(x=of_dat, y=match_df, by="model_name",all.x=TRUE)
  of_df$model_name <- ordered(of_df$model_name, levels=model_order)
  
  p <- ggplot(df, aes(model_name, objective_function, colour=Second.Element, fill=First.Element, shape=Model.Type)) +
    geom_point(size=2, stroke=1) +
    geom_point(size=1, stroke=0, aes(fill=Third.Element, colour=First.Element))+
    theme_bw(base_size=9) +
    ylim(0, NA) +
    xlab('Model Name')+
    ylab('Sum of Squares Objective Function')+
    scale_colour_manual(values=colors, name = "Process Element", labels=names(colors), limits=names(colors)) +
    scale_fill_manual(values=colors) +
    scale_shape_manual(values = c(21, 22, 23, 24), name = "Model Type") +
    guides(fill=FALSE, size=FALSE,  
           colour = guide_legend(order = 2), 
           shape = guide_legend(order = 1)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))#+
    #ggtitle(paste(loc, 'Calibration Summary'))
  print(p)
  figure_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'calibration', loc, 'EGO2', paste(loc, 'calibration_of_figure.pdf', sep='.'))
  ggsave(figure_fp, width=6.5, height=5, units='in')
  
  # first build, no points,
  #df_build <- df %>% filter(Model.Type == " Base Model")
  p <- ggplot(df, aes(model_name, objective_function, colour=Second.Element, fill=First.Element, shape=Model.Type)) +
    geom_point(size=2, stroke=1, alpha=0.00005) +
    #geom_point(mapping=aes(model_name, objective_function, colour=Second.Element, fill=First.Element, shape=Model.Type), 
    #           data=df_build, 
    #           size=2, stroke=1) +
    theme_bw(base_size=9) +
    ylim(0, NA) +
    xlab('Model Name')+
    ylab('Sum of Squares Objective Function')+
    scale_colour_manual(values=colors, name = "Process Element", labels=names(colors), limits=names(colors)) +
    scale_fill_manual(values=colors) +
    scale_shape_manual(values = c(21, 22, 23, 24), name = "Model Type") +
    guides(fill=FALSE, size=FALSE,  
           colour = guide_legend(order = 2), 
           shape = guide_legend(order = 1)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))#+
    #ggtitle(paste(loc, 'Calibration Summary'))
  print(p)
  figure_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'calibration', loc, 'EGO2', paste(loc, 'calibration_of_figure.build.0.pdf', sep='.'))
  ggsave(figure_fp, width=6.5, height=5, units='in')
  
  # second build, only basic
  df_build <- df %>% filter(Model.Type == " Base Model")
  p <- ggplot(df, aes(model_name, objective_function, colour=Second.Element, fill=First.Element, shape=Model.Type)) +
    geom_point(size=2, stroke=1, alpha=0.00005) +
    geom_point(mapping=aes(model_name, objective_function, colour=Second.Element, fill=First.Element, shape=Model.Type), 
               data=df_build, 
               size=2, stroke=1) +
    theme_bw(base_size=9) +
    ylim(0, NA) +
    xlab('Model Name')+
    ylab('Sum of Squares Objective Function')+
    scale_colour_manual(values=colors, name = "Process Element", labels=names(colors), limits=names(colors)) +
    scale_fill_manual(values=colors) +
    scale_shape_manual(values = c(21, 22, 23, 24), name = "Model Type") +
    guides(fill=FALSE, size=FALSE,  
           colour = guide_legend(order = 2), 
           shape = guide_legend(order = 1)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))#+
    #ggtitle(paste(loc, 'Calibration Summary'))
  print(p)
  figure_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'calibration', loc, 'EGO2', paste(loc, 'calibration_of_figure.build.1.pdf', sep='.'))
  ggsave(figure_fp, width=6.5, height=5, units='in')
  
  # second build, only one component or basic
  df_build <- df %>% filter(Model.Type %in% c(" Base Model", " One Component"))
  p <- ggplot(df, aes(model_name, objective_function, colour=Second.Element, fill=First.Element, shape=Model.Type)) +
    geom_point(size=2, stroke=1, alpha=0.00005) +
    geom_point(mapping=aes(model_name, objective_function, colour=Second.Element, fill=First.Element, shape=Model.Type), 
               data=df_build, 
               size=2, stroke=1) +
    theme_bw(base_size=9) +
    ylim(0, NA) +
    xlab('Model Name')+
    ylab('Sum of Squares Objective Function')+
    scale_colour_manual(values=colors, name = "Process Element", labels=names(colors), limits=names(colors)) +
    scale_fill_manual(values=colors) +
    scale_shape_manual(values = c(21, 22, 23, 24), name = "Model Type") +
    guides(fill=FALSE, size=FALSE,  
           colour = guide_legend(order = 2), 
           shape = guide_legend(order = 1)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))#+
    #ggtitle(paste(loc, 'Calibration Summary'))
  print(p)
  figure_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'calibration', loc, 'EGO2', paste(loc, 'calibration_of_figure.build.2.pdf', sep='.'))
  ggsave(figure_fp, width=6.5, height=5, units='in')
  
  # third build, only one, or two, component or basic
  df_build <- df %>% filter(Model.Type %in% c(" Base Model", " One Component", " Two Component"))
  p <- ggplot(df, aes(model_name, objective_function, colour=Second.Element, fill=First.Element, shape=Model.Type)) +
    geom_point(size=2, stroke=1, alpha=0.00005) +
    geom_point(mapping=aes(model_name, objective_function, colour=Second.Element, fill=First.Element, shape=Model.Type), 
               data=df_build, 
               size=2, stroke=1) +
    theme_bw(base_size=9) +
    ylim(0, NA) +
    xlab('Model Name')+
    ylab('Sum of Squares Objective Function')+
    scale_colour_manual(values=colors, name = "Process Element", labels=names(colors), limits=names(colors)) +
    scale_fill_manual(values=colors) +
    scale_shape_manual(values = c(21, 22, 23, 24), name = "Model Type") +
    guides(fill=FALSE, size=FALSE,  
           colour = guide_legend(order = 2), 
           shape = guide_legend(order = 1)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))#+
    #ggtitle(paste(loc, 'Calibration Summary'))
  print(p)
  figure_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'calibration', loc, 'EGO2', paste(loc, 'calibration_of_figure.build.3.pdf', sep='.'))
  ggsave(figure_fp, width=6.5, height=5, units='in')
  
  # final build, all models
  p <- ggplot(df, aes(model_name, objective_function, colour=Second.Element, fill=First.Element, shape=Model.Type)) +
    geom_point(size=2, stroke=1) +
    geom_point(size=1, stroke=0, aes(fill=Third.Element, colour=First.Element))+
    theme_bw(base_size=9) +
    ylim(0, NA) +
    xlab('Model Name')+
    ylab('Sum of Squares Objective Function')+
    scale_colour_manual(values=colors, name = "Process Element", labels=names(colors), limits=names(colors)) +
    scale_fill_manual(values=colors) +
    scale_shape_manual(values = c(21, 22, 23, 24), name = "Model Type") +
    guides(fill=FALSE, size=FALSE,  
           colour = guide_legend(order = 2), 
           shape = guide_legend(order = 1)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))#+
    #ggtitle(paste(loc, 'Calibration Summary'))
  print(p)
  figure_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'calibration', loc, 'EGO2', paste(loc, 'calibration_of_figure.build.4.pdf', sep='.'))
  ggsave(figure_fp, width=6.5, height=5, units='in')
  
  # final build, all models WITH Error bars
  p <- ggplot(df, aes(model_name, objective_function, colour=Second.Element, fill=First.Element, shape=Model.Type)) +
    geom_errorbar(aes(ymin=objective_function, ymax=objective_function_95_f_statistic)) + 
    geom_errorbar(aes(ymin=objective_function, ymax=objective_function_95_t_statistic)) +
    geom_point(size=2, stroke=1) +
    geom_point(size=1, stroke=0, aes(fill=Third.Element, colour=First.Element))+
    theme_bw(base_size=9) +
    ylim(0, NA) +
    xlab('Model Name')+
    ylab('Sum of Squares Objective Function')+
    scale_colour_manual(values=colors, name = "Process Element", labels=names(colors), limits=names(colors)) +
    scale_fill_manual(values=colors) +
    scale_shape_manual(values = c(21, 22, 23, 24), name = "Model Type") +
    guides(fill=FALSE, size=FALSE,  
           colour = guide_legend(order = 2), 
           shape = guide_legend(order = 1)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))#+
    #ggtitle(paste(loc, 'Calibration Summary'))
  print(p)
  figure_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'calibration', loc, 'EGO2', paste(loc, 'calibration_of_figure.with_error.pdf', sep='.'))
  ggsave(figure_fp, width=6.5, height=5, units='in')
  
  
  if(loc=='sew'){
    valid_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'validation', loc, 'BEST_PARAMETERS_SMALL_DOMAIN', 'validation_summary.csv')
    valid_df <- read.table(valid_fp, header=TRUE,sep=",",row.names=NULL)
    
    ag <- valid_df %>% dplyr::select(model, objective_function) %>%
                       dplyr::group_by(model) %>%
                       dplyr::summarise(validation.objective_function.mean = mean(objective_function),
                                 validation.objective_function.sd = sd(objective_function)) %>%
                       separate(model, c('temp', 'model_name'), sep='_', remove=FALSE, convert=FALSE) %>%
                       select(-model, -temp)
    

    valid_ag_df <- merge(x=df, y=ag, by="model_name", all.x=TRUE)
    
    
    p <- ggplot(valid_ag_df, aes(objective_function, validation.objective_function.mean, colour=Second.Element, fill=First.Element, shape=Model.Type)) +
      #geom_errorbar(aes(ymin = validation.objective_function.mean-2*validation.objective_function.sd, 
      #                  ymax = validation.objective_function.mean+2*validation.objective_function.sd)) +
      geom_point(size=2, stroke=1) +
      geom_point(size=1, stroke=0, aes(fill=Third.Element, colour=First.Element))+
      theme_bw(base_size=9) +
      ylim(0, NA) +
      xlim(0, NA) +
      xlab('Calibration - Sum of Squares Objective Function')+
      ylab('Validation - Sum of Squares Objective Function')+
      scale_colour_manual(values=colors, name = "Process Element", labels=names(colors), limits=names(colors)) +
      scale_fill_manual(values=colors) +
      scale_shape_manual(values = c(21, 22, 23, 24), name = "Model Type") +
      guides(fill=FALSE, size=FALSE,  
             colour = guide_legend(order = 2), 
             shape = guide_legend(order = 1)) +
      theme(axis.text.x = element_text(angle = 90, hjust = 1))#+
      #ggtitle(paste(loc, 'Calibration - Validation Summary'))
    print(p)
    figure_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'calibration', loc, 'EGO2', paste(loc, 'calibration_of_figure.withValidation.pdf', sep='.'))
    ggsave(figure_fp, width=6.5, height=5, units='in')
    
  }
}
 # 




