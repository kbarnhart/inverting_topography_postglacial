library(plyr)
library(tidyverse)
library(latex2exp)

across_color_list = c('#bdbdbd', '#1b9e77','#d95f02')

chi_elev_levels <- c('chi_elev_01', 'chi_elev_02', 'chi_elev_03', 'chi_elev_04', 'chi_elev_05',
                     'chi_elev_06', 'chi_elev_07', 'chi_elev_08', 'chi_elev_09', 'chi_elev_10',
                     'chi_elev_11', 'chi_elev_12', 'chi_elev_13', 'chi_elev_14', 'chi_elev_15',
                     'chi_elev_16', 'chi_elev_17', 'chi_elev_18', 'chi_elev_19', 'chi_elev_20')

elev_levels <- cbind(rep(c('Lowest 20% Elevation', '20%-40% Elevation', '40%-60% Elevation', '60%-80% Elevation', '80%-100% Elevation'), 4))
chi_levels <- cbind(rep('Lowest 5% Chi', 5), rep('5% - 20% Chi', 5), rep('20% - 50% Chi', 5), rep('50% - 100% Chi', 5))   

elev_labels <- rev(c('Lowest 20% Elevation', '20%-40% Elevation', '40%-60% Elevation', '60%-80% Elevation', '80%-100% Elevation'))
chi_labels <- c('Lowest 5% Chi', '5% - 20% Chi', '20% - 50% Chi', '50% - 100% Chi')  

ASV_levels <- cbind( 'ASV_mean_elevation', 'ASV_var_elevation', 'ASV_var_gradient', 'ASV_mean_gradient', 
                     'ASV_mean_elevation_chi_area', 'ASV_var_elevation_chi_area', 'ASV_mean_gradient_chi_area', 'ASV_var_gradient_chi_area',
                     'ASV_one_cell_nodes', 'ASV_two_cell_nodes', 'ASV_three_cell_nodes', 'ASV_four_cell_nodes', 
                     'ASV_cumarea95', 'ASV_cumarea96', 'ASV_cumarea97', 'ASV_cumarea98', 
                     'ASV_cumarea99',  'ASV_chi_density_sum_squares','ASV_chi_gradient','ASV_chi_intercept',                   
                     'ASV_elev02', 'ASV_elev08', 'ASV_elev23', 'ASV_elev30', 
                     'ASV_elev36', 'ASV_elev50', 'ASV_elev75', 'ASV_elev85', 
                     'ASV_elev90','ASV_elev96', 'ASV_elev100', 'ASV_hypsometric_integral')

ASV_labels <- cbind( 'mean elevation', 'var elevation', 'var gradient', 'mean gradient', 
                     'mean elevation chi area', 'var elevation chi area', 'mean gradient chi area', 'var gradient chi area',
                     'one cell nodes', 'two cell nodes', 'three cell nodes', 'four cell nodes', 
                     'cumulative area95', 'cumulative area96', 'cumulative area97', 'cumulative area98', 
                     'cumulative area99',  'chi density sum squares','chi gradient','chi intercept',                   
                     'elev02', 'elev08', 'elev23', 'elev30', 
                     'elev36', 'elev50', 'elev75', 'elev85', 
                     'elev90','elev96', 'elev100', 'hypsometric integral')

match_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'auxillary_inputs')
match_file <- dir(match_fp, pattern='model_name_element_match.csv', full.names=TRUE)
match_df <- read.table(match_file,header=TRUE,sep=",",row.names=NULL, na.strings = "")
names(match_df)[names(match_df) == 'ID'] <- 'model_name'

fsep = .Platform$file.sep
for (loc in c('sew', 'gully')) {
  fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'sensitivity_analysis', loc)
  cat_of_list <- list()
  
  for(param.file_path in dir(fp, pattern='*.morris_df_short.*.parameter.csv', full.names=TRUE)) {
    
    
    lowering.file_path <- gsub('parameter', 'lowering', param.file_path)
    initial.file_path <- gsub('parameter', 'initial', param.file_path)
    
    param.file_list  <- unlist(strsplit(param.file_path, fsep))
    param.file_name <- param.file_list[8]
    
    of_name = unlist(strsplit(param.file_name, "\\."))[1]
    
    model_name = unlist(strsplit(param.file_name, "\\."))[3]
    model.ID <- unlist(strsplit(model_name, '_'))[2]
    model.code <- as.character(unlist(match_df$Model.Name[match_df$model_name==model.ID])[1])
    
    catagorical.file_list  <- unlist(strsplit(lowering.file_path, fsep))
    catagorical.file_name <- catagorical.file_list[8]

    param.outfile <- gsub('csv', 'scatter.pdf', param.file_name)
    catagorical.outfile <- gsub('csv', 'scatter.pdf', catagorical.file_name)

    param.outfile.bar <- gsub('csv', 'bar.pdf', param.file_name)
    catagorical.outfile.bar <- gsub('csv', 'bar.pdf', catagorical.file_name)

    out_folder_list <- param.file_list
    out_folder_list[5] <- 'result_figures'
    
    param.out_full_path = paste(paste(out_folder_list[1:7], sep='', collapse = fsep), param.outfile, sep=fsep)
    catagorical.out_full_path = paste(paste(out_folder_list[1:7], sep='', collapse = fsep), catagorical.outfile, sep=fsep)
    param.bar.out_full_path = paste(paste(out_folder_list[1:7], sep='', collapse = fsep), param.outfile.bar, sep=fsep)
    catagorical.bar.out_full_path = paste(paste(out_folder_list[1:7], sep='', collapse = fsep), catagorical.outfile.bar, sep=fsep)

    
    param.data <- read.table(param.file_path,header=TRUE,sep=",",row.names=NULL)
    param.data <- param.data[is.finite(param.data$mu_star+param.data$sigma_star),]

    
    lowering.data <- read.table(lowering.file_path,header=TRUE,sep=",",row.names=NULL)
    lowering.data <- lowering.data[is.finite(lowering.data$mu_star+lowering.data$sigma_star),]

    initial.data <- read.table(initial.file_path,header=TRUE,sep=",",row.names=NULL)
    initial.data <- initial.data[is.finite(initial.data$mu_star+initial.data$sigma_star),]

    if(of_name %in% c('cat', 'orig')){ 
      if(of_name == 'cat'){
        param.data$Elevation <- ordered(mapvalues(param.data$Metric, 
                                                  from = chi_elev_levels,
                                                  to = elev_levels),
                                        levels=elev_labels)
        param.data$Chi <- ordered(mapvalues(param.data$Metric, 
                                            from = chi_elev_levels,
                                            to = chi_levels),
                                  levels=chi_labels)
        
        lowering.data$Elevation <- ordered(mapvalues(lowering.data$Metric, 
                                                     from = chi_elev_levels,
                                                     to = elev_levels),
                                           levels=elev_labels)
        lowering.data$Chi <- ordered(mapvalues(lowering.data$Metric, 
                                               from = chi_elev_levels,
                                               to = chi_levels),
                                     levels=chi_labels)
        
        initial.data$Elevation <- ordered(mapvalues(initial.data$Metric, 
                                                    from = chi_elev_levels,
                                                    to = elev_levels),
                                          levels=elev_labels)
        initial.data$Chi <- ordered(mapvalues(initial.data$Metric, 
                                              from = chi_elev_levels,
                                              to = chi_levels),
                                    levels=chi_labels)
      }else{
        param.data$Metric <- ordered(mapvalues(param.data$Metric, 
                                               from = ASV_levels, 
                                               to = ASV_labels),
                                     levels = ASV_labels)
        
        lowering.data$Metric <- ordered(mapvalues(lowering.data$Metric, 
                                                  from = ASV_levels, 
                                                  to = ASV_labels),
                                        levels = ASV_labels)
        
        initial.data$Metric <- ordered(mapvalues(initial.data$Metric, 
                                                 from = ASV_levels, 
                                                 to = ASV_labels),
                                       levels = ASV_labels)
      } 
    }
    
    # first a mu star vs sigma start plot
    p <- ggplot(param.data, aes(mu_star, sigma_star, color=Input, shape=Initial.Condition)) +
      geom_point() +
      theme_bw(base_size=9) +
      scale_shape(name = "Initial Condition", labels=lapply(levels(param.data$Initial.Condition), TeX)) +
      scale_colour_hue(name = "Input Parameter", labels=lapply(levels(param.data$Input), TeX)) +
      guides(colour = guide_legend(order = 1, nrow=5), 
             shape = guide_legend(order = 2)) +
      xlim(0, NA) + ylim(0, NA) +
      xlab(TeX('$\\mu^*$'))+
      ylab(TeX('$\\sigma^*$'))+
      ggtitle(paste('Model ', model.ID, ' (', model.code, ')', sep=''))
    
    if(of_name %in% c('cat', 'orig')){ 
      if(of_name == 'cat'){
        p <- p + facet_grid(Elevation~Chi)
        yrange <- ggplot_build(p)$layout$panel_ranges[[1]]$y.range
        xrange <- ggplot_build(p)$layout$panel_ranges[[1]]$x.range
        
        label_data <- param.data %>% select(Metric, Elevation, Chi) %>% distinct() 
        p <- p  + geom_text(data=label_data, aes(x=(xrange[1] + 0.05*(xrange[2]- xrange[1])), 
                                                 y=(yrange[1] + 0.90*(yrange[2]- yrange[1])), label=Metric), 
                            color='black', hjust = 0, vjust=0, size=2.5, inherit.aes = FALSE, show.legend = FALSE)
      }else{
        
        p <- p + facet_wrap(~Metric, ncol = 4, scales = 'free')
        
      }
      ggsave(param.out_full_path, width=6.5, height=8, units='in')
    }else{
      ggsave(param.out_full_path, width=5.5, height=3.5, units='in')
    }
    
    # next, the equivalent version that is a bar plot
    if(of_name %in% c('cat', 'orig')){ 
      if(of_name == 'cat'){
        p <- param.data %>% select(Input, Metric, mu_star, sigma_star, Chi, Elevation) %>% 
          group_by(Input, Metric, Chi, Elevation) %>%
          summarise(mu_star = mean(mu_star), 
                    sigma_star = mean(sigma_star)) %>%
          ggplot(aes(x=Input, y=mu_star, fill=Input)) +
          geom_bar(position=position_dodge(), stat="identity") +
          #geom_errorbar(aes(ymin=mu_star-sigma_star, ymax=mu_star+sigma_star))+
          theme_bw(base_size=9) +
          scale_fill_hue(name = "", labels=lapply(levels(param.data$Input), TeX)) +
          scale_x_discrete("Input Parameter", labels=NULL)+
          ylab('Mean Sensitivity')+
          ggtitle(paste('Model ', model.ID, ' (', model.code, ')', sep='')) +
          facet_grid(Elevation~Chi)
        
        yrange <- ggplot_build(p)$layout$panel_ranges[[1]]$y.range
        xrange <- ggplot_build(p)$layout$panel_ranges[[1]]$x.range
        
        label_data <- param.data %>% select(Metric, Elevation, Chi) %>% distinct() 
        
        p <- p + geom_text(data=label_data, aes(x=(xrange[1] + 0.05*(xrange[2]- xrange[1])), 
                                                y=(yrange[1] + 0.90*(yrange[2]- yrange[1])), label=Metric), 
                           color='black', hjust = 0, vjust=0, size=2.5, inherit.aes = FALSE, show.legend = FALSE)
      }else{
        
        p <- param.data %>% select(Input, Metric, mu_star, sigma_star) %>% 
          group_by(Input, Metric) %>%
          summarise(mu_star = mean(mu_star), 
                    sigma_star = mean(sigma_star)) %>%
          ggplot(aes(x=Input, y=mu_star, fill=Input)) +
          geom_bar(position=position_dodge(), stat="identity") +
          #geom_errorbar(aes(ymin=mu_star-sigma_star, ymax=mu_star+sigma_star))+
          theme_bw(base_size=9) +
          scale_fill_hue(name = "", labels=lapply(levels(param.data$Input), TeX)) +
          scale_x_discrete("Input Parameter", labels=NULL)+
          ylab('Mean Sensitivity')+
          ggtitle(paste('Model ', model.ID, ' (', model.code, ')', sep=''))
        
        p <- p + facet_wrap(~Metric, ncol = 4, scales = 'free')
        
      }
      
      ggsave(param.bar.out_full_path, width=6.5, height=8, units='in')
      
    }else{
      
      p <- param.data %>% select(Input, mu_star, sigma_star) %>% 
        group_by(Input) %>%
        summarise(mu_star = mean(mu_star), 
                  sigma_star = mean(sigma_star)) %>%
        ggplot(aes(x=Input, y=mu_star, fill=Input)) +
        geom_bar(position=position_dodge(), stat="identity") +
        #geom_errorbar(aes(ymin=mu_star-sigma_star, ymax=mu_star+sigma_star))+
        theme_bw(base_size=9) +
        scale_fill_hue(name = "", labels=lapply(levels(param.data$Input), TeX)) +
        scale_x_discrete("Input Parameter", labels=NULL)+
        ylab('Mean Sensitivity')+
        ggtitle(paste('Model ', model.ID, ' (', model.code, ')', sep=''))
      
      ggsave(param.bar.out_full_path, width=5, height=3.5, units='in')
      
    }
    
    
    #%% lowering
    
    # combine datframes so that the legend is tractable.
    # use manual colors. 
    # then do the same for the initial variations. 
    lowering.data$Input <- as.factor(lowering.data$Lowering.History)
    lowering.data$Vary <- 'Lowering History'
    initial.data$Input <-initial.data$Initial.Condition
    initial.data$Vary <- 'Initial Condition'
    
    param.data$Vary <- 'Continuous Parameter'
    
    plot.data <- rbind(param.data, initial.data, lowering.data)
    plot.data$Vary <- ordered(plot.data$Vary, levels=c('Continuous Parameter', 'Initial Condition', 'Lowering History'))
    
    p <- ggplot(plot.data, aes(mu_star, sigma_star, color=Vary)) +
      geom_point() +
      theme_bw(base_size=9) +
      scale_colour_manual(values=across_color_list,name = "Type of Parameter Varied") +
      #scale_shape(name = "Initial Condition", labels=lapply(levels(param.data$Initial.Condition), TeX)) +
      guides(colour = guide_legend(order = 1), 
             shape = guide_legend(order = 2)) +
      xlim(0, NA) + ylim(0, NA) +
      xlab(TeX('$\\mu^*$'))+
      ylab(TeX('$\\sigma^*$')) #+
      #ggtitle(paste('Parameter, Initial Condition, and Lowering Sensitivity:\n', 'Model ', model.ID, ' (', model.code, ')', sep='')) 
    
    if(of_name %in% c('cat', 'orig')){ 
      if(of_name == 'cat'){
        p <- p + facet_grid(Elevation~Chi) 
        
        yrange <- ggplot_build(p)$layout$panel_ranges[[1]]$y.range
        xrange <- ggplot_build(p)$layout$panel_ranges[[1]]$x.range
        
        label_data <- plot.data %>% select(Metric, Elevation, Chi) %>% distinct() 
        p <- p + geom_text(data=label_data, aes(x=(xrange[1] + 0.05*(xrange[2]- xrange[1])), 
                                                y=(yrange[1] + 0.90*(yrange[2]- yrange[1])), label=Metric), 
                           color='black', hjust = 0, vjust=0, size=2.5, inherit.aes = FALSE, show.legend = FALSE)
      }else{
        p <- p + facet_wrap(~Metric, ncol = 4, scales = 'free')
      }
      ggsave(catagorical.out_full_path, width=6.5, height=8, units='in')
    }else{
      ggsave(catagorical.out_full_path, width=5.2, height=3.5, units='in')
    }
    
    # next, the equivalent version that is a bar plot
    lowering.data$Input <-'Lowering History'
    initial.data$Input <- 'Initial Condition'
    plot.data <- rbind(param.data, initial.data, lowering.data)
    
    
    if(of_name %in% c('cat', 'orig')){ 
      if(of_name =='cat'){
        p <- plot.data %>% select(Input, Metric, mu_star, sigma_star, Chi, Elevation) %>% 
          group_by(Input, Metric, Chi, Elevation) %>%
          summarise(mu_star = mean(mu_star), 
                    sigma_star = mean(sigma_star)) %>%
          ggplot(aes(x=Input, y=mu_star, fill=Input)) +
          geom_bar(position=position_dodge(), stat="identity") +
          #geom_errorbar(aes(ymin=mu_star-sigma_star, ymax=mu_star+sigma_star))+
          theme_bw(base_size=9) +
          scale_fill_hue(name = "", labels=lapply(levels(plot.data$Input), TeX)) +
          scale_x_discrete("Input Parameter, Initial Condition, or Lowering History", labels=NULL)+
          ylab('Mean Sensitivity')+
          #ggtitle(paste('Parameter, Initial Condition, and Lowering Sensitivity:\n', 'Model ', model.ID, ' (', model.code, ')', sep='')) + 
          facet_grid(Elevation~Chi) 
        yrange <- ggplot_build(p)$layout$panel_ranges[[1]]$y.range
        xrange <- ggplot_build(p)$layout$panel_ranges[[1]]$x.range
        label_data <- plot.data %>% select(Metric, Elevation, Chi) %>% distinct() 
        p <- p + geom_text(data=label_data, aes(x=(xrange[1] + 0.05*(xrange[2]- xrange[1])), 
                                                y=(yrange[1] + 0.90*(yrange[2]- yrange[1])), label=Metric), 
                           color='black', hjust = 0, vjust=0, size=2.5, inherit.aes = FALSE, show.legend = FALSE)
        
      }else{
        p <- plot.data %>% select(Input, Metric, mu_star, sigma_star) %>% 
          group_by(Input, Metric) %>%
          summarise(mu_star = mean(mu_star), 
                    sigma_star = mean(sigma_star)) %>%
          ggplot(aes(x=Input, y=mu_star, fill=Input)) +
          geom_bar(position=position_dodge(), stat="identity") +
          #geom_errorbar(aes(ymin=mu_star-sigma_star, ymax=mu_star+sigma_star))+
          theme_bw(base_size=9) +
          scale_fill_hue(name = "", labels=lapply(levels(plot.data$Input), TeX)) +
          scale_x_discrete("Input Parameter, Initial Condition, or Lowering History", labels=NULL)+
          ylab('Mean Sensitivity') +
          #ggtitle(paste('Parameter, Initial Condition, and Lowering Sensitivity:\n', 'Model ', model.ID, ' (', model.code, ')', sep=''))
          facet_wrap(~Metric, ncol = 4, scales = 'free')
      }
      ggsave(catagorical.bar.out_full_path, width=6.5, height=8, units='in')
    }else{
      p <- plot.data %>% select(Input, mu_star, sigma_star) %>% 
        group_by(Input) %>%
        summarise(mu_star = mean(mu_star), 
                  sigma_star = mean(sigma_star)) %>%
        ggplot(aes(x=Input, y=mu_star, fill=Input)) +
        geom_bar(position=position_dodge(), stat="identity") +
        #geom_errorbar(aes(ymin=mu_star-sigma_star, ymax=mu_star+sigma_star))+
        theme_bw(base_size=9) +
        scale_fill_hue(name = "", labels=lapply(levels(plot.data$Input), TeX)) +
        scale_x_discrete("Input Parameter, Initial Condition, or Lowering History", labels=NULL)+
        ylab('Mean Sensitivity')#+
        #ggtitle(paste('Parameter, Initial Condition, and Lowering Sensitivity:\n', 'Model ', model.ID, ' (', model.code, ')', sep=''))
      
      ggsave(catagorical.bar.out_full_path, width=5, height=3.5, units='in')
    }
    
    if(of_name == 'cat_of'){
      plot.data$model_code <- model.ID
      plot.data$model_name <- model.code
      cat_of_list[[length(cat_of_list) +1 ]] <- plot.data
    }
    

    # print(p)
    print(paste(loc, of_name, model_name))
  }
  # make synthesis plot
  
  df <- bind_rows(cat_of_list) %>% select(Input, mu_star, sigma_star, model_code) %>% 
    filter(model_code %in% c("800", "802", "804", "808", "810", "840", "842", "A00", "C00")) %>%
    group_by(Input, model_code) %>%
    summarise(mu_star_m = mean(mu_star)) 
  
  df$Input <- ordered(df$Input, levels=unique(df$Input))
  p <- df %>%
    ggplot(aes(x=Input, y=mu_star_m, fill=Input)) +
    geom_bar(position=position_dodge(), stat="identity") +
    facet_wrap(~model_code, ncol=2) +
    theme_bw(base_size=9) +
    scale_fill_hue(name = "", labels=lapply(levels(df$Input), TeX)) +
    scale_x_discrete("Input Parameter, Initial Condition, or Lowering History", labels=NULL)+
    guides(fill = guide_legend(ncol = 1)) +
    ylab('Mean Sensitivity')
  
  print(p)
  fig.out = paste(paste(out_folder_list[1:7], sep='', collapse = fsep), 'MOAT_summary.pdf', sep=fsep)
  ggsave(fig.out, width=6.5, height=8, units='in')
  
}


