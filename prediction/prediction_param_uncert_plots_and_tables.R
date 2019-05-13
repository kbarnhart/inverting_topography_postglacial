library(plyr)
library(tidyverse)
library(viridis)
library(ncdf4)
library(grid)
library(parallel)
library(png)

fsep = .Platform$file.sep
reference_depth = 50
used_models = c('800', '802', '804', '808', '842', 'A00', 'C00')
# create evaluations structure that ids the summary statistics saved 
evaluations = funs(elevation_mean = mean(.), 
                   elevation_std = sd(.),
                   elevation_cv = (sd(.)/mean(.)*100.0))

vals = c(seq(0, 4.9, 0.1), seq(5, 9.5, 0.5), seq(10, 90, 1), seq(90.5, 95, 0.5), seq(95.1, 100, 0.1))/100
for(v in vals){
  name = paste('elevation_quant_', v, sep='')
  evaluations[[name]] = quo(quantile(., probs=UQE(v)))
}


summarize_results <- function(data_fp){
  # open df_temp
  df_temp <- read.table(data_fp, header=TRUE)
  data_columns = colnames(df_temp)[which(endsWith(colnames(df_temp), '.0')): length(colnames(df_temp))]
  
  # gather and separate values
  df_temp_gather <- gather(df_temp, key, elevation, data_columns)
  df_temp_gather_separate <- separate(df_temp_gather, key, into=c('plot_location', 'model_time'), convert=TRUE, sep='\\.')
  df_model <- dplyr::select(df_temp_gather_separate, plot_location, model_time, elevation)
  remove(df_temp, df_temp_gather, df_temp_gather_separate)
  
  # model time is in 100 year increments, modify to be in Ka
  df_model$model_time <- as.numeric(df_model$model_time)/10.0
  
  # get and add additional variable names
  model_name <- unlist(strsplit(unlist(strsplit(data_fp, fsep))[8], '_'))[2]
  lowering_future <- unlist(strsplit(unlist(strsplit(data_fp, fsep))[9], '\\.'))[1]
  initial_condition <- unlist(strsplit(unlist(strsplit(data_fp, fsep))[9], '\\.'))[2]
  climate_future <- unlist(strsplit(unlist(strsplit(data_fp, fsep))[9], '\\.'))[3]
  
  df_model$model_name <- model_name
  df_model$lowering_future <- lowering_future
  df_model$initial_condition <- initial_condition
  df_model$climate_future <- climate_future
  
  # create a df that has summary statistics
  sum <- df_model %>% group_by(model_name, climate_future, lowering_future, initial_condition, plot_location, model_time) %>%
    summarise_all(evaluations)
  
  surface_elev <- sum %>% filter(model_time == 0) %>%
    dplyr::select(model_name, climate_future, lowering_future, initial_condition, plot_location, elevation_mean)
  
  # add a surface elevation and 
  sum$modern_surface_elevation <- surface_elev$elevation_mean
  
  sum # return summary
}# end function

# for each location
for (loc in c('gully', 'sew')){
  fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction', loc, 'PARAMETER_UNCERTAINTY')
  
  if (file.exists(fp)){ # if location DIR directory exits. 
    
    output_dir <- gsub('study3py', paste('study3py', 'result_figures', sep=fsep), fp) 
    if(dir.exists(output_dir) == FALSE){
      dir.create(output_dir, recursive=TRUE)
    } # end if dir doesn't exist
    
    table_dir <- gsub('study3py', paste('study3py', 'result_tables', sep=fsep), fp) 
    if(dir.exists(table_dir) == FALSE){
      dir.create(table_dir, recursive=TRUE)
    } # end
    
    # plot locations given as column, row (e.g. x, y)
    point_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction', loc, 'PredictionPoints_ShortList.csv')
    plot_loc_df <- read.csv(point_fp)
    
    plot_locations <- plot_loc_df$Point_Name
    
    save_list = list()
    
    for(pln in plot_locations){
      print(pln)
      
      out_file <- file.path(table_dir, paste('summary_table', loc, pln, 'csv', sep='.'))
      
      if(file.exists(out_file)){
        df_summary = read.csv(out_file)
      }else{
        all_data_fps = list()
        
        # for each model
        for(model_folder in dir(fp, pattern = 'model_*', full.names = TRUE)) {
          #print(model_folder)
          
          model_key <- unlist(strsplit(model_folder, fsep))[8]
          model_name <- unlist(strsplit(unlist(strsplit(model_folder, fsep))[8], '_'))[2]
          
          # get climage/lowering/IC future
          for(cli_folder in dir(model_folder, pattern = 'lowering*', full.names = TRUE)){
            
            lowering_future <- unlist(strsplit(unlist(strsplit(cli_folder, fsep))[9], '\\.'))[1]
            initial_condition <- unlist(strsplit(unlist(strsplit(cli_folder, fsep))[9], '\\.'))[2]
            climate_future <- unlist(strsplit(unlist(strsplit(cli_folder, fsep))[9], '\\.'))[3]
            
            data_fn <- paste(model_key, lowering_future, initial_condition, climate_future, pln, 'surrogate_samples', 'dat', sep='.')
            data_fp <- paste(cli_folder,  'surrogates', pln, data_fn, sep=fsep)
            
            if (file.exists(data_fp)){
              print(cli_folder)
              # add data fp to list
              all_data_fps[[length(all_data_fps)+1]] <- data_fp
            } # end if sample exists. 
            
          } # end climate IC future
          
        }# end model
        
        # summary list
        no_cores <- 3#detectCores() - 1
        cl <- makeCluster(no_cores, type="FORK")
        summary_list <- parLapply(cl, all_data_fps, summarize_results)
        stopCluster(cl)
        
        # save file
        df_summary <- bind_rows(summary_list)
        write.csv(df_summary, file=out_file)
        
      } # end if data file does not exit
      
      
      # first make a plot that shows the mean, and 95% confidence interval 
      img_fp <- file.path(gsub('PARAMETER_UNCERTAINTY', 'INSET_PLOTS', output_dir), paste('loc_pred','map', pln, 'png', sep='.'))
      img = readPNG(img_fp)
      g = rasterGrob(img, interpolate=TRUE)
      
      p <-df_summary %>%
        filter(model_name %in% used_models) %>%
        ggplot(aes(model_time, y=elevation_mean, linetype=climate_future, color=lowering_future)) +
        geom_ribbon(aes(ymin=(modern_surface_elevation - reference_depth ), ymax = modern_surface_elevation), fill='gray', color='black', alpha=.1, size=0.1) +
        geom_ribbon(aes(ymin=elevation_quant_0.025, ymax=elevation_quant_0.975, fill=lowering_future), size=0.3, alpha = 0.25)+
        geom_line() +
        theme_bw(base_size=9) +
        facet_wrap(~model_name) +
        xlab('Model Time (ka)') +
        ylab('Elevation (MCMC mean and 95% CI), (ft)') +
        ggtitle(paste(pln, 'MCMC projections'))
      figure_fp <- file.path(output_dir, paste('uncert', loc, pln, 'pdf', sep='.'))
      
      pdf(file = figure_fp, width=6.5, height=5)
      grid.draw(p)
      img_vp <- viewport(x = 1, y = 0.985, 
                         width = 0.23, height = .25,
                         just = c("right", "top"))
      pushViewport(img_vp)
      grid.draw(g)
      dev.off()      
      
    } # end for plot location
    
    search <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'prediction', loc, 'PARAMETER_UNCERTAINTY', 'summary_table.*.csv')
    
    summary_table_files <- Sys.glob(paths=search) 
    summary_list <- list()
    for(sf in summary_table_files){
      summary <- read.csv(sf, row.names = 1)
      summary_list[[length(summary_list) + 1]] <- summary
    }
    summary_df <-bind_rows(summary_list)
    
    # make plot
    
    # save sd ove time at each locaiton so that it can be used in a panel plot and ID how different parameter 
    # uncertainty estimates are across time and space
    df_save <- summary_df %>% dplyr::select(plot_location, 
                                     model_name, 
                                     climate_future, 
                                     lowering_future, 
                                     initial_condition, 
                                     model_time, 
                                     modern_surface_elevation,
                                     elevation_std, 
                                     elevation_mean, 
                                     elevation_cv)
    
    df_save$model_name <- ordered(df_save$model_name, levels=used_models) # '842', '810',
    out_file <- file.path(table_dir, paste('short_summary_table', 'csv', sep='.'))
    write.csv(df_save, file=out_file)
    df_save <- df_save %>% unite("boundary_condition", c('lowering_future', 'climate_future'), sep='.')
    
    p <- df_save %>%
      filter(model_name %in% used_models) %>%
      ggplot(aes(x=model_time, y=elevation_std, linetype=boundary_condition, color=model_name)) +
      geom_line()+
      theme_bw(base_size=9) +
      facet_wrap(~plot_location) +
      xlab('Model Time (ka)') +
      ylab('Standard deviation of elevation (ft)') +
      ggtitle('MCMC Prediction Standard Deviation')
    
    figure_fp <- file.path(output_dir, paste('std_comparison', 'pdf', sep='.'))
    ggsave(figure_fp,  width=6.5, height=8, units='in')
    
    p <- df_save %>%
      filter(model_name %in% used_models) %>%
      ggplot(aes(x=model_time, y=elevation_mean, linetype=boundary_condition, color=model_name)) +
      geom_ribbon(aes(ymin=(modern_surface_elevation - reference_depth ), ymax = modern_surface_elevation), fill='gray', color='black', alpha=.1, size=0.1) +
      geom_line()+
      theme_bw(base_size=9) +
      facet_wrap(~plot_location) +
      xlab('Model Time (ka)') +
      ylab('Mean of elevation (ft)') +
      ggtitle('MCMC Prediction Mean')
    
    figure_fp <- file.path(output_dir, paste('mean_comparison', 'pdf', sep='.'))
    ggsave(figure_fp,  width=6.5, height=8, units='in')
    
    df_save %>%
      filter(model_name %in% used_models) %>%
      ggplot(aes(x=model_time, y=elevation_cv, linetype=boundary_condition, color=model_name)) +
      geom_line()+
      theme_bw(base_size=9) +
      facet_wrap(~plot_location) +
      xlab('Model Time (ka)') +
      ylab('Coefficient of Variation of elevation (-)') +
      ggtitle('MCMC Prediction Coefficent of Variation (100*std/mean)')
    figure_fp <- file.path(output_dir, paste('cv_comparison', 'pdf', sep='.'))
    ggsave(figure_fp,  width=6.5, height=8, units='in')
    
    
  } # end if location DIR exists
} # end for loction

# stop cluster
