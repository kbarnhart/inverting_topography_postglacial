library(plyr)
library(tidyverse)
library(viridis)
library(ncdf4)
library(png)
library(grid)

fsep = .Platform$file.sep
reference_depth = 50

# create evaluations structure that ids the summary statistics saved 
evaluations = funs(elevation_mean = mean(.), 
                   elevation_std = sd(.),
                   elevation_cv = (sd(.)/mean(.)*100.0))

vals = sort(c(seq(0, 100, 1), c(2.5, 97.5))/100) # this dataset only has 100 points per file

for(v in vals){
  name = paste('elevation_quant_', v, sep='')
  evaluations[[name]] = quo(quantile(., probs=UQE(v)))
}


# for each location
for (loc in c('gully', 'sew')) {
  fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction', loc, 'IC_UNCERTAINTY')
  
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
    
    # read in data 
    data_file <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction', loc, 'IC_UNCERTAINTY', 'compilation_of_sew_IC_uncert_output.csv')
    df_temp <- read.csv(data_file)
    
    data_columns = colnames(df_temp)[which(endsWith(colnames(df_temp), '.0'))[1]: length(colnames(df_temp))]
    
    # gather and separate values
    df <- df_temp %>% gather(key, elevation, data_columns) %>%
                      separate(key, into=c('plot_location', 'model_time'), convert=TRUE, sep='\\.') %>%
                      mutate(model_time = model_time/10.0) %>%
                      dplyr::select(-X)
    
    for(pln in plot_locations){
      print(pln)
      out_file <- file.path(table_dir, paste('summary_table', loc, pln, 'csv', sep='.'))
      
      if(file.exists(out_file)){
        sum <- read.csv(out_file)
      }else{
        
      # create a df that has summary statistics
      sum <- df %>% filter(plot_location == pln) %>%
                    group_by(model_name, climate_future, lowering_future, initial_condition, plot_location, model_time) %>%
                    summarise_all(evaluations)
      # here we can't get surface elevation b/c noise was added. get it from the Param Uncert Summary Table
      param_table_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'prediction', loc, 'PARAMETER_UNCERTAINTY', paste('summary_table', loc, pln, 'csv', sep='.'))
      param_df <- read.csv(param_table_fp)
      
      # add a surface elevation  
      sum$modern_surface_elevation <- mean(param_df$modern_surface_elevation)
       
      # save file
      write.csv(sum, file=out_file)
      }
        
      img_fp <- file.path(gsub('IC_UNCERTAINTY', 'INSET_PLOTS', output_dir), paste('loc_pred','map', pln, 'png', sep='.'))
      img = readPNG(img_fp)
      g = rasterGrob(img, interpolate=TRUE)
      
      # first make a plot that shows the mean, and 95% confidence interval 
      p <- ggplot(sum, aes(model_time, y=elevation_mean, linetype=climate_future, color=lowering_future)) +
        geom_ribbon(aes(ymin=(modern_surface_elevation - reference_depth ), ymax = modern_surface_elevation), color='gray', alpha=.1, size=0.1) +
        geom_ribbon(aes(ymin=elevation_quant_0.025, ymax=elevation_quant_0.975, fill=lowering_future), size=0.3, alpha = 0.25)+
        geom_line() +
        theme_bw(base_size=9) +
        facet_wrap(~model_name) +
        xlab('Model Time (ka)') +
        ylab('Elevation (IC mean and 95% CI), (ft)') +
        ggtitle(paste(loc, pln, 'IC predictions'))
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
    
    search <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'prediction', loc, 'IC_UNCERTAINTY', 'summary_table.*.csv')
    
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
    df_save <- summary_df %>% select(plot_location, 
                                     model_name, 
                                     climate_future, 
                                     lowering_future, 
                                     initial_condition, 
                                     model_time, 
                                     modern_surface_elevation,
                                     elevation_std, 
                                     elevation_mean, 
                                     elevation_cv)
    
    df_save$model_name <- ordered(df_save$model_name, levels=c("800", "802", "804", "808", "810", "840", "842", "A00", "C00"))
    out_file <- file.path(table_dir, paste('short_summary_table', 'csv', sep='.'))
    write.csv(df_save, file=out_file)
    df_save <- df_save %>% unite("boundary_condition", c('lowering_future', 'climate_future'), sep='.')
    
    p <- ggplot(df_save, aes(x=model_time, y=elevation_std, linetype=boundary_condition, color=model_name)) +
      geom_line()+
      theme_bw(base_size=9) +
      facet_wrap(~plot_location) +
      xlab('Model Time (ka)') +
      ylab('Standard deviation of elevation (ft)') +
      ggtitle('IC Prediction Standard Deviation')
    
    figure_fp <- file.path(output_dir, paste('std_comparison', 'pdf', sep='.'))
    ggsave(figure_fp, width=6.5, height=5, units='in')
    
    p <- ggplot(df_save, aes(x=model_time, y=elevation_mean, linetype=boundary_condition, color=model_name)) +
      geom_ribbon(aes(ymin=(modern_surface_elevation - reference_depth ), ymax = modern_surface_elevation), color='gray', alpha=.1, size=0.1) +
      geom_line()+
      theme_bw(base_size=9) +
      facet_wrap(~plot_location) +
      xlab('Model Time (ka)') +
      ylab('Mean of elevation (ft)') +
      ggtitle('IC Prediction Mean')
    
    figure_fp <- file.path(output_dir, paste('mean_comparison', 'pdf', sep='.'))
    ggsave(figure_fp, width=6.5, height=5, units='in')
    
    p <- ggplot(df_save, aes(x=model_time, y=elevation_cv, linetype=boundary_condition, color=model_name)) +
      geom_line()+
      theme_bw(base_size=9) +
      facet_wrap(~plot_location) +
      xlab('Model Time (ka)') +
      ylab('Coefficient of Variation of elevation (-)') +
      ggtitle('IC Prediction Coefficent of Variation (100*std/mean)')
    figure_fp <- file.path(output_dir, paste('cv_comparison', 'pdf', sep='.'))
    ggsave(figure_fp, width=6.5, height=5, units='in')
    
  } # end if location DIR exists
} # end for loction

