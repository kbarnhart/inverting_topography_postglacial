library(plyr)
library(tidyverse)
library(viridis)
library(ncdf4)
library(png)
library(grid)
fsep = .Platform$file.sep
reference_depth = 50

# for each location
for (loc in c('gully', 'sew')) {
  fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction', loc, 'BREACHING')
  
  if (file.exists(fp)){ # if location DIR directory exits. 
    
    output_dir <- gsub('study3py', paste('study3py', 'result_figures', sep=fsep), fp) 
    if(dir.exists(output_dir) == FALSE){
      dir.create(output_dir, recursive=TRUE)
    } # end if dir doesn't exist
    
    # table_dir <- gsub('study3py', paste('study3py', 'result_tables', sep=fsep), fp) 
    # if(dir.exists(table_dir) == FALSE){
    #   dir.create(table_dir, recursive=TRUE)
    # } # end
    
    # plot locations given as column, row (e.g. x, y)
    point_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction', loc, 'PredictionPoints_ShortList.csv')
    plot_loc_df <- read.csv(point_fp)
    
    plot_locations <- plot_loc_df$Point_Name
    
    # read in data 
    data_file <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction', loc, 'BREACHING', 'compilation_of_sew_breaching_uncert_output.csv')
    df_temp <- read.csv(data_file)
    
    data_columns = colnames(df_temp)[which(endsWith(colnames(df_temp), '.0'))[1]: length(colnames(df_temp))]
    
    # gather and separate values
    df <- df_temp %>% gather(key, elevation, data_columns) %>%
                      separate(key, into=c('plot_location', 'model_time'), convert=TRUE, sep='\\.') %>%
                      mutate(model_time = model_time/10.0) %>%
                      dplyr::select(-X)
    df$capture_start_time <- ordered(df$capture_start_time)
    for(pln in plot_locations){
      print(pln)
      for(bl in unique(df$breach_location)){
        
      
        # create a df that has summary statistics
        df_pln <- df %>% filter(plot_location == pln, breach_location == bl) %>%
                         group_by(model_name, climate_future, lowering_future, initial_condition, plot_location, model_time) %>%
                         mutate(Scenario=paste(climate_future, lowering_future, sep=' & '))
        
        modern_surface_elevation <- df_pln %>% filter(model_time == 0) %>% ungroup %>% summarise(modern_surface_elevation = mean(elevation))
        # add a surface elevation  
        df_pln$modern_surface_elevation <- modern_surface_elevation$modern_surface_elevation[1]
         
        # save file
        #out_file <- file.path(table_dir, paste('summary_table', loc, pln, 'csv', sep='.'))
        #write.csv(sum, file=out_file)
        
        # first make a plot that shows the mean, and 95% confidence interval 
        img_fp <- file.path(gsub('BREACHING', 'INSET_PLOTS', output_dir), paste('loc_pred','map', pln, 'png', sep='.'))
        img = readPNG(img_fp)
        g = rasterGrob(img, interpolate=TRUE)
        
        p <- ggplot(df_pln, aes(model_time, y=elevation, linetype=Scenario, color=capture_start_time)) +
          geom_ribbon(aes(ymin=(modern_surface_elevation - reference_depth ), ymax = modern_surface_elevation), color='gray', alpha=.1, size=0.1) +
          geom_line() +
          theme_bw(base_size=9) +
          facet_wrap(~model_name) +
          xlab('Model Time (ka)') +
          ylab('Elevation (ft)') +
          ggtitle(paste(pln, bl, 'capture scenario'))
        figure_fp <- file.path(output_dir, paste('uncert', loc, bl, pln, 'pdf', sep='.'))
        
        pdf(file = figure_fp, width=6.5, height=5)
        grid.draw(p)
        img_vp <- viewport(x = 1, y = 0.985, 
                           width = 0.23, height = .25,
                           just = c("right", "top"))
        pushViewport(img_vp)
        grid.draw(g)
        dev.off()
      } # end breach location
    } # end for plot location
    
    
  } # end if location DIR exists
} # end for loction

