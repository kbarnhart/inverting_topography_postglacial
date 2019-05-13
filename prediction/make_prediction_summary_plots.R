library(plyr)
library(viridis)
library(latex2exp)
library(png)
library(tidyverse)
library(gridExtra)
library(grid)
library(stargazer)

fsep = .Platform$file.sep
reference_depth = 50
used_models_for_param_uncert = c('800', '802', '804', '808', '842', 'A00', 'C00')

# for each location
for (loc in c('gully', 'sew')) {
  fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction', loc, 'BEST_PARAMETERS')
  
  if (file.exists(fp)){ # if location DIR directory exits. 
    
    # model summary file from EGO2
    ego2_sum_fp <-file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'calibration', loc, paste('ego2.', loc, '.calibration.summary.csv', sep=''))
    ego_df <- read.csv(ego2_sum_fp) %>% filter(!(X %in% c('model_840_160_evals', 'model_842_larger_range'))) %>% 
      separate(X, into=c('temp', 'model_name'), convert=TRUE, sep='_') %>% 
      select(-temp) %>% drop_na()
    
    
    # plot locations given as column, row (e.g. x, y)
    point_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction', loc, 'PredictionPoints_ShortList.csv')
    plot_loc_df <- read.csv(point_fp)
    
    xy <- cbind(plot_loc_df$Column_number+1, plot_loc_df$Row_number+1)
    plns <- plot_loc_df$Point_Name
    
    output_dir <- gsub('study3py', paste('study3py', 'result_figures', sep=fsep), fp) 
    if(dir.exists(output_dir) == FALSE){
      dir.create(output_dir, recursive=TRUE)
    } # end if dir doesn't exist
    
    table_dir <- gsub('study3py', paste('study3py', 'result_tables', sep=fsep), fp) 
    if(dir.exists(table_dir) == FALSE){
      dir.create(table_dir, recursive=TRUE)
    } # end
    
    results_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction', paste('plot_locations','predictions', loc,'csv', sep='.'))
    
    if(file.exists(results_fp)){
      result_df <- read.csv(results_fp, row.names = 1)
    }else{
      
      result_df <- data.frame(plot_location = character(),
                              model_name = character(),
                              lowering_future = character(),
                              initial_condition = character(),
                              climate_future = character(), 
                              model_time = numeric(),  
                              elevation = numeric())
      
      # for each model
      for(model_folder in dir(fp, pattern = 'model_*', full.names = TRUE)) {
        #print(model_folder)
        
        model_name = unlist(strsplit(unlist(strsplit(model_folder, fsep))[8], '_'))[2]
        
        # get climage/lowering/IC future
        for(cli_folder in dir(model_folder, pattern = '*', full.names = TRUE)) {
          
          print(cli_folder)
          
          lowering_future = unlist(strsplit(unlist(strsplit(cli_folder, fsep))[9], '\\.'))[1]
          initial_condition = unlist(strsplit(unlist(strsplit(cli_folder, fsep))[9], '\\.'))[2]
          climate_future = unlist(strsplit(unlist(strsplit(cli_folder, fsep))[9], '\\.'))[3]
          
          for(nc_file in dir(cli_folder, pattern = "*.nc", full.names = TRUE)){
            model_time = as.numeric(sub('\\.nc$', '', unlist(strsplit(unlist(strsplit(nc_file, fsep))[10], '_'))[3])) / 10.0 # convert to ka
            
            # get brick
            brk <- brick(nc_file, varname='topographic__elevation') # ncdf
            
            # extract values
            vals <- extract(brk, xy)
            
            # create a structure to save
            save_df = data.frame('plot_location' = plns,
                                 'model_name' = model_name,
                                 'lowering_future' = lowering_future,
                                 'initial_condition' = initial_condition,
                                 'climate_future' = climate_future,
                                 'model_time' = model_time,
                                 'elevation' = unname(vals))
            
            result_df <- rbind(result_df, save_df)
            
          } # end NC file
          
        } # end climate/lowering/IC
      } #end model
      write.csv(result_df, file=results_fp)
    } # end if file exists
    
    if(dim(result_df)[1] > 0){
      
      #%% open each of the parameter_uncertainty summary tables
      search <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'prediction', loc, 'PARAMETER_UNCERTAINTY', 'summary_table.*.csv')
      
      param_summary_table_files <- Sys.glob(paths=search) 
      param_summary_list <- list()
      for(sf in param_summary_table_files){
        summary <- read.csv(sf, row.names = 1)
        param_summary_list[[length(param_summary_list) + 1]] <- summary
      }
      
      param_uncert_df <-bind_rows(param_summary_list)
      param_uncert_df$model_name <- factor(param_uncert_df$model_name)
      param_uncert_df$model_time <- as.numeric(param_uncert_df$model_time)
      param_uncert_df$plot_location <- factor(param_uncert_df$plot_location)
      param_uncert_df$climate_future <- factor(param_uncert_df$climate_future)
      param_uncert_df$lowering_future <- factor(param_uncert_df$lowering_future)
      param_uncert_df$initial_condition <- factor(param_uncert_df$initial_condition)
      param_uncert_df_t <- param_uncert_df %>% rename_at(vars(-plot_location, -model_name, -lowering_future, -initial_condition, -climate_future, -model_time), funs(paste0("param_", .)))
      
      # open each of the IC uncertainty summary tables
      search <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables',  'prediction', loc, 'IC_UNCERTAINTY', 'summary_table.*.csv')
      
      ic_uncert_summary_table_files <- Sys.glob(paths=search) 
      ic_uncert_summary_list <- list()
      for(sf in ic_uncert_summary_table_files){
        summary <- read.csv(sf, row.names = 1)
        ic_uncert_summary_list[[length(ic_uncert_summary_list) + 1]] <- summary
      }
      
      ic_uncert_df <-bind_rows(ic_uncert_summary_list)
      ic_uncert_df$model_name <- factor(ic_uncert_df$model_name)
      ic_uncert_df$model_time <- as.numeric(ic_uncert_df$model_time)
      ic_uncert_df$plot_location <- factor(ic_uncert_df$plot_location)
      ic_uncert_df$climate_future <- factor(ic_uncert_df$climate_future)
      ic_uncert_df$lowering_future <- factor(ic_uncert_df$lowering_future)
      ic_uncert_df$initial_condition <- factor(ic_uncert_df$initial_condition)
      ic_uncert_df_t <- ic_uncert_df %>% rename_at(vars(-plot_location, -model_name, -lowering_future, -initial_condition, -climate_future, -model_time), funs(paste0("ic_", .)))
      
      modern_surface_elevation <- ic_uncert_df_t %>%
        select(plot_location, ic_modern_surface_elevation) %>%
        group_by(plot_location) %>%
        summarise(modern_surface_elevation = mean(ic_modern_surface_elevation, na.omit=TRUE))
        
      # combine result_df and param uncertanty df and the 
      join_vars = c("plot_location", "model_name", "lowering_future", "initial_condition", "climate_future", "model_time")
      full_df <- left_join(left_join(result_df, ic_uncert_df_t, by=join_vars), param_uncert_df_t, by=join_vars) %>%
        select(-ends_with('modern_surface_elevation')) %>%
        left_join(., modern_surface_elevation, by=c('plot_location'))

              
      
      # remove St variants as they were mistakenly plotted
      df_plot <- filter(full_df, !(model_name %in% c("100", "102", "104", "108", "110", '300')))
      
      plot_sets = list('all800s' = c("800", "802", "804", "808", "810", "840", "842", "A00", "C00"), 
                       #'all' = levels(df_plot$model_name),
                       'three800s' = c("800", "802", "842"))
      
      # loop through locations, and plot. 
      for(row in 1:nrow(plot_loc_df)){
        pln <- plot_loc_df$Point_Name[row]
        print(as.character(pln))
        
        
        img_fp <- file.path(gsub('BEST_PARAMETERS', 'INSET_PLOTS', output_dir), paste('loc_pred','map', pln, 'png', sep='.'))
        img = readPNG(img_fp)
        g = rasterGrob(img, interpolate=TRUE)

        # plot each set in a different plot
        for(psn in names(plot_sets)){
          
          set = unname(plot_sets[[psn]])
          
          p <- df_plot %>% filter((plot_location == pln) & (model_name %in% set)) %>%
            ggplot(aes(x=model_time, y=elevation, color=lowering_future, linetype=climate_future)) +
            geom_ribbon(aes(ymax = modern_surface_elevation, ymin=(modern_surface_elevation - reference_depth)), fill='gray', color='black', alpha=.1, size=0.1) +
            geom_line() +
            theme_bw(base_size=9) +
            facet_wrap(~model_name) +
            xlab('Model Time (ka)') +
            ylab('Elevaion (ft)') +
            ggtitle(paste(loc, pln, 'Expected Parameter Set Predictions'))
          figure_fp <- file.path(output_dir, paste('loc_pred', 'expected_means', loc, pln, psn, 'pdf', sep='.'))
         
          pdf(file = figure_fp, width=6.5, height=5)
          grid.draw(p)
          img_vp <- viewport(x = 1, y = 0.985, 
                             width = 0.23, height = .25,
                             just = c("right", "top"))
          pushViewport(img_vp)
          grid.draw(g)
          dev.off()
          
          p <- df_plot %>% filter((plot_location == pln) & (model_name %in% set) & (model_name %in% used_models_for_param_uncert)) %>% 
            ggplot(aes(x=model_time, y=elevation, color=lowering_future, linetype=climate_future)) +
            geom_ribbon(aes(ymax = modern_surface_elevation, ymin=(modern_surface_elevation - reference_depth)), fill='gray', color='black', alpha=.1, size=0.1) +
            geom_ribbon(aes(ymin=param_elevation_quant_0.025, ymax=param_elevation_quant_0.975, fill=lowering_future), size=0.1, alpha = 0.25) +
            geom_line() +
            theme_bw(base_size=9) +
            facet_wrap(~model_name) +
            xlab('Model Time (ka)') +
            ylab('Elevaion (ft)') +
            ggtitle(paste(loc, pln, 'Calibration Parameter Set Uncertainty'))
          figure_fp <- file.path(output_dir, paste('loc_pred', 'param_uncert', loc, pln, psn, 'pdf', sep='.'))
          
          pdf(file = figure_fp, width=6.5, height=5)
          grid.draw(p)
          img_vp <- viewport(x = 1, y = 0.985, 
                             width = 0.23, height = .25,
                             just = c("right", "top"))
          pushViewport(img_vp)
          grid.draw(g)
          dev.off()
          
          p <- df_plot %>% filter((plot_location == pln) & (model_name %in% set)) %>% 
            ggplot(aes(x=model_time, y=elevation, color=lowering_future, linetype=climate_future)) +
            geom_ribbon(aes(ymax = modern_surface_elevation, ymin=(modern_surface_elevation - reference_depth)), fill='gray', color='black', alpha=.1, size=0.1) +
            geom_ribbon(aes(ymin=ic_elevation_quant_0.025, ymax=ic_elevation_quant_0.975, fill=lowering_future), size=0.1, alpha = 0.25) +
            geom_line() +
            theme_bw(base_size=9) +
            facet_wrap(~model_name) +
            xlab('Model Time (ka)') +
            ylab('Elevation (ft)') +
            ggtitle(paste(loc, pln, 'IC Uncertainty'))
          figure_fp <- file.path(output_dir, paste('loc_pred', 'ic_uncert', loc, pln, psn, 'pdf', sep='.'))
          
          pdf(file = figure_fp, width=6.5, height=5)
          grid.draw(p)
          img_vp <- viewport(x = 1, y = 0.985, 
                             width = 0.23, height = .25,
                             just = c("right", "top"))
          pushViewport(img_vp)
          grid.draw(g)
          dev.off()
          
          
        } # for each set
      } # end plot locations
    } # end if restults list>0
    
    
    { # begin combined uncertainty analysis.
      
      
      # set colors explicity
      colors = c("Black", "Black", 'Black', "#FFC107","#E53935", '#FF5722', "#AED581", "#FB8C00", "#9C27B0",  '#81D4FA','#9FA8DA','#BCAAA4', '#78909C')
      key <- data.frame(source=c("combined_std",
                                 'independent_combined_std',
                                 'covary_combined_std',
                                 'model_std',
                                 "calibration_std", 
                                 "model_calibration_std",
                                 "ic_std", 
                                 "climate_std", 
                                 "lowering_std", 
                                 "climate_lowering_interaction",
                                 'lowering_model_interaction',
                                 'climate_model_interaction',
                                 'model_lowering_climate_interaction'),
                        color=colors,
                        type=c('combined' , 'combined' , 'combined' , 'std', 'std', 'std', 'std', 'std', 'std', 'interaction','interaction', 'interaction', 'interaction'),
                        full_name = c('Total Uncertainty' , 
                                      'Total Uncertainty\n(Model and Calibration Independent)',
                                      'Total Uncertainty\n(Model and Calibration Covary)',
                                      'Model Selection', 'Model Calibration', 
                                      'Model Selection\nand Calibration',
                                      'Initial Condition', 'Climate Future',  'Lowering Future', 
                                      'Lowering-Climate\nInteraction', 'Model-Lowering\nInteraction', 'Model-Climate\nInteraction', 
                                      'Model-Lowering-Climate\nInteraction'))
      
      key <- key %>% mutate(scaled_source = paste('scaled_', source, sep=''))
      key$full_name <- ordered(key$full_name, levels=unique(key$full_name))
      names(colors) <- as.character(key$full_name)
      
      
      
    # compare MCMC mean and best parameter set values. 
    partial_df <- left_join(param_uncert_df,result_df, c("plot_location", 
                                                         "model_name", 
                                                         "lowering_future", 
                                                         "initial_condition", 
                                                         "climate_future", 
                                                         "model_time"))
    
    # mean/best comparision 
    p <- partial_df %>% filter(model_name %in% c("800", "802", "842")) %>%
      dplyr::select(model_time, model_name, plot_location, lowering_future, climate_future, modern_surface_elevation, elevation, elevation_mean) %>%
      rename(calibrated_parameter_set=elevation, mcmc_mean=elevation_mean) %>%
      unite("boundary_condition", c('lowering_future', 'climate_future'), sep='.') %>%
      gather(key='method', value='elevation', calibrated_parameter_set:mcmc_mean) %>%
      ggplot(aes(x=model_time, y=elevation, color=model_name, linetype=method)) +
      geom_ribbon(aes(ymin=(modern_surface_elevation - reference_depth), ymax = modern_surface_elevation), fill='gray', color='black', alpha=.1, size=0.1) +
      geom_line() +
      theme_bw(base_size=9) +
      facet_grid(plot_location~boundary_condition) +
      ggtitle('comparison')
    figure_fp <- file.path(output_dir, paste('mean_expected_comparison', 'pdf', sep='.'))
    ggsave(figure_fp, width=6, height=20, units='in')
    
    
    # partition uncertainty out
    model800s = unlist(unname(plot_sets['all800s']))
    
    uncert_data <- filter(full_df, (model_name %in% model800s))
    
    calib_dat <- ego_df %>% filter(model_name %in% model800s) %>%
      select(model_name, AICc) %>%
      mutate(del = AICc - min(AICc),
             probability = exp(-0.5*del)/(sum(exp(-0.5*del))))
    
    calib_dat
    
    # calibration probabiliteis alone say we should only pay attention to model 842, but validation 
    # comparison indicates we should consider other models too. So we;ll do this two ways
    # option 1: just look at 842
    # option 2: look at all 9 and weight them evenly
    
    # option 1
    df_842 <- filter(uncert_data, model_name == '842')
    
    # first, param_uncertainty, and IC uncertainty use STD of the models
    
    overall_mean <- df_842 %>% 
      dplyr::select(plot_location, model_time, model_name, ic_elevation_mean, modern_surface_elevation) %>%
      group_by(plot_location, model_name, model_time) %>%
      summarise(expected_elevation = mean(ic_elevation_mean),
                modern_surface_elevation = mean(modern_surface_elevation))
    
    # by_climate_mean
    climate_mean <- df_842 %>% 
      dplyr::select(plot_location, model_time, model_name, ic_elevation_mean, climate_future) %>%
      group_by(plot_location, model_name, model_time, climate_future) %>%
      summarise(by_climate_mean = mean(ic_elevation_mean))
    
    # by_lowering_mean
    lowering_mean <- df_842 %>% 
      dplyr::select(plot_location, model_time, model_name, ic_elevation_mean, lowering_future) %>%
      group_by(plot_location,  model_name, model_time, lowering_future) %>%
      summarise(by_lowering_mean = mean(ic_elevation_mean))
    
    # add the overall mean to the df_842
    join_vars <- c('plot_location', 'model_time', 'model_name')
    means_df <- left_join(overall_mean, climate_mean, by=join_vars) %>%
      left_join(., lowering_mean, by=join_vars) %>%
      select(plot_location, model_time, model_name, climate_future, lowering_future, 
             expected_elevation, modern_surface_elevation,
             by_lowering_mean, by_climate_mean )
    
    join_vars <- c('plot_location', 'model_name', 'model_time', 'climate_future', 'lowering_future')
    df_842_with_means <- left_join(df_842, means_df, join_vars) %>%
      select(plot_location, model_name, model_time, climate_future, lowering_future,  expected_elevation, ic_elevation_mean,
             by_lowering_mean, by_climate_mean, elevation,
             param_elevation_std, ic_elevation_std)
    
    # first, param_uncertainty, and IC uncertainty use STD of the models
    # these 
    param_ic_uncert <- df_842_with_means %>% 
      select(plot_location, model_time, param_elevation_std, ic_elevation_std) %>%
      drop_na() %>%
      group_by(plot_location, model_time) %>%
      summarise(calibration_std = (mean(param_elevation_std^2))^0.5,
                ic_std = (mean(ic_elevation_std^2))^0.5)
    
    # lowering future variance is the variability of the lowering future means around the
    # ensemble mean. 
    lowering_uncert <- df_842_with_means %>% 
      select(plot_location, model_time, by_lowering_mean, expected_elevation) %>%
      distinct() %>%
      mutate(sq_residual = (by_lowering_mean-expected_elevation)^2) %>%
      group_by(plot_location, model_time) %>%
      summarise(lowering_std = (mean(sq_residual))^0.5)
    
    # climate future variance is the variability of the climate future means around the
    # ensemble mean. 
    climate_uncert <- df_842_with_means %>% 
      dplyr::select(plot_location, model_time, by_climate_mean, expected_elevation) %>%
      mutate(sq_residual = (by_climate_mean-expected_elevation)^2) %>%
      group_by(plot_location, model_time) %>%
      summarise(climate_std = (mean(sq_residual))^0.5)
    
    # climate future and lowering future interactions
    climate_lowering_interaction = df_842_with_means %>%
      dplyr::select(plot_location, model_time, climate_future, lowering_future, by_climate_mean, by_lowering_mean, expected_elevation, ic_elevation_mean) %>%
      group_by(plot_location, model_time, climate_future, lowering_future) %>%
      summarise(cross_model_mean = mean(ic_elevation_mean),
                by_lowering_mean2 = mean(by_lowering_mean),
                by_climate_mean2 = mean(by_climate_mean),
                expected_elevation2 = mean(expected_elevation)) %>%
      mutate(sq_residual = (cross_model_mean+expected_elevation2-by_climate_mean2-by_lowering_mean2)^2) %>%
      group_by(plot_location, model_time) %>%
      summarise(climate_lowering_interaction = (mean(sq_residual))^0.5)
    
    # combine the dataframes and gather
    join_vars <- c("plot_location", "model_time")
    uncert_df_842 <- left_join(left_join(left_join(means_df, param_ic_uncert, by = join_vars), 
                                         left_join(climate_uncert, lowering_uncert, by = join_vars),
                                         by=join_vars), climate_lowering_interaction, by=join_vars) %>%
      mutate(combined_std = (climate_std^2 +
                               lowering_std^2 + 
                               calibration_std^2 + 
                               ic_std^2 + 
                               climate_lowering_interaction^2)^0.5) 
    
    
    uncert_df_842_g <- uncert_df_842 %>%
      gather(source, uncertainty_metric, -model_time, -plot_location, -model_name,
             -climate_future, -lowering_future, 
             -expected_elevation, -modern_surface_elevation,
             -by_lowering_mean, -by_climate_mean)
    
    
    # make a plot of uncertanty through time
    p <- uncert_df_842_g %>% 
      left_join(., key, by='source') %>% 
      ggplot(aes(x=model_time, y=uncertainty_metric, color=full_name)) + 
      geom_line() + 
      theme_bw(base_size=9) + 
      scale_color_manual(name = 'Source of Uncertainty', values = colors) +
      facet_wrap(~plot_location, nrow=5)+
      xlab('Model Time (ka)') +
      ylab('Standard Deviation (ft)') +
      ggtitle('Components of Uncertainty Through Time')
    figure_fp <- file.path(output_dir, paste('uncertainty_summary.842_only', loc, 'pdf', sep='.'))
    ggsave(figure_fp,  width=6.5, height=8, units='in')
    
    
    # stack and scale, and make a stacked uncertainty plot
    p <- uncert_df_842 %>% 
      select(-expected_elevation, -starts_with('by'), -ends_with('future')) %>%
      mutate(scaled_climate_std = combined_std*(climate_std^2/combined_std^2),
             scaled_ic_std = combined_std*(ic_std^2/combined_std^2),
             scaled_lowering_std = combined_std*(lowering_std^2/combined_std^2),
             scaled_calibration_std = combined_std*(calibration_std^2/combined_std^2),
             scaled_climate_lowering_interaction = combined_std*(climate_lowering_interaction^2/combined_std^2)) %>%
      ungroup() %>%
      select(plot_location, model_time, starts_with('scaled_')) %>% distinct() %>%
      gather(scaled_source, scaled_uncertainty_metric, starts_with('scaled_'), -model_time, -plot_location) %>% 
      left_join(., key, by='scaled_source') %>% 
      ggplot(aes(x=model_time, y=scaled_uncertainty_metric, fill=full_name)) + 
      geom_area(position = 'fill')+
      scale_fill_manual(name = 'Source of Uncertainty', values = colors) +
      theme_bw(base_size=9)+
      xlab('Model Time (ka)') +
      ylab('Proportion of Uncertainty (-)') +
      ggtitle('Proportion of Uncertainty Through Time') +
      facet_wrap(~plot_location, nrow=5)
    figure_fp <- file.path(output_dir, paste('uncertainty_scaled.842_only', loc, 'pdf', sep='.'))
    ggsave(figure_fp,  width=6.5, height=8, units='in')
    
    
    # next option 2 Consider all models equally, do this two ways, one in which model and calibration are independent
    # and one in which they covary. 
    generic_summarise <- function(df, output_name, expr, ...) {
      group_var <- quos(...)
      name <- quo_name(output_name)
      expr <- enquo(expr)
      df %>%
        group_by(!!!group_var) %>%
        summarise(!!name := mean(!!expr))
    }
    
    # construct dataframes that represent each of the means used to construct the alpha-kappa estimators
    overall_mean <- generic_summarise(df=uncert_data, output_name="expected_elevation", expr=ic_elevation_mean, plot_location, model_time)
    modern_elevation <- generic_summarise(df=uncert_data, output_name="modern_surface_elevation", expr=modern_surface_elevation, plot_location, model_time)
    climate_mean <- generic_summarise(df=uncert_data, output_name="by_climate_mean", expr=ic_elevation_mean, plot_location, model_time, climate_future)
    model_mean <- generic_summarise(df=uncert_data, output_name="by_model_mean", expr=ic_elevation_mean, plot_location, model_time, model_name)
    lowering_mean <- generic_summarise(df=uncert_data, output_name="by_lowering_mean", expr=ic_elevation_mean, plot_location, model_time, lowering_future)
    model_climate_mean <- generic_summarise(df=uncert_data, output_name="by_model_climate_mean", expr=ic_elevation_mean, plot_location, model_time, climate_future, model_name)
    model_lowering_mean <- generic_summarise(df=uncert_data, output_name="by_model_lowering_mean", expr=ic_elevation_mean, plot_location, model_time, lowering_future, model_name)
    climate_lowering_mean  <- generic_summarise(df=uncert_data, output_name="by_climate_lowering_mean", expr=ic_elevation_mean, plot_location, model_time, climate_future, lowering_future)
    model_climate_lowering_mean <- generic_summarise(df=uncert_data, output_name = 'by_model_climate_lowering_mean', expr=ic_elevation_mean, plot_location, model_time, model_name, climate_future, lowering_future)
    
    # there are many NAs in the param_elevation_std. We want to use the by-model mean of the uncertainty
    # we only have 7 of 9 models sucessfull, so we'll use those and use the geom_avg of them for 810 and 840
    
    # set those models we are not using to NA, 810 completed but shows signs of instability in the surrogate
    # 840 did not complete
    
    uncert_data$param_elevation_std[!(uncert_data$model_name %in% used_models_for_param_uncert)] = NA
    temp_param_std <- uncert_data %>% 
      dplyr::select(plot_location, model_time, model_name, param_elevation_std) %>%
      group_by(plot_location,  model_time) %>%
      summarise(temp_param_std = mean((param_elevation_std)^2, na.rm=TRUE)^0.5)
    
    param_std <- uncert_data %>% 
      dplyr::select(plot_location, model_time, model_name, param_elevation_std) %>%
      group_by(plot_location,  model_name, model_time) %>%
      summarise(param_std = mean(param_elevation_std^2, na.rm=TRUE)^0.5) %>%
      left_join(., temp_param_std, by=c('model_time', 'plot_location')) %>%
      mutate(param_std = case_when(is.na(param_std)==TRUE ~ temp_param_std,
                                   is.na(param_std)==FALSE ~ param_std)) %>%
      select(-temp_param_std)
    
    # add the overall mean to the uncert_data
    means_df <- overall_mean %>% 
      left_join(., modern_elevation, by=c('plot_location', 'model_time')) %>%
      left_join(., climate_mean, by=c('plot_location', 'model_time')) %>%
      left_join(., lowering_mean, by=c('plot_location', 'model_time')) %>%
      left_join(., model_mean, by=c('plot_location', 'model_time')) %>%
      left_join(., param_std, by=c('plot_location', 'model_time', 'model_name')) %>%
      left_join(., model_climate_mean, by=c('plot_location', 'model_time', 'model_name', 'climate_future')) %>%
      left_join(., model_lowering_mean, by=c('plot_location', 'model_time', 'model_name', 'lowering_future')) %>%
      left_join(., climate_lowering_mean, by=c('plot_location', 'model_time', 'lowering_future', 'climate_future')) %>%
      left_join(., model_climate_lowering_mean, by=c('plot_location', 'model_time', 'model_name', 'lowering_future', 'climate_future')) %>%
      select(plot_location, model_time, model_name, climate_future, lowering_future, 
             modern_surface_elevation,
             expected_elevation, param_std,
             by_lowering_mean, by_climate_mean, by_model_mean,
             by_climate_lowering_mean, by_model_climate_mean, by_model_lowering_mean,
             by_model_climate_lowering_mean)
    
    uncert_data_with_means <- uncert_data %>% select(-modern_surface_elevation) %>%
      left_join(., means_df, by = c('plot_location', 'model_time', 'model_name', 'climate_future', 'lowering_future')) %>%
      select(plot_location, model_name, model_time, climate_future, lowering_future,  
             expected_elevation, modern_surface_elevation, 
             elevation, 
             by_lowering_mean, by_climate_mean, by_model_mean, 
             by_climate_lowering_mean, by_model_climate_mean, by_model_lowering_mean,
             by_model_climate_lowering_mean,
             param_std, param_elevation_std, ic_elevation_std)

    # first, param_uncertainty, and IC uncertainty use STD of the models
    # here we will consider param uncerrtainty and model uncertainty both independently and together 
    
    param_ic_uncert <- uncert_data_with_means %>% 
      dplyr::select(plot_location, model_time, param_elevation_std, ic_elevation_std) %>%
      drop_na() %>%
      group_by(plot_location, model_time) %>% 
      summarise(calibration_std = mean(param_elevation_std^2)^0.5,
                ic_std = mean(ic_elevation_std^2)^0.5)
    
    # model variance is the variability of the model means around the
    # ensemble mean. 
    # we will consider this with and without the param uncertainty so we can compare the difference of 
    # considering them independently and togehter
    #   
    
    # since the difference between independent and not independent consideration of model and parameter calibration
    # ends ub being the same, we double check that this makes sense. it does. 
    # (ggplot(uncert_data_with_means, aes(param_std^2, (by_model_mean-expected_elevation)^2, color=model_name)) + geom_point() + facet_wrap(~plot_location, ncol=5))

    model_uncert <- uncert_data_with_means %>% 
      select(plot_location, model_time, model_name, by_model_mean, expected_elevation, param_std) %>%
      distinct() %>%
      mutate(sq_residual = (by_model_mean-expected_elevation)^2,
             sq_resid_with_param = param_std^2 + (by_model_mean-expected_elevation)^2) %>% 
      group_by(plot_location, model_time) %>%
      summarise(model_std = (mean(sq_residual))^0.5,
                model_calibration_std = (mean(sq_resid_with_param))^0.5)
    
    # lowering future variance is the variability of the lowering future means around the
    # ensemble mean. 
    lowering_uncert <- uncert_data_with_means %>% 
      select(plot_location, model_time, model_name, by_lowering_mean, expected_elevation) %>%
      distinct() %>%
      mutate(sq_residual = (by_lowering_mean-expected_elevation)^2) %>%
      group_by(plot_location, model_time) %>%
      summarise(lowering_std = (mean(sq_residual))^0.5)
    
    # climate future variance is the variability of the climate future means around the
    # ensemble mean. 
    climate_uncert <- uncert_data_with_means %>% 
      dplyr::select(plot_location, model_time, model_name, by_climate_mean, expected_elevation) %>%
      mutate(sq_residual = (by_climate_mean-expected_elevation)^2) %>%
      group_by(plot_location, model_time) %>%
      summarise(climate_std = (mean(sq_residual))^0.5)
    
    # climate future and lowering future interactions
    climate_lowering_interaction = uncert_data_with_means %>%
      group_by(plot_location, model_time, climate_future, lowering_future) %>%
      mutate(sq_residual = (by_climate_lowering_mean+expected_elevation-by_climate_mean-by_lowering_mean)^2) %>%
      group_by(plot_location, model_time) %>%
      summarise(climate_lowering_interaction = (mean(sq_residual))^0.5)
    
    # climate future and model interactions
    climate_model_interaction = uncert_data_with_means %>%
      group_by(plot_location, model_time, climate_future, model_name) %>%
      mutate(sq_residual = (by_model_climate_mean+expected_elevation-by_climate_mean-by_model_mean)^2) %>%
      group_by(plot_location, model_time) %>%
      summarise(climate_model_interaction = (mean(sq_residual))^0.5) 
    
    # lowering future and model interactions
    lowering_model_interaction = uncert_data_with_means %>%
      group_by(plot_location, model_time, lowering_future, model_name) %>%
      mutate(sq_residual = (by_model_lowering_mean+expected_elevation-by_lowering_mean-by_model_mean)^2) %>%
      group_by(plot_location, model_time) %>%
      summarise(lowering_model_interaction = (mean(sq_residual))^0.5) 
    
    # three_way interactions
    model_lowering_climate_interaction = uncert_data_with_means %>%
      group_by(plot_location, model_time, lowering_future, model_name, lowering_future, climate_future) %>%
      mutate(sq_residual = (by_model_climate_lowering_mean-
                              ((expected_elevation) + 
                              (by_model_lowering_mean + by_model_climate_mean + by_climate_lowering_mean) -
                              (by_lowering_mean+by_model_mean+by_climate_mean)))^2) %>%
      group_by(plot_location, model_time) %>%
      summarise(model_lowering_climate_interaction = (mean(sq_residual))^0.5) 
    
    
    # combine the dataframes and gather
    join_vars <- c("plot_location", "model_time")
    uncert_df <- means_df %>% select(plot_location, model_time, modern_surface_elevation, expected_elevation) %>% distinct() %>%
      left_join(., param_ic_uncert, by = join_vars) %>%
      left_join(., climate_uncert, by = join_vars) %>%
      left_join(., lowering_uncert, by = join_vars) %>%
      left_join(., model_uncert, by = join_vars) %>%
      left_join(., climate_lowering_interaction, by = join_vars) %>%
      left_join(., climate_model_interaction, by = join_vars) %>%
      left_join(., lowering_model_interaction, by = join_vars) %>%
      left_join(., model_lowering_climate_interaction, by=join_vars)%>%
      mutate(independent_combined_std = (climate_std^2 +
                                           lowering_std^2 + 
                                           calibration_std^2 + 
                                           ic_std^2 +
                                           model_std^2 +
                                           climate_lowering_interaction^2 +
                                           climate_model_interaction^2 +
                                           lowering_model_interaction^2+
                                           model_lowering_climate_interaction^2)^0.5,
             covary_combined_std = (climate_std^2 +
                                      lowering_std^2 + 
                                      ic_std^2 +
                                      model_calibration_std^2 +
                                      climate_lowering_interaction^2 +
                                      climate_model_interaction^2 +
                                      lowering_model_interaction^2+
                                      model_lowering_climate_interaction^2)^0.5
      )
    
    
    # make a plot of uncertanty through time (first with independent model and parameters)
    p <- uncert_df %>%
      select(model_time, plot_location, climate_std, lowering_std, ic_std, model_std, calibration_std,
             climate_lowering_interaction, climate_model_interaction, lowering_model_interaction, model_lowering_climate_interaction,
             independent_combined_std) %>% 
      gather(source, uncertainty_metric, -model_time, -plot_location) %>% 
      left_join(., key, by='source') %>% 
      ggplot(aes(x=model_time, y=uncertainty_metric, color=full_name)) + 
      geom_line() + 
      theme_bw(base_size=9) + 
      scale_color_manual(name = 'Source of Uncertainty', values = colors) +
      facet_wrap(~plot_location, nrow=5)+
      xlab('Model Time (ka)') +
      ylab('Standard Deviation (ft)') +
      ggtitle('Components of Uncertainty Through Time')
    figure_fp <- file.path(output_dir, paste('uncertainty_summary.all_independent', loc, 'pdf', sep='.'))
    ggsave(figure_fp,  width=6.5, height=8, units='in')
    
    
    # stack and scale, and make a stacked uncertainty plot (first with independent model and parameters)
    p <- uncert_df %>% select(-expected_elevation, -starts_with('by'), -ends_with('future')) %>%
      mutate(scaled_climate_std = independent_combined_std*(climate_std^2/independent_combined_std^2),
             scaled_model_std = independent_combined_std*(model_std^2/independent_combined_std^2),
             scaled_ic_std = independent_combined_std*(ic_std^2/independent_combined_std^2),
             scaled_lowering_std = independent_combined_std*(lowering_std^2/independent_combined_std^2),
             scaled_calibration_std = independent_combined_std*(calibration_std^2/independent_combined_std^2),
             scaled_climate_lowering_interaction = independent_combined_std*(climate_lowering_interaction^2/independent_combined_std^2),
             scaled_climate_model_interaction = independent_combined_std*(climate_model_interaction^2/independent_combined_std^2),
             scaled_lowering_model_interaction = independent_combined_std*(lowering_model_interaction^2/independent_combined_std^2),
             scaled_model_lowering_climate_interaction = independent_combined_std*(model_lowering_climate_interaction^2/independent_combined_std^2)) %>%
      ungroup() %>%
      select(plot_location, model_time, starts_with('scaled_')) %>% distinct() %>%
      gather(scaled_source, scaled_uncertainty_metric, starts_with('scaled_'), -model_time, -plot_location) %>% 
      left_join(., key, by='scaled_source') %>% 
      ggplot(aes(x=model_time, y=scaled_uncertainty_metric, fill=full_name)) + 
      geom_area(position = 'fill')+
      scale_fill_manual(name = 'Source of Uncertainty', values = colors) +
      theme_bw(base_size=9)+
      xlab('Model Time (ka)') +
      ylab('Proportion of Uncertainty (-)') +
      ggtitle('Proportion of Uncertainty Through Time') +
      facet_wrap(~plot_location, nrow=5)
    figure_fp <- file.path(output_dir, paste('uncertainty_scaled.all_independent', loc, 'pdf', sep='.'))
    ggsave(figure_fp,  width=6.5, height=8, units='in')
    
    # make a plot of uncertanty through time (now with combined model and parameters)
    p <- uncert_df %>%
      select(model_time, plot_location, climate_std, lowering_std, ic_std, model_calibration_std, 
             climate_lowering_interaction, climate_model_interaction, lowering_model_interaction, model_lowering_climate_interaction,
             covary_combined_std) %>% 
      gather(source, uncertainty_metric, -model_time, -plot_location) %>% 
      left_join(., key, by='source') %>% 
      ggplot(aes(x=model_time, y=uncertainty_metric, color=full_name)) + 
      geom_line() + 
      theme_bw(base_size=9) + 
      scale_color_manual(name = 'Source of Uncertainty', values = colors) +
      facet_wrap(~plot_location, nrow=5)+
      xlab('Model Time (ka)') +
      ylab('Standard Deviation (ft)') +
      ggtitle('Components of Uncertainty Through Time')
    figure_fp <- file.path(output_dir, paste('uncertainty_summary.all_covary', loc, 'pdf', sep='.'))
    ggsave(figure_fp,  width=6.5, height=8, units='in')
    
    
    # stack and scale, and make a stacked uncertainty plot (now with combined model and parameters)
    p <- uncert_df %>% select(-expected_elevation, -starts_with('by'), -ends_with('future')) %>%
      mutate(scaled_climate_std = covary_combined_std*(climate_std^2/covary_combined_std^2),
             scaled_model_calibration_std = covary_combined_std*(model_calibration_std^2/covary_combined_std^2),
             scaled_ic_std = covary_combined_std*(ic_std^2/covary_combined_std^2),
             scaled_lowering_std = covary_combined_std*(lowering_std^2/covary_combined_std^2),
             scaled_climate_lowering_interaction = covary_combined_std*(climate_lowering_interaction^2/covary_combined_std^2),
             scaled_climate_model_interaction = covary_combined_std*(climate_model_interaction^2/covary_combined_std^2),
             scaled_lowering_model_interaction = covary_combined_std*(lowering_model_interaction^2/covary_combined_std^2),             
             scaled_model_lowering_climate_interaction = covary_combined_std*(model_lowering_climate_interaction^2/covary_combined_std^2)) %>%

      ungroup() %>%
      select(plot_location, model_time, starts_with('scaled_')) %>% distinct() %>%
      gather(scaled_source, scaled_uncertainty_metric, starts_with('scaled_'), -model_time, -plot_location) %>% 
      left_join(., key, by='scaled_source') %>% 
      ggplot(aes(x=model_time, y=scaled_uncertainty_metric, fill=full_name)) + 
      geom_area(position = 'fill')+
      scale_fill_manual(name = 'Source of Uncertainty', values = colors) +
      theme_bw(base_size=9)+
      xlab('Model Time (ka)') +
      ylab('Proportion of Uncertainty (-)') +
      ggtitle('Proportion of Uncertainty Through Time') +
      facet_wrap(~plot_location, nrow=5)
    figure_fp <- file.path(output_dir, paste('uncertainty_scaled.all_covary', loc, 'pdf', sep='.'))
    ggsave(figure_fp,  width=6.5, height=8, units='in')
    
    
    # finally compare the two methods:
    uncert_df_842$method <- 'Model 842 Only'
    uncert_df_842$model_method <- 'Independent'
    compare_df <- uncert_df %>%
      gather(met, combined_std, covary_combined_std, independent_combined_std) %>%
      mutate(method = case_when(met == 'covary_combined_std' ~ 'Nine Model Average:\nCombined', 
                                met == 'independent_combined_std'~ 'Nine Model Average:\nIndependent'),
             model_method = case_when(met == 'independent_combined_std'~ 'Independent',
                                      met == 'covary_combined_std' ~ 'Combined')) %>%
      select(-met) %>%
      bind_rows(., uncert_df_842)
    
    compare_df$method <- ordered(compare_df$method, levels = c('Model 842 Only', 'Nine Model Average:\nIndependent', 'Nine Model Average:\nCombined'))
    compare_df$model_method <- ordered(compare_df$model_method, levels = c('Independent', 'Combined'))
    colors = c('#1b9e77', '#542788', '#f1a340')
    names(colors) <- levels(compare_df$method)
    
    # write out a csv here to put into the report withe mean and sd at 200, 500, 1000, 10000 for each approach
     
    # select mean and std for correct times, 
    output_df <- compare_df %>%
      ungroup()%>%
      filter(model_time %in% c(0.2, 0.5, 1, 10)) %>%
      filter(method %in% c('Model 842 Only', 'Nine Model Average:\nIndependent')) %>%
      mutate(model_time = model_time*1000,
             expected_erosion = round(modern_surface_elevation - expected_elevation, digits=2),
             combined_std = round(combined_std, digits = 2), 
             method = revalue(method, c("Nine Model Average:\nIndependent" = "All nine 800s")),
             plot_location=as.character(plot_location)) %>%
      select(plot_location, method, model_time, expected_erosion, combined_std) %>%
      distinct() 

    output_elev <- output_df %>%
      mutate(model_time = paste('mean.', model_time, sep='')) %>%
      select(-combined_std) %>%
      spread(key=model_time, value=expected_erosion)
    
    output_std <- output_df %>%
      mutate(model_time = paste('std.', model_time, sep='')) %>%
      select(-expected_erosion) %>%
      spread(key=model_time, value=combined_std)
    
    output_all <- left_join(output_elev, output_std, by=c('plot_location', 'method')) %>%
      select(method, plot_location, mean.200, std.200, mean.500, std.500, mean.1000, std.1000, mean.10000, std.10000) %>%
      ungroup()
    
    write.csv(output_all, file=file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction', loc, 'output_summary_file.csv'))
    # save to latex using stargazer
    #stargazer(output_all[1:10,], rownames=FALSE, summary=FALSE, digits=2, out='prediction_summary_table.tex')
    
    
    p <- compare_df %>%
      select(model_method, method, plot_location, model_time, expected_elevation, modern_surface_elevation, combined_std) %>%
      ggplot(aes(x = model_time, y = expected_elevation, color = method, fill = method, linetype=model_method)) +
      geom_ribbon(aes(ymax = modern_surface_elevation, ymin=(modern_surface_elevation - reference_depth)), fill='gray', color='black', alpha=.1, size=0.1, linetype='solid') +
      geom_ribbon(aes(ymin = (expected_elevation - (2*combined_std)),
                      ymax = (expected_elevation + (2*combined_std))), alpha=0.2) +
      geom_line() +
      theme_bw(base_size=9) +
      scale_linetype_manual(name = 'Model-Calibration\nUncertainty Approach', values=c('Combined'='dashed', 'Independent'='solid')) +
      scale_fill_manual(name = 'Method', values = colors) +
      scale_color_manual(name = 'Method', values = colors) +
      xlab('Model Time (ka)') +
      ylab(TeX('Elevation (multi-model expected value $\\pm 2 \\sigma$, ft)')) +
      ggtitle('Comparison of Three Uncertainty Quantification Methods') +
      facet_wrap(~plot_location, nrow=5)
    figure_fp <- file.path(output_dir, paste('prediction_comparison', loc, 'pdf', sep='.'))
    ggsave(figure_fp,  width=6.5, height=8, units='in')
    
    p <- compare_df %>%
      select(model_method, method, plot_location, model_time, expected_elevation, modern_surface_elevation, combined_std) %>%
      ggplot(aes(x = model_time, y = combined_std, color = method, linetype=model_method)) +
      geom_line() +
      theme_bw(base_size=9) +
      scale_color_manual(name = 'Method', values = colors) +
      scale_linetype_manual(name = 'Model-Calibration\nUncertainty Approach', values=c('Combined'='dashed', 'Independent'='solid')) +
      xlab('Model Time (ka)') +
      ylab(TeX(' Uncertanty in Predicted Elevation ( $1 \\sigma$, ft)')) +
      ggtitle('Comparison of Three Uncertainty Quantification Methods') +
      facet_wrap(~plot_location, nrow=5)
    figure_fp <- file.path(output_dir, paste('uncertainty_comparison', loc, 'pdf', sep='.'))
    ggsave(figure_fp,  width=6.5, height=8, units='in')
    
    for(pln in plns){
      
      img_fp <- file.path(gsub('BEST_PARAMETERS', 'INSET_PLOTS', output_dir), paste('loc_pred','map', pln, 'png', sep='.'))
      img = readPNG(img_fp)
      g = rasterGrob(img, interpolate=TRUE)
      
      
      p <- compare_df %>%
        ungroup() %>%
        filter(plot_location == pln) %>%
        select(model_method, method, model_time, expected_elevation, modern_surface_elevation, combined_std) %>%
        ggplot(aes(x = model_time, y = expected_elevation, color = method, fill = method, linetype=model_method)) +
        geom_ribbon(aes(ymax = modern_surface_elevation, ymin=(modern_surface_elevation - reference_depth)), fill='gray', color='black', color='black', alpha=.1, size=0.1, linetype='solid') +
        geom_ribbon(aes(ymin = (expected_elevation - (2*combined_std)),
                        ymax = (expected_elevation + (2*combined_std))), alpha=0.2) +
        geom_line() +
        theme_bw(base_size=9) +
        scale_fill_manual(name = 'Method', values = colors) +
        scale_color_manual(name = 'Method', values = colors) +
        scale_linetype_manual(name = 'Model-Calibration\nUncertainty Approach', values=c('Combined'='dashed', 'Independent'='solid')) +
        xlab('Model Time (ka)') +
        ylab(TeX('Elevation (multi-model expected value $\\pm 2 \\sigma$, ft)')) +
        ggtitle(pln) 
      figure_fp <- file.path(output_dir, paste('uncertainty_comparison', loc, pln, 'pdf', sep='.'))
      
      pdf(file = figure_fp, width=6.5, height=5)
      grid.draw(p)
      img_vp <- viewport(x = 1, y = 0.985, 
                         width = 0.23, height = .25,
                         just = c("right", "top"))
      pushViewport(img_vp)
      grid.draw(g)
      dev.off()    }
    
  } # end combined uncertainty analysis. 
   } # end if location exists. 
} # end loc


