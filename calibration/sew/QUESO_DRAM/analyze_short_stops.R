# at the beginning of each model, use system() to combine the EGO2 and MCMC .rst and convert from .rst to .csv
# calculate AICc, and DEL, and then plot the points on the current plots
# make a second set of plots that shows the probability as a function of space. 

library(tidyverse)
library(plyr)
library(viridis)
library(gridExtra)
library(GGally)
library(MASS)
library(reshape2)
system('source ~/.bash_profile')
fsep = .Platform$file.sep

chi_elev_levels <- c('chi_elev_1', 'chi_elev_2', 'chi_elev_3', 'chi_elev_4', 'chi_elev_5',
                     'chi_elev_6', 'chi_elev_7', 'chi_elev_8', 'chi_elev_9', 'chi_elev_10',
                     'chi_elev_11', 'chi_elev_12', 'chi_elev_13', 'chi_elev_14', 'chi_elev_15',
                     'chi_elev_16', 'chi_elev_17', 'chi_elev_18', 'chi_elev_19', 'chi_elev_20')

param_df <- read_csv(file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'auxillary_inputs', 'parameter_ranges.csv'),
                     na=c('NaN', '', 'NA'))

# specify desired contour levels:
prob <- c(0.99, 0.95,0.90,0.5)

Sigma <- matrix(c(0.1,0.3,0.3,4),2,2)
mv <- data.frame(mvrnorm(4000,c(1.5,16),Sigma))
mv.kde <- kde2d(mv[,1], mv[,2], n = 400)
dx <- diff(mv.kde$x[1:2])  # lifted from emdbook::HPDregionplot()
dy <- diff(mv.kde$y[1:2])
sz <- sort(mv.kde$z)
c1 <- cumsum(sz) * dx * dy

# plot:
dimnames(mv.kde$z) <- list(mv.kde$x,mv.kde$y)
dc <- melt(mv.kde$z)
dc$prob <- approx(sz,1-c1,dc$value)$y
temp <- ggplot(dc,aes(x=Var1,y=Var2))+
  geom_contour(aes(z=prob,color=as.factor(..level..)),breaks=prob) +
  scale_color_viridis('Probability', direction = -1, discrete = TRUE) +
  theme_light()

grabbed <- grab_legend(temp)
print_short_stops <- TRUE
print_complex <- TRUE

# for each location
for (loc in c('sew', 'gully')) {
  fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'calibration', loc, 'QUESO_DRAM')
  df_list <- list()
  
  output_dir <- file.path(fp, 'figures')
  if(dir.exists(output_dir) == FALSE){
    dir.create(output_dir)
  }
  
  # for each model
  for(model_folder in dir(fp, pattern='model_*', full.names=TRUE)) {
    print(model_folder)
    
    model_key <- unlist(strsplit(model_folder, fsep))[8]
    model_name = unlist(strsplit(model_key, '_'))[2]
    
    # get lowering histories
    for(lh_folder in dir(model_folder, pattern='*', full.names=TRUE)) {
      bcic = unlist(strsplit(lh_folder, fsep))[9]
      
      # get list of short_stop folders
      short_stop_list <- list()
      
      if(file.exists(file.path(lh_folder, 'short_stop'))){
        
        full_extent_fig_list <- list()
        small_exstent_fig_list <- list()
        
        # get linear parameter estimate values
        estimate_fn <- paste('ego2', loc, 'parameters', 'full', model_key, 'csv', sep='.')
        estimate_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'calibration', loc, estimate_fn)
        df_est <- read.csv(estimate_fp)
        
        df_est <- df_est[, c('X', 'best_parameters', 'parameter_standard_deviation')]
        colnames(df_est) <- c('param', 'mean', 'std')
        df_est$lower <- df_est$mean - 2*df_est$std
        df_est$upper <- df_est$mean + 2*df_est$std
        
        est_extrema <- t(df_est[, c('lower', 'upper')])
        rownames(est_extrema) <- seq(1, nrow(est_extrema))
        colnames(est_extrema) <- df_est$param
        
        for(ss_folder in dir(file.path(lh_folder, 'short_stop'), pattern='*')) {
          print(ss_folder)
          
          posterior_file <- paste(lh_folder, 'short_stop', ss_folder, paste('posterior_', ss_folder, '.dat', sep=''), sep=fsep)
          
          if(file.exists(posterior_file)){
            df_pos <-read_table2(posterior_file)
            #df_pos$objective_function <- apply(select(df_pos, starts_with("chi")), 1,function(x) sum(x^2))
            interface_ind <- which(colnames(df_pos) == "interface")
            chi_1_ind <- which(colnames(df_pos) == "chi_elev_1")
            
            # select only those points that input parameters
            df_sel <- df_pos[,(interface_ind+1):(chi_1_ind-1)]
            var_order = colnames(df_sel)
            nvar <- length(var_order)
            
            # calculate the variance covariance matrix
            sel_cov <- cov(df_sel)
            
            # melt the variance covariance matrix to combine with 
            cov_melt <- reshape2::melt(sel_cov)
            cov_melt$itteration <- as.integer(ss_folder)
            
            # save covariance data frame to the short stop list
            short_stop_list[[as.integer(ss_folder)]] <- cov_melt
            
            if(print_short_stops){
              # plot the marginal scatterplots of all distributions
              # make sure limits are consitent, include one point that is range 
              extrema <- t(param_df[param_df$`Short Name` %in% colnames(df_sel), c('Short Name', 'Minimum Value', 'Maximum Value')])
              colnames(extrema) <- extrema[1,]
              extrema <- extrema[2:3,]
              extrema <- extrema[,colnames(df_sel)] # Re-orders columns
              if('linear_diffusivity_exp' %in% colnames(extrema)){
                extrema['Minimum Value', 'linear_diffusivity_exp'] <- "-4.0"
              }
              rownames(extrema) <- seq(1, nrow(extrema))
              mode(extrema) <- "numeric"
              
              # get extent of rectangle 
              df_rect <- extrema  
              rownames(df_rect) <- c('min', 'max')
              df_rect <- melt(df_rect)
              rowns <- paste(df_rect$Var2, df_rect$Var1, sep='.')
              df_rect <- as.data.frame(t(df_rect[,3]))
              colnames(df_rect) <- rowns
              
              title <- paste(loc, 'model', model_name, bcic, 'iteration', ss_folder)
              
              lower_function <- function(data, mapping, df_rect, df_est, 
                                         plot_error_bars=TRUE, set_limits=TRUE,
                                         plot_surrogate=TRUE){
                
                # get names
                xvar <- as.character(mapping$x)
                yvar <- as.character(mapping$y)
                
                #print(xvar)
                #print(yvar)
                #print(colnames(data))
                
                # first get x-y of estimate
                df_x <- df_est[df_est$param==xvar,]
                df_y <- df_est[df_est$param==yvar,]
                
                df_x[xvar] <- df_x$mean
                df_x[yvar] <- df_y$mean
                df_y[xvar] <- df_x$mean
                df_y[yvar] <- df_y$mean
                
                # get the density 
                mv <- data.frame(x = data[xvar], y = data[yvar])
                
                # get the kde2d information: 
                # this does not always work -- positive definite issues with 808. 
                # this causes error out. 
                mv.kde <- kde2d(mv[,1], mv[,2], n = 400)
                dx <- diff(mv.kde$x[1:2])  # lifted from emdbook::HPDregionplot()
                dy <- diff(mv.kde$y[1:2])
                sz <- sort(mv.kde$z)
                c1 <- cumsum(sz) * dx * dy
                
                # plot:
                dimnames(mv.kde$z) <- list(mv.kde$x,mv.kde$y)
                dc <- melt(mv.kde$z)
                dc$probability <- approx(sz,1-c1,dc$value)$y
                names(dc) <- c(xvar, yvar, 'value', 'probability')
                
                # still need to put custom min-max of density contours
                cust_plot <- ggplot(dc) + # base plot
                  geom_rect(data = df_rect, aes_string(xmin = paste(xvar, 'min', sep='.'), # extent rectangle 
                                                       ymin = paste(yvar, 'min', sep='.'), 
                                                       xmax = paste(xvar, 'max', sep='.'),
                                                       ymax = paste(yvar, 'max', sep='.')), fill=NA, color="black") +
                  geom_contour(aes_string(x=xvar,y=yvar, z='probability', color='as.factor(..level..)'), breaks=prob) + # plot contours
                  scale_color_viridis('Probability', direction = -1, discrete=TRUE) # set color of contours
                
                if(plot_error_bars){
                  cust_plot <- cust_plot + 
                    geom_errorbarh(data=df_x, aes_string(x=xvar, y=yvar, xmin='lower', xmax='upper'), color='gray') + # vertical error bar
                    geom_errorbar(data =df_y, aes_string(x=xvar, ymin='lower', ymax='upper'), color='gray') # horizontal error bar
                }
                
                if(set_limits){ }
                
                if(plot_surrogate){ }
                
                return(cust_plot)
                
              } #end lower function
              
              
              # make ininitial ggpairs structure       
              lower= list(continuous = wrap(lower_function, df_est=df_est, df_rect=df_rect)) #'density'
              p <- ggpairs(rbind(df_sel, extrema, est_extrema), lower=lower, switch='y', title=title, legend=grabbed) + theme_light()
              
              #print(p)
              
              lower= list(continuous = wrap(lower_function, df_est=df_est, df_rect=df_rect, plot_error_bars=FALSE)) #'density'
              
              p_s <- ggpairs(rbind(df_sel, extrema), lower=lower, switch='y', title=title, legend=grabbed) + theme_light()
              
              #print(p_s)
              
              full_extent_fig_list[[length(full_extent_fig_list)+1]] <-p
              small_exstent_fig_list[[length(small_exstent_fig_list)+1]] <-p_s
            } # end if plot  
            
          } # end if posterior file. 
          
        } # end shortstop folder
        
        if(print_short_stops){
          # print figures
          # figure that has full extent of linear estimate
          figure_fp <- file.path(output_dir, paste('scatterplotmatrix', loc, model_name, bcic, 'pdf', sep='.'))
          pdf(figure_fp, width=11, height=8.5)
          for(p in full_extent_fig_list){ # for fig list
            print(p)
          } #end fig list
          dev.off()
          
          # figure without full extent of linear estimate
          figure_fp <- file.path(output_dir, paste('scatterplotmatrix_zoom', loc, model_name, bcic, 'pdf', sep='.'))
          pdf(figure_fp, width=11, height=8.5)
          for(p in small_exstent_fig_list){ # for fig list
            print(p)
          } #end fig list
          dev.off()
        } # end if print scatterpots
        
      } # end if shortstop folder
      
      if(print_complex){
        # make plot of complex dataframe.
        
        # get ego and mcmc restart files
        ego_rst <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'calibration', loc, 'EGO2', model_key, bcic, 'dakota_calib.rst')
        mcmc_rst <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'calibration', loc, 'QUESO_DRAM', model_key, bcic, 'dakota_mcmc.rst')
        
        if((file.exists(ego_rst)&file.exists(mcmc_rst))){ # if .rst files exist
          
          # ego .txt file
          ego_txt <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'calibration', loc, 'QUESO_DRAM', model_key, bcic, 'dakota_calib.txt')
          #cmnd <- paste('dakota_restart_util', 'to_tabular', ego_rst, ego_txt)
          system2(command='dakota_restart_util', args = c('to_tabular', ego_rst, ego_txt))
          ego_df <- read_table2(ego_txt)
          
          # mcmc .txt file
          mcmc_txt <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'calibration', loc, 'QUESO_DRAM', model_key, bcic, 'dakota_mcmc.txt')
          #cmnd <- paste('dakota_restart_util', 'to_tabular', mcmc_rst, mcmc_txt)
          system2(command='dakota_restart_util', args = c('to_tabular', mcmc_rst, mcmc_txt))
          mcmc_df <- read_table2(mcmc_txt)
          
          # assign method names to the two data structures
          ego_df$method <- 'EGO'
          mcmc_df$method <- 'MCMC-initial'
          num_refine <- 5*10
          mcmc_df$method[(length(mcmc_df$method)-num_refine+1):length(mcmc_df$method)] <- 'MCMC-refinement'
          
          # create a dataframe of all complex model evalutions from EGO2 and MCMC
          complex_df <- rbind(ego_df, mcmc_df)
          interface_ind <- which(colnames(complex_df) == "interface")
          chi_1_ind <- which(colnames(complex_df) == "chi_elev_1")
          var_order <- colnames(complex_df[,(interface_ind+1):(chi_1_ind-1)])
          NP = length(var_order)
          ND <- length(chi_elev_levels)
          NPR <- 0
          w <- diag(ND)
          
          # calculate objective function, maximum likelihood objective function, and AICc
          complex_df$objective_function <- rowSums(complex_df[chi_elev_levels]^2)
          complex_df$ml_objective_function <- ((ND + NPR)*(log(2*pi))) - det(w) + complex_df$objective_function
          complex_df$AICc <- complex_df$objective_function + (NP * 2) + ((2*NP*(NP+1))/(ND + NPR - NP -1))
          
          # calculate del and the associated probability
          complex_df$del <- complex_df$AICc - min(complex_df$AICc)
          complex_small <- complex_df[complex_df$method == 'MCMC-initial',]
          complex_df$probability <- exp(-0.5*complex_df$del)/sum(exp(-0.5*complex_df$del))
          complex_small$probability <- exp(-0.5*complex_small$del)/sum(exp(-0.5*complex_small$del))
          
          lower_function <- function(data, mapping, ...){
            ggplot(data = data, mapping = mapping) +
              geom_point() +
              scale_color_viridis('Probability', limits = c(0,1))
          }
          
          complex_temp <- lower_function(data.frame(mvrnorm(4000,c(0.5,0.5),Sigma)), mapping=aes(x=X1, y=X2, color=X1))
          grabbed_complex <- grab_legend(complex_temp)
          
          upper= list(continuous = 'points',
                      mapping = aes(color=objective_function)) #'density'
          lower= list(continuous = lower_function,
                      mapping = aes(color=probability)) #'density'
          
          diagPlot = list(continuous = 'blankDiag')
          
          # SMALL COMPLEX 
          title <- paste(loc, 'model', model_name, bcic, '\nComplex Evaluations Initial MCMC')
          
          pm <- ggpairs(complex_small, 
                        mapping = aes(shape=method, color = probability), 
                        lower=lower, upper = upper, diag=diagPlot,
                        columns = var_order,
                        title=title,
                        legend = grabbed_complex) +
            theme_light()
          
          figure_fp <- file.path(output_dir, paste('complex_evals_small', loc, model_name, bcic, 'pdf', sep='.'))
          pdf(figure_fp, width=11, height=8.5)
          print(pm)
          dev.off()
          
          # Full COMPLEX 
          title <- paste(loc, 'model', model_name, bcic, '\nComplex Evaluations EGO&All MCMC')
          
          pm <- ggpairs(complex_df, 
                        mapping = aes(shape=method, color = probability), 
                        lower=lower, upper = upper, diag=diagPlot,
                        columns = var_order,
                        title=title,
                        legend = grabbed_complex) +
            theme_light()
          
          figure_fp <- file.path(output_dir, paste('complex_evals_ALL', loc, model_name, bcic, 'pdf', sep='.'))
          pdf(figure_fp, width=11, height=8.5)
          print(pm)
          dev.off()
          
        } # end if .rst files exist
        
      } #end print complex
      
      if(length(short_stop_list) > 0){
        # combine the shortstop list and plot
        df <- short_stop_list %>% reduce(bind_rows)
        df$Var1 <- ordered(df$Var1, levels=var_order)
        df$Var2 <- ordered(df$Var2, levels=var_order)
        
        p <- ggplot(df, aes(itteration, value)) +
          geom_line() + geom_point() +
          theme_light() +
          facet_grid(Var1~Var2, switch='y') + #scales='free',
          ggtitle(paste(loc, model_name, bcic, 'MCMC progression'))
        
        figure_fp <- file.path(output_dir, paste('mcmc_progression', loc, model_name, bcic, 'pdf', sep='.'))
        ggsave(figure_fp, width=11, height=8.5, units='in')
        
      }
      
    }# end lowering
    
  } # end model
  
} # end location
