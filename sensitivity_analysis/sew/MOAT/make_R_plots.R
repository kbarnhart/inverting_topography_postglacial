library(tidyverse)

fsep = .Platform$file.sep
for (loc in c('sew', 'gully')) {
  fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'sensitivity_analysis', loc)
  for(infile in dir(fp, pattern='cat_of.morris_df_short.*.csv', full.names=TRUE)) {
    
    file_list  <- unlist(strsplit(infile, fsep))
    file_name <- file_list[8]
    model_name = unlist(strsplit(file_name, "\\."))[3]
    
    outfile <- paste(model_name, 'Morris_Summary', 'pdf', sep='.')
    out_folder_list <- file_list
    out_folder_list[5] <- 'result_figures'
    
    out_full_path = paste(paste(out_folder_list[1:7], sep='', collapse = fsep), outfile, sep=fsep)
    
    data <- read.table(infile,header=TRUE,sep=",",row.names=NULL)
    
    p <- ggplot(data, aes(mu_star, sigma_star, color=Input, shape=Initial.Condition)) +
      geom_point() +
      theme_light() +
      scale_colour_hue() +
      xlim(0, NA) + ylim(0, NA) +
      ggtitle(paste('MOAT Summary: ', loc, ' model ', unlist(strsplit(model_name, '_'))[2]))
    
    ggsave(out_full_path, width=6, height=4, units='in')
    print(paste(loc, model_name))
  }
  
}


