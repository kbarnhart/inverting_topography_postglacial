library(raster)
library(ncdf4)
library(rgeos)
# for each location

fsep = .Platform$file.sep

for (loc in c('gully', 'sew')) {
  fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction', loc, 'BEST_PARAMETERS')
  
  if (file.exists(fp)){ # if location DIR directory exits. 
    
    od <- gsub('study3py', paste('study3py', 'result_figures', sep=fsep), fp) 
    output_dir <- gsub('BEST_PARAMETERS', 'INSET_PLOTS', od)
    if(dir.exists(output_dir) == FALSE){
      dir.create(output_dir, recursive=TRUE)
    } # end if dir doesn't exist
    
    
    # plot locations given as column, row (e.g. x, y)
    point_fp <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction', loc, 'PredictionPoints_ShortList.csv')
    plot_loc_df <- read.csv(point_fp)
    
    xy <- cbind(plot_loc_df$Column_number+1, plot_loc_df$Row_number+1)
    plns <- plot_loc_df$Point_Name
    
    # finally plot the locations
    folder <- file.path('', 'work', 'WVDP_EWG_STUDY3', 'study3py', 'auxillary_inputs', 'dems',  loc, 'modern')
    nc_file <- list.files(folder, pattern='*nc', full.names = TRUE)[1]
    
    # read netcdf file
    z <- raster(nc_file, unit='feet')
    NAvalue(z) <- -9999
    z[z<0] <- NA
    crs(z) <- "+proj=tmerc +lat_0=40 +lon_0=-78.58333333333333 +k=0.9999375 +x_0=350000.0001016001 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +to_meter=0.3048006096012192 +no_defs " 
    
    slope <- terrain(z, opt='slope')
    aspect <- terrain(z, opt='aspect')
    hill <- hillShade(slope, aspect, angle=45, direction=315)
    
    is_data = z>0
    is_data[is_data<1] <- NA
    pp <- rasterToPolygons(is_data, dissolve=TRUE, digits = 2, na.rm=TRUE)
    
    plot(pp, lwd=1, border='black')
    
    figure_fp <- file.path(output_dir, paste('loc_pred','map', 'png', sep='.'))
    png(figure_fp, width=2*dim(z)[1] , height=2*dim(z)[2])
    plot.new()
    par(mar=c(0,0,0,0), oma=c(0,0,0,0))
    plot.window(xlim=extent(z)[1:2], ylim=extent(z)[3:4], xaxs="i",yaxs="i")
    
    i <- plot(hill, col=grey(0:100/100, alpha=0.6), add=TRUE, axes=FALSE, legend=FALSE, asp=1)
    #plot(z, col=terrain.colors(25, alpha=0.15), add=TRUE, legend=FALSE, axes=FALSE, asp=1)
    plot(pp, lwd=1, border='black', add=TRUE, axes=FALSE)

    for(row in 1:nrow(plot_loc_df)){
      
      # get index -- note that R is base one and python is base zero. 
      # in python, Z is of shape (415, 438) expects (rn, cn) while 
      # in R, Z is of shape (438, 415) expects (cn, rn)
      pln <- plot_loc_df$Point_Name[row]
      
      rn <- plot_loc_df$Row_number[row] + 1 
      cn <- plot_loc_df$Column_number[row] + 1
      
      i <- i+ points(cn+0.5, rn+0.5, bg='#900C3F', col='#FFE22C', cex=2, lwd=1, pch=21, add=TRUE, legend=FALSE, frame.plot=FALSE) # plot in the middle of the cell. 
      #i <- i+ text(cn+15,rn+15,labels=pln)
    } # end for row
    
    dev.off()
    
    
    figure_fp <- file.path(output_dir, paste('loc_pred','map', 'pdf', sep='.'))
    pdf(figure_fp, width=6.5 , height=6.5)
    plot.new()
    par(mar=c(0,0,0,0), oma=c(0,0,0,0))
    plot.window(xlim=extent(z)[1:2], ylim=extent(z)[3:4], xaxs="i",yaxs="i")
    
    i <- plot(hill, col=grey(0:100/100, alpha=0.6), add=TRUE, axes=FALSE, legend=FALSE, asp=1)
    #plot(z, col=terrain.colors(25, alpha=0.15), add=TRUE, legend=FALSE, axes=FALSE, asp=1)
    plot(pp, lwd=1, border='black', add=TRUE, axes=FALSE)
    
    for(row in 1:nrow(plot_loc_df)){
      
      # get index -- note that R is base one and python is base zero. 
      # in python, Z is of shape (415, 438) expects (rn, cn) while 
      # in R, Z is of shape (438, 415) expects (cn, rn)
      pln <- plot_loc_df$Point_Name[row]
      
      rn <- plot_loc_df$Row_number[row] + 1 
      cn <- plot_loc_df$Column_number[row] + 1
      
      i <- i+ points(cn+0.5, rn+0.5, bg='#900C3F', col='#FFE22C', cex=2, lwd=1, pch=21, add=TRUE, legend=FALSE, frame.plot=FALSE) # plot in the middle of the cell. 
      i <- i+ text(cn,rn,labels=pln,cex=0.7)
    } # end for row
    
    dev.off()
    
    
    # now make one of each with plot:
    for(row in 1:nrow(plot_loc_df)){
      pln <- plot_loc_df$Point_Name[row]
      
      rn <- plot_loc_df$Row_number[row] + 1 
      cn <- plot_loc_df$Column_number[row] + 1
      
      figure_fp <- file.path(output_dir, paste('loc_pred','map', pln, 'png', sep='.'))
      
      png(figure_fp, width=830 , height=876)
      plot.new()
      par(mar=c(0,0,0,0), oma=c(0,0,0,0))
      plot.window(xlim=extent(z)[1:2], ylim=extent(z)[3:4], xaxs="i",yaxs="i")
      i <- plot(hill, col=grey(0:100/100, alpha=0.6), add=TRUE, axes=FALSE, legend=FALSE, asp=1)
      plot(pp, lwd=1, border='black', add=TRUE, axes=FALSE)
      #plot(z, col=terrain.colors(25, alpha=0.15), add=TRUE, legend=FALSE, axes=FALSE, asp=1)

      # get index -- note that R is base one and python is base zero. 
      # in python, Z is of shape (415, 438) expects (rn, cn) while 
      # in R, Z is of shape (438, 415) expects (cn, rn)
      
      # for(rowi in 1:nrow(plot_loc_df)){
      # 
      #   # get index -- note that R is base one
      #and python is base zero.
      #   # in python, Z is of shape (415, 438) expects (rn, cn) while
      #   # in R, Z is of shape (438, 415) expects (cn, rn)
      #   plni <- plot_loc_df$Point_Name[rowi]
      # 
      #   rni <- plot_loc_df$Row_number[rowi] + 1
      #   cni <- plot_loc_df$Column_number[rowi] + 1
      # 
      #   i <- i+ points(cni+0.5, rni+0.5, pch=21, col='white', bg='gray', cex=3, add=TRUE, legend=FALSE, frame.plot=FALSE) # plot in the middle of the cell.
      # }

      i <- i+ points(cn+0.5, rn+0.5, bg='#900C3F', col='#FFE22C', cex=6, lwd=1.5, pch=21, add=TRUE, legend=FALSE, frame.plot=FALSE) # plot in the middle of the cell. 
      #i <- i+ text(cn+15, rn+15, labels=pln)
      dev.off()
      
    } # end make one for each 
    
    remove(z)
  } # end if dir exists
} # end for location
