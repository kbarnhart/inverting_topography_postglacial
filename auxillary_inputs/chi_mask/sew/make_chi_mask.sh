gdal_rasterize \
-burn 1 \
-init 0 \
-of GTiff \
-tr 24 24 \
-te 1121548.2005005 886503.65051269 1132012.2005005 896415.65051269 \
chi_mask.shp chi_mask.tiff


gdal_translate -of AAIGrid \
chi_mask.tiff \
chi_mask.asc
