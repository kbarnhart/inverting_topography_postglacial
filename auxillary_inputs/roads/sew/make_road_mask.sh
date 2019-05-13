gdal_rasterize \
-burn 1 \
-init 0 \
-of GTiff \
-tr 24 24 \
-te 1121548.2005005 886503.65051269 1132012.2005005 896415.65051269 \
sew_roads_50_buffer.shp sew_road_mask.tiff

gdal_translate -of AAIGrid \
sew_road_mask.tiff \
sew_road_mask.txt

rm sew_road_mask.tiff
