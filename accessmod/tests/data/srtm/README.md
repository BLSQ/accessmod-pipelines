SRTM tiles to cover Djibouti, resampled to a lower spatial resolution to save space (~90 m instead of 30 m) and speed up the tests :

``` sh
    mkdir 90m
    for raster in *.hgt;
        gdalwarp $raster "90m/$raster" -ts 1201 1201
    end
```
