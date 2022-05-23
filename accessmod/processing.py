import json
import logging
import os
import tempfile
from typing import Tuple

import geopandas as gpd
import numpy as np
import rasterio
from osgeo import gdal
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import Resampling, aligned_target, calculate_default_transform
from rasterio.warp import reproject as rio_reproject
from rasterio.warp import transform_bounds, transform_geom
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from shapely.geometry import Polygon
from utils import filesystem

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


GDAL_CREATION_OPTIONS = [
    "TILED=TRUE",
    "BLOCKXSIZE=256",
    "BLOCKYSIZE=256",
    "COMPRESS=ZSTD",
    "PREDICTOR=2",
    "NUM_THREADS=ALL_CPUS",
]


RASTERIO_DEFAULT_PROFILE = {
    "driver": "GTiff",
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
    "compress": "zstd",
    "predictor": 2,
    "num_threads": "all_cpus",
}


GDAL_DTYPES = {
    "int16": gdal.GDT_Int16,
    "uint8": gdal.GDT_Byte,
}


GDAL_RESAMPLING_ALGS = [
    "near",
    "bilinear",
    "cubic",
    "cubicspline",
    "lanczos",
    "average",
    "rms",
    "mode",
    "max",
    "min",
    "med",
    "q1",
    "q2",
    "sum",
]


def create_grid(
    geom: Polygon, dst_crs: CRS, dst_res: int
) -> Tuple[rasterio.Affine, Tuple[int], Tuple[float]]:
    """Create a raster grid for a given area of interest.

    Parameters
    ----------
    geom : shapely geometry
        Area of interest.
    dst_crs : CRS
        Target CRS as a rasterio CRS object.
    dst_res : int or float
        Spatial resolution (in `dst_crs` units).
    Returns
    -------
    transform: Affine
        Output affine transform object.
    shape : tuple of int
        Output shape (height, width).
    bounds : tuple of float
        Output bounds.
    """
    bounds = transform_bounds(CRS.from_epsg(4326), dst_crs, *geom.bounds)
    xmin, ymin, xmax, ymax = bounds
    transform = from_origin(xmin, ymax, dst_res, dst_res)
    ncols = (xmax - xmin) / dst_res
    nrows = (ymax - ymin) / dst_res
    transform, ncols, nrows = aligned_target(transform, ncols, nrows, dst_res)
    logger.info(f"Generated raster grid of shape ({nrows}, {ncols}).")
    return transform, (nrows, ncols), bounds


def reproject(
    src_raster: str,
    dst_raster: str,
    dst_crs: CRS,
    dtype: str,
    bounds: Tuple[float] = None,
    height: int = None,
    width: int = None,
    xres: float = None,
    yres: float = None,
    resampling_alg: str = "bilinear",
    src_nodata: float = None,
    dst_nodata: float = None,
) -> str:
    """Reproject a raster.

    Reproject a raster with GDAL based on either:
        * bounds and shape
        * bounds and spatial resolution

    Parameters
    ----------
    src_raster : str
        Path to source raster.
    dst_raster : str
        Path to output raster.
    dst_crs : CRS
        Target CRS.
    dtype : str
        Target data type (ex: "int16").
    bounds : tuple of float, optional
        Target raster bounds (xmin, ymin, xmax, ymax).
    height : int, optional
        Target raster height.
    width : int, optional
        Target raster width.
    xres : float, optional
        Target X spatial resolution.
    yres : float, optional
        Target Y spatial resolution.
    resampling_alg : str, optional
        GDAL Resampling algorithm (default=bilinear).
    src_nodata : float, optional
        Nodata value in source raster.
    dst_nodata : float, optional
        Nodata value in output raster.

    Return
    ------
    str
        Path to output raster.
    """
    if height and not width:
        return ValueError("Both height and width must be provided.")
    if xres and not yres:
        return ValueError("Both xres and yres must be provided.")
    if xres and height:
        return ValueError("Shape and resolution cannot be used together.")
    if dtype not in GDAL_DTYPES:
        return ValueError(f"Data type {dtype} is not supported.")
    if resampling_alg not in GDAL_RESAMPLING_ALGS:
        return ValueError(f"Resampling algorithm {resampling_alg} is not supported.")

    gdal.Warp(
        dst_raster,
        src_raster,
        dstSRS=dst_crs.to_string(),
        outputType=GDAL_DTYPES[dtype],
        format="GTiff",
        outputBounds=bounds,
        height=height,
        width=width,
        xRes=xres,
        yRes=yres,
        resampleAlg=resampling_alg,
        srcNodata=src_nodata,
        dstNodata=dst_nodata,
        creationOptions=GDAL_CREATION_OPTIONS,
    )
    logger.info(f"Reprojected {src_raster} to {dst_crs.to_string()}.")
    return dst_raster


def mask(src_raster: str, dst_raster: str, geom: Polygon, src_crs: CRS = None):
    """Clip raster data based on a polygon.

    Parameters
    ----------
    src_raster : str
        Path to source raster.
    dst_raster : str
        Path to output raster.
    geom : shapely polygon
        Area of interest.
    src_crs : CRS, optional
        CRS of input geometry if different from source raster.

    Return
    ------
    str
        Path to output raster.
    """
    geom = geom.__geo_interface__
    # Reproject input geometry to same CRS as raster if needed
    with rasterio.open(src_raster) as src:
        dst_crs = src.crs
    if src_crs:
        geom = transform_geom(src_crs, dst_crs, geom)
        logger.info(
            f"Reprojected geometry from {src_crs.to_string()} to {dst_crs.to_string()}."
        )

    # GDAL needs the geometry as a file
    with tempfile.TemporaryDirectory("AccessMod_") as tmp_dir:

        geom_fp = os.path.join(tmp_dir, "geom.geojson")
        with open(geom_fp, "w") as f:
            json.dump(geom, f)

        options = gdal.WarpOptions(
            cutlineDSName=geom_fp,
            cropToCutline=False,
            creationOptions=GDAL_CREATION_OPTIONS,
        )

        gdal.Warp(dst_raster, src_raster, options=options)

    logger.info(f"Clipped raster {src_raster}.")
    return dst_raster


def enforce_crs(geodataframe: gpd.GeoDataFrame, crs: CRS) -> gpd.GeoDataFrame:
    """Enforce a given CRS on a geodataframe.

    If the geodataframe does not have any CRS assigned, it is assumed
    to be in WGS84.

    Parameters
    ----------
    geodataframe : geodataframe
        Input geopandas geodataframe.
    crs : pyproj CRS
        Target CRS.

    Return
    ------
    geodataframe
        Projected geodataframe.
    """
    if not geodataframe.crs:
        geodataframe.crs = CRS.from_epsg(4326)
        logger.debug("Geodataframe did not have any CRS assigned.")
    if geodataframe.crs != crs:
        geodataframe.to_crs(crs, inplace=True)
        logger.debug("Reprojected geodataframe.")
    return geodataframe


def get_raster_statistics(src_file: str) -> dict:
    """Compute basic raster statistics.

    This includes min, max, 1st percentile, 2nd percentile, 98th percentile, and
    99th percentile.
    """
    meta = {}
    fs = filesystem(src_file)
    with fs.open(src_file, "rb") as f:
        with rasterio.open(f) as src:
            nodata = src.nodata
            data = src.read(1)
            meta["dtype"] = src.dtypes[0]
            meta["nodata"] = nodata
            meta["min"] = data[data != nodata].min()
            meta["max"] = data[data != nodata].max()
            for percentile in (1, 2, 98, 99):
                meta[f"p{percentile}"] = np.percentile(
                    data[data != nodata].ravel(), percentile
                )
            # unique values
            if src.dtypes[0] in ("uint8", "int8", "int16"):
                unique = list(np.unique(data[data != nodata]))
                if len(unique) <= 20:
                    meta["unique_values"] = unique
    return meta


def generate_cog(src_file: str, dst_file: str, **options) -> str:
    """Generate cloud optimized geotiff from source raster.

    This file will be for dataviz purposes. NB: Paths must be local.
    """
    with rasterio.open(src_file) as src:

        # reproject to epsg:4326 in memory
        src_crs = src.crs
        dst_crs = CRS.from_epsg(4326)
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
            }
        )
        memfile = MemoryFile()
        with memfile.open(**kwargs) as mem:
            for i in range(1, src.count + 1):
                rio_reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(mem, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )

        # convert to cloud-optimized geotiff
        # set format creation options (see gdalwarp `-co` option)
        dst_profile = cog_profiles.get("deflate")
        dst_profile.update(dict(BIGTIFF="IF_SAFER"))

        # dataset open option (see gdalwarp `-oo` option)
        config = dict(
            GDAL_NUM_THREADS="ALL_CPUS",
            GDAL_TIFF_INTERNAL_MASK=True,
            GDAL_TIFF_OVR_BLOCKSIZE="128",
        )

        with memfile.open() as src_mem:
            cog_translate(
                src_mem,
                dst_file,
                dst_profile,
                config=config,
                in_memory=True,
                quiet=True,
                use_cog_driver=True,
                # web_optimized=True, # TODO: understand why this param hardcode epsg:3857 into the resulting file
                **options,
            )

        return dst_file


def generate_geojson(src_file: str, dst_file: str) -> str:
    """Generate a GeoJSON copy from input vector file.

    This is for dataviz purposes. NB: Paths must be local.
    """
    # target crs is epsg:4326 in dataviz
    DST_CRS = CRS.from_epsg(4326)

    src_geodata = gpd.read_file(src_file)

    # set default crs
    if not src_geodata.crs:
        src_geodata.crs = DST_CRS

    # reproject if needed
    if src_geodata.crs == DST_CRS:
        return src_file
    else:
        dst_geodata = src_geodata.to_crs(DST_CRS)
        dst_geodata.to_file(dst_file, driver="GeoJSON")
