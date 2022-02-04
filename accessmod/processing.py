import json
import logging
import os
import tempfile
from typing import Tuple

import rasterio
from osgeo import gdal
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import aligned_target, transform_bounds, transform_geom
from shapely.geometry import Polygon

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
            json.dump(geom.__geo_interface__, f)

        options = gdal.WarpOptions(
            cutlineDSName=geom_fp,
            cropToCutline=False,
            creationOptions=GDAL_CREATION_OPTIONS,
        )

        gdal.Warp(dst_raster, src_raster, options=options)

    logger.info(f"Clipped raster {src_raster}.")
