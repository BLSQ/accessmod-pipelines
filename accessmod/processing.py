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
from rasterio.transform import from_origin
from rasterio.warp import aligned_target, transform_bounds, transform_geom
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
        xres = src.transform.a
        yres = src.transform.e
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
            xRes=xres,
            yRes=yres,
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

    return dst_file


def reproject_vector(src_file: str, dst_file: str, src_crs: CRS, dst_crs: CRS) -> str:
    """Reproject a vector file with GDAL.

    NB: Paths must be local.
    """
    options = gdal.VectorTranslateOptions(
        srcSRS=src_crs.to_string(), dstSRS=dst_crs.to_string()
    )
    gdal.VectorTranslate(dst_file, src_file, options=options)
    return dst_file
