"""Search and download SRTM tiles.

The module provides a `SRTM` class to search and download SRTM tiles from the
NASA EarthData server.

Examples
--------
Downloading SRTM tiles to cover the area of interest `extent` into `output_dir`::

    srtm = SRTM()
    srtm.login(username, password)
    extent = country_geometry("COD")
    tiles = srtm.find(extent)
    for tile in tiles:
        srtm.download(tile, output_dir)

Notes
-----
EarthData credentials are required. Registration [1]_ is free.

References
----------
.. [1] `NASA EarthData Register <https://urs.earthdata.nasa.gov/users/new>`_
"""

import logging
import os
import shutil
import tempfile
from typing import List

import click
import geopandas as gpd
import processing
import rasterio
import rasterio.merge
import requests
import utils
from appdirs import user_cache_dir
from bs4 import BeautifulSoup
from osgeo import gdal
from processing import GDAL_CREATION_OPTIONS, RASTERIO_DEFAULT_PROFILE
from rasterio.crs import CRS
from requests.adapters import HTTPAdapter
from shapely.geometry.base import BaseGeometry
from urllib3.util import Retry

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SRTMError(Exception):
    pass


class SRTM:
    """Search and download SRTM data."""

    def __init__(self, timeout=30):
        """Initialize SRTM catalog."""
        self.LPDAAC_DOWNLOAD_URL = (
            "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/"
        )
        self.EARTHDATA_URL = "https://urs.earthdata.nasa.gov"
        self.EARTHDATA_LOGIN_URL = "https://urs.earthdata.nasa.gov/login"
        self.EARTHDATA_PROFILE_URL = "https://urs.earthdata.nasa.gov/profile"

        self.bounding_boxes = self.get_bounding_boxes()

        retry_adapter = HTTPAdapter(
            max_retries=Retry(
                total=3,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET"],
            )
        )

        self._session = requests.Session()
        self._session.mount("https://", retry_adapter)
        self._session.mount("http://", retry_adapter)
        self._timeout = timeout

    @property
    def _token(self) -> str:
        """Find authentiticy token in EarthData homepage as it is required to login.
        Returns
        -------
        token : str
            Authenticity token.
        """
        r = self._session.get(self.EARTHDATA_URL, timeout=self._timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        token = ""
        for element in soup.find_all("input"):
            if element.attrs.get("name") == "authenticity_token":
                token = element.attrs.get("value")
        if not token:
            raise requests.exceptions.ConnectionError(
                "Token not found in EarthData login page."
            )
        return token

    def login(self, username: str, password: str):
        """Login to NASA EarthData.

        Parameters
        ----------
        username : str
            NASA EarthData username.
        password : str
            NASA EarthData password.
        """
        r = self._session.post(
            self.EARTHDATA_LOGIN_URL,
            data={
                "username": username,
                "password": password,
                "authenticity_token": self._token,
            },
            timeout=self._timeout,
        )
        r.raise_for_status()
        logger.info(f"EarthData: Logged in as {username}.")

    def get_bounding_boxes(self):
        """Bounding boxes of SRTM tiles."""
        return gpd.read_file(
            os.path.join(
                os.path.dirname(__file__),
                "srtm30m_bounding_boxes.json",
            ),
            driver="GeoJSON",
        )

    def find(self, geom: BaseGeometry) -> List[str]:
        """Get the list of SRTM tiles required to cover a geometry.

        Parameters
        ----------
        geom : shapely geometry
            Area of interest.

        Return
        ------
        tiles : list of str
            Required tiles as a list of URLs.
        """
        tiles = self.bounding_boxes[self.bounding_boxes.intersects(geom)]
        if tiles.empty:
            raise ValueError("No SRTM tile found for the area of interest.")
        logger.info(
            f"{len(tiles)} SRTM tiles are required to cover the area of interest."
        )
        return [self.LPDAAC_DOWNLOAD_URL + tile for tile in tiles["dataFile"].values]

    def download(
        self, url: str, output_dir: str, overwrite: bool = False, use_cache: bool = True
    ) -> str:
        """Download a SRTM tile.

        Parameters
        ----------
        url : str
            URL of the tile.
        output_dir : str
            Path to output directory.
        overwrite : bool, optional
            Overwrite existing files (default=False).
        use_cache : bool, optional
            Use cache version if possible (default=True).

        Return
        ------
        fp : str
            Path to downloaded file.
        """
        fname = url.split("/")[-1]
        fp = os.path.join(output_dir, fname)
        os.makedirs(output_dir, exist_ok=True)

        fp_cache = os.path.join(user_cache_dir("accessmod"), "srtm", "tiles", fname)

        if os.path.isfile(fp) and not overwrite:
            logger.info(f"File {fp} already exists.")
            return fp

        if os.path.isfile(fp_cache) and not overwrite and use_cache:
            shutil.copyfile(fp_cache, fp)
            logger.info(f"Found SRTM tile in cache at {fp_cache}.")
            return fp

        with self._session.get(url, stream=True, timeout=self._timeout) as r:

            try:
                r.raise_for_status()
            except Exception as e:
                logger.error(e)

            if os.path.isfile(fp) and overwrite:
                os.remove(fp)

            size = r.headers.get("content-length")
            if not size:
                raise requests.exceptions.ConnectionError(
                    f"Cannot get size from URL {url}."
                )
            if int(size) < 1024:
                raise requests.exceptions.ConnectionError(
                    f"File at {url} appears to be empty."
                )

            with open(fp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            logger.info(f"Downloaded SRTM tile from {url} to {fp}.")

            size_local = os.path.getsize(fp)
            if size_local != int(size):
                raise SRTMError(
                    f"Size of {fp} is invalid "
                    f"(expected {utils._human_readable_size(int(size), decimals=3)}, "
                    f"got {utils._human_readable_size(size_local, decimals=3)})."
                )

            if use_cache:
                os.makedirs(os.path.dirname(fp_cache), exist_ok=True)
                shutil.copyfile(fp, fp_cache)
                logger.info(f"Cached SRTM tile to {fp_cache}.")

        return fp


def merge_tiles(tiles: List[str], dst_file: str, overwrite: bool = False):
    """Merge SRTM tiles into a single mosaic.

    Parameters
    ----------
    tiles : list of str
        Paths to SRTM tiles.
    dst_file : str
        Path to output geotiff.
    overwrite : bool, optional
        Overwrite existing files.

    Return
    ------
    dst_file : str
        Path to output geotiff.
    """
    if os.path.isfile(dst_file) and not overwrite:
        logger.info(f"File {dst_file} already exists.")
        return dst_file

    with rasterio.open(tiles[0]) as src:
        meta = src.meta.copy()

    mosaic, dst_transform = rasterio.merge.merge(tiles)
    meta.update(RASTERIO_DEFAULT_PROFILE)
    meta.update(transform=dst_transform, height=mosaic.shape[0], width=mosaic.shape[1])

    with rasterio.open(dst_file, "w", **meta) as dst:
        dst.write(mosaic)
    logger.info(f"Merged {len(tiles)} tiles into mosaic {dst_file}.")

    return dst_file


def compute_slope(dem: str, dst_file: str, overwrite: bool = False) -> str:
    """Compute slope (in degrees) from a DEM.

    NB: DEM is expected to be in WGS84.

    Parameters
    ----------
    dem : str
        Path to DEM.
    dst_file : str
        Path to output slope GeoTIFF.
    overwrite : bool, optional
        Overwrite existing files.

    Return
    ------
    dst_file : str
        Path to output slope GeoTIFF.
    """
    # if source DEM is in WGS 84 but pixel values are
    # in meters, the scale parameter must be set to 111120
    src_ds = gdal.Open(dem)
    scale = None
    if not src_ds.GetSpatialRef().IsProjected():
        scale = 111120

    if os.path.isfile(dst_file) and not overwrite:
        logger.info(f"File {dst_file} already exists.")
        return dst_file

    options = gdal.DEMProcessingOptions(
        format="GTiff",
        scale=scale,
        slopeFormat="degree",
        creationOptions=GDAL_CREATION_OPTIONS,
    )
    gdal.DEMProcessing(dst_file, dem, "slope", options=options)
    logger.info(f"Computed slope {dst_file} from DEM {dem}.")
    return dst_file


@click.group()
def cli():
    pass


@cli.command()
@click.option("--country", type=str, required=True, help="country code")
@click.option(
    "--username",
    type=str,
    required=True,
    envvar="EARTHDATA_USERNAME",
    help="earthdata username",
)
@click.option(
    "--password",
    type=str,
    required=True,
    envvar="EARTHDATA_PASSWORD",
    help="earthdata password",
)
@click.option("--output-dir", type=str, required=True, help="output directory")
@click.option(
    "--overwrite", is_flag=True, default=False, help="overwrite existing files"
)
def download(
    country: str,
    username: str,
    password: str,
    output_dir: str,
    overwrite: bool,
):
    """Download SRTM tiles."""
    geom = utils.country_geometry(country)
    catalog = SRTM()
    catalog.login(username, password)
    tiles = catalog.find(geom)
    for tile in tiles:
        catalog.download(tile, output_dir, overwrite=overwrite, use_cache=True)


@cli.command()
@click.option("--country", type=str, required=True, help="country code")
@click.option("--epsg", type=int, required=True, help="target epsg code")
@click.option("--resolution", type=int, required=True, help="spatial resolution (m)")
@click.option("--input-dir", type=str, required=True, help="input directory")
@click.option("--output-dir", type=str, required=True, help="output directory")
@click.option(
    "--overwrite", is_flag=True, default=False, help="overwrite existing files"
)
def process(
    country: str,
    epsg: int,
    resolution: float,
    input_dir: str,
    output_dir: str,
    overwrite: bool,
):
    """Process SRTM tiles.

    Tiles are merged into a single mosaic, reprojected and masked according to
    the area of interest. In addition, a slope raster (in degrees) is computed.
    Raster grid is created from country, epsg and spatial resolution parameters.
    """
    dst_dem = os.path.join(output_dir, "dem.tif")
    dst_slope = os.path.join(output_dir, "slope.tif")
    os.makedirs(output_dir, exist_ok=True)

    # get raster metadata from country boundaries, spatial resolution and EPSG
    geom = utils.country_geometry(country)
    dst_crs = CRS.from_epsg(int(epsg))
    _, shape, bounds = processing.create_grid(
        geom=geom, dst_crs=dst_crs, dst_res=resolution
    )

    for fp, label in zip((dst_dem, dst_slope), ("DEM", "Slope")):
        if os.path.isfile(fp):
            if overwrite:
                os.remove(fp)
                logger.info(f"Deleted old {label} file at {fp}.")
            else:
                return FileExistsError(f"{label} already exists at {fp}.")

    with tempfile.TemporaryDirectory(prefix="AccessMod_") as tmp_dir:

        tiles = utils.unzip_all(input_dir, tmp_dir)
        if len(tiles) == 0:
            return FileNotFoundError(f"No SRTM tile found at {input_dir}.")

        mosaic = merge_tiles(
            tiles, os.path.join(tmp_dir, "mosaic.tif"), overwrite=overwrite
        )

        # slope is computed from DEM before reprojection and masking
        slope = compute_slope(
            mosaic, os.path.join(tmp_dir, "slope.tif"), overwrite=overwrite
        )

        for src_fp, dst_fp in zip((mosaic, slope), (dst_dem, dst_slope)):

            tmp_fp = src_fp.replace(".tif", "_reproj.tif")
            tmp_fp = processing.reproject(
                src_fp,
                tmp_fp,
                dst_crs=dst_crs,
                dtype="int16",
                bounds=bounds,
                height=shape[0],
                width=shape[1],
                resampling_alg="bilinear",
            )
            processing.mask(tmp_fp, dst_fp, geom)


if __name__ == "__main__":
    cli()
