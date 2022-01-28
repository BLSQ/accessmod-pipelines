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

import json
import logging
import os
import shutil
import zipfile
from typing import List

import geopandas as gpd
import pandas as pd
import rasterio
import rasterio.merge
import requests
from appdirs import user_cache_dir
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry
from urllib3.util import Retry

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SRTMError(Exception):
    pass


def to_iso_a2(iso_a3):
    """Convert ISO-A3 country code to ISO-A2."""
    countries = pd.read_csv("countries.csv")
    if iso_a3 not in countries["ISO-A3"].values:
        raise ValueError(f"Country code {iso_a3} is not a valid ISO-A3 code.")
    return countries[countries["ISO-A3"] == iso_a3]["ISO-A2"].values[0]


def _human_readable_size(size, decimals=1):
    """Transform size in bytes into human readable text."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1000:
            break
        size /= 1000
    return f"{size:.{decimals}f} {unit}"


def country_geometry(country_code: str, use_cache=True) -> Polygon:
    """Get country geometry from Eurostat.

    See <https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data
    /administrative-units-statistical-units/countries> for more info.

    Parameters
    ----------
    country_code : str
        ISO-A2 or ISO-A3 country code.
    use_cache : bool, optional
        Use cache if possible (default=True).

    Return
    ------
    shapely polygon
        Country geometry.
    """
    SCALE = "01m"  # highest spatial accuracy
    EPSG = "4326"
    RELEASE_YEAR = "2020"  # latest release

    country_code = country_code.upper()
    if len(country_code) == 3:
        country_code = to_iso_a2(country_code)

    fname = f"{country_code}-region-{SCALE}-{EPSG}-{RELEASE_YEAR}.geojson"
    url = (
        "https://gisco-services.ec.europa.eu/"
        f"distribution/v2/countries/distribution/{fname}"
    )

    # do not make a request to the Eurostat API if the country geometry
    # has already been downloaded.
    fp_cache = os.path.join(user_cache_dir("accessmod"), "countries", fname)
    if os.path.isfile(fp_cache) and use_cache:
        with open(fp_cache) as f:
            geojson = json.load(f)
            logger.debug(f"Loaded {country_code} geometry from cache {fp_cache}.")
            return shape(geojson["features"][0]["geometry"])

    os.makedirs(os.path.dirname(fp_cache), exist_ok=True)
    with requests.get(url) as r:
        geojson = r.json()
        logger.debug(f"Downloaded {country_code} geometry from {url}")

        if use_cache:
            with open(fp_cache, "w") as f:
                json.dump(geojson, f)
            logger.debug(f"Written {country_code} geometry to cache at {fp_cache}.")

        return shape(geojson["features"][0]["geometry"])


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

        retry_adapter = HTTPAdapter(
            max_retries=Retry(
                total=3,
                status_forcelist=[429, 500, 502, 503, 504],
                method_whitelist=["HEAD", "GET"],
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
        logger.debug(f"EarthData: Logged in as {username}.")

    @property
    def bounding_boxes(self):
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
        logger.debug(
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
            logger.debug(f"File {fp} already exists.")
            return fp

        if os.path.isfile(fp_cache) and not overwrite and use_cache:
            shutil.copyfile(fp_cache, fp)
            logger.debug(f"Found SRTM tile in cache at {fp_cache}.")
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
            logger.debug(f"Downloaded SRTM tile from {url} to {fp}.")

            size_local = os.path.getsize(fp)
            if size_local != int(size):
                raise SRTMError(
                    f"Size of {fp} is invalid "
                    f"(expected {_human_readable_size(int(size), decimals=3)}, "
                    f"got {_human_readable_size(size_local, decimals=3)})."
                )

            if use_cache:
                os.makedirs(os.path.dirname(fp_cache), exist_ok=True)
                shutil.copyfile(fp, fp_cache)
                logger.debug(f"Cached SRTM tile to {fp_cache}.")

        return fp


def unzip(
    src: str, dst_dir: str = None, remove_archive: bool = False, overwrite: bool = False
) -> List[str]:
    """Extract a zip archive.

    Parameters
    ----------
    src : str
        Path to zip archive.
    dst_dir : str, optional
        Destination directory (same as zip archive by default).
    remove_archive : bool, optional
        Remove source archive after decompression
        (default=False).
    overwrite : bool, optional
        Overwrite existing files (default=False).

    Return
    ------
    list of str
        List of extracted files.
    """
    if not dst_dir:
        dst_dir = os.path.dirname(src)

    with zipfile.ZipFile(src, "r") as z:
        filenames = z.namelist()
        fp = os.path.join(dst_dir, filenames[0])
        if os.path.isfile(fp) and not overwrite:
            logger.debug(f"File {fp} already exists.")
        else:
            z.extractall(dst_dir)
            logger.debug(f"Extracted zip archive {src} to {dst_dir}.")

    if remove_archive:
        os.remove(src)
        logger.debug(f"Removed old zip archive {src}.")

    return [os.path.join(dst_dir, fn) for fn in filenames]


def _flatten(src_list: list) -> list:
    """Flatten a list of lists."""
    return [item for sublist in src_list for item in sublist]


def unzip_all(src_dir, dst_dir: str = None, remove_archive: bool = False) -> str:
    """Unzip all zip archives in a directory.

    Parameters
    ----------
    src_dir : str
        Source directory containing the zip archives.
    dst_dir : str, optional
        Target directory where archives are extracted to.
        Equals to source directory by default.

    Return
    ------
    list of str
        List of extracted files.
    """
    if not dst_dir:
        dst_dir = src_dir

    filenames = []
    for fn in os.listdir(src_dir):
        if fn.lower().endswith(".zip"):
            fp = os.path.join(src_dir, fn)
            filenames.append(unzip(fp, dst_dir))

    return _flatten(filenames)


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
        logger.debug(f"File {dst_file} already exists.")
        return dst_file

    with rasterio.open(tiles[0]) as src:
        meta = src.meta.copy()

    mosaic, dst_transform = rasterio.merge.merge(tiles)
    meta.update(
        {
            "driver": "GTiff",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": "zstd",
            "predictor": 2,
            "num_threads": "all_cpus",
            "transform": dst_transform,
        }
    )

    with rasterio.open(dst_file, "w", **meta) as dst:
        dst.write(mosaic)
    logger.debug(f"Merged {len(tiles)} tiles into mosaic {dst_file}.")

    return dst_file
