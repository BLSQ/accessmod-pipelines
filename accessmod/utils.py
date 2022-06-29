import base64
import json
import logging
import os
import random
import string
import uuid
import zipfile
from datetime import datetime
from typing import Any, List

import fsspec
import pandas as pd
import requests
from appdirs import user_cache_dir
from fiona.transform import transform_geom
from google.oauth2.credentials import Credentials
from shapely.geometry import Polygon, shape

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

APP_NAME = "AccessMod"
APP_AUTHOR = "Bluesquare"


def to_iso_a2(iso_a3: str) -> str:
    """Convert ISO-A3 country code to ISO-A2."""
    countries = pd.read_csv(os.path.join(os.path.dirname(__file__), "countries.csv"))
    if iso_a3 not in countries["ISO-A3"].values:
        raise ValueError(f"Country code {iso_a3} is not a valid ISO-A3 code.")
    return countries[countries["ISO-A3"] == iso_a3]["ISO-A2"].values[0]


def country_is_valid(country_code: str) -> bool:
    """Check validitity of country ISO A3 code."""
    if len(country_code) != 3:
        return False
    countries = pd.read_csv(os.path.join(os.path.dirname(__file__), "countries.csv"))
    return country_code.upper() in countries["ISO-A3"].values


def _flatten(iterable: Any) -> List:
    """Flatten a list or tuple of any depth."""
    if isinstance(iterable, list) or isinstance(iterable, tuple):
        return [value for i in iterable for value in _flatten(i)]
    else:
        return [iterable]


def is_latlon(coordinates: List) -> bool:
    """Check if geom coordinates seem to be in lat/lon.

    This is only based on min/max coordinates values for now.
    """
    xy = _flatten(coordinates)
    return min(xy) >= -180 and max(xy) <= 180


def parse_extent(extent: List, epsg: int) -> Polygon:
    """Convert extent from config into a shapely polygon.

    NB: extent is expected to be in WGS84 coordinate reference system, so if
    needed coordinates are reprojected to lat/lon based on the epsg parameter.
    Epsg parameter is ignored if the extent is detected as being in lat/lon.

    Parameters
    ----------
    extent : list
        Polygon coordinates.
    epsg : int
        EPSG code of the extent. Ignore if extent is detected as being in
        lat/lon coordinates.

    Return
    ------
    shapely polygon
        Extent geometry.
    """
    if is_latlon(extent):
        return Polygon(extent)
    else:
        geom = Polygon(extent)
        geom_latlon = transform_geom(
            f"EPSG:{epsg}", "EPSG:4326", geom.__geo_interface__
        )
        return shape(geom_latlon)


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


def _human_readable_size(size, decimals=1):
    """Transform size in bytes into human readable text."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1000:
            break
        size /= 1000
    return f"{size:.{decimals}f} {unit}"


def filesystem(target_path: str, cache: bool = False) -> fsspec.AbstractFileSystem:
    """Guess filesystem based on path.

    Parameters
    ----------
    target_path : str
        Target file path starting with protocol.
    cache : bool, optional
        Cache remote file locally (default=False).

    Return
    ------
    AbstractFileSystem
        Local, S3, or GCS filesystem. WholeFileCacheFileSystem if
        cache=True.
    """
    if "://" in target_path:
        target_protocol = target_path.split("://")[0]
    else:
        target_protocol = "file"

    if target_protocol not in ("file", "s3", "gcs"):
        raise ValueError(f"Protocol {target_protocol} not supported.")

    kwargs = {}
    if target_protocol == "s3":
        kwargs = {"client_kwargs": {"endpoint_url": os.environ.get("AWS_S3_ENDPOINT")}}
    elif target_protocol == "gcs":
        GCS_credentials = Credentials(token=os.environ.get("GCS_TOKEN"))
        kwargs = {"token": GCS_credentials}

    if cache:
        return fsspec.filesystem(
            protocol="filecache",
            target_protocol=target_protocol,
            target_options=kwargs,
            cache_storage=user_cache_dir(appname=APP_NAME, appauthor=APP_AUTHOR),
        )
    else:
        return fsspec.filesystem(protocol=target_protocol, **kwargs)


def random_string(length=16):
    chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
    return "".join(random.choice(chars) for i in range(length))


def call_webhook(event_type: str, data: dict, url: str = None, token: str = None):
    if not url or not token:
        return

    if event_type not in ["status_update", "acquisition_completed"]:
        raise ValueError(f"Webhook event_type {event_type} invalid.")

    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json={
            "id": str(uuid.uuid4()),
            "object": "event",
            "created": datetime.timestamp(datetime.utcnow()),
            "type": event_type,
            "data": data,
        },
    )
    r.raise_for_status()


def status_update(status: str, data: dict, url: str = None, token: str = None):
    """Update analysis status with webhook."""
    if not url or not token:
        return

    status = status.upper()
    if status not in (
        "DRAFT",
        "READY",
        "QUEUED",
        "RUNNING",
        "SUCCESS",
        "FAILED",
    ):
        raise ValueError(f"Analysis status {status} invalid.")
    data["status"] = status

    call_webhook(event_type="status_update", data=data, url=url, token=token)


def parse_config(string) -> dict:
    """Parse config option from CLI.

    Can be a path to a JSON file or a base 64 encoded JSON string. Parsed into a
    dictionary.
    """
    if string.endswith(".json"):
        fs = filesystem(string)
        with fs.open(string) as f:
            logger.info(f"Loading config from file {string}")
            return json.load(f)

    else:
        logger.info("Loading config from string")
        return json.loads(base64.b64decode(string.encode()).decode())


def upload_file(src_file: str, dst_file: str, overwrite: bool):
    fs = filesystem(dst_file)
    fs.makedirs(os.path.dirname(dst_file), exist_ok=True)

    if fs.exists(dst_file) and not overwrite:
        logger.info(f"upload_file(): {dst_file} exists and not overwrite")
        return

    src_fd = open(src_file, "rb")
    dst_fd = fs.open(dst_file, "wb")
    dst_fd.write(src_fd.read())
    dst_fd.close()


def fpath_suffix(src_fpath: str, suffix: str, dst_extension: None) -> str:
    """Add a suffix (before extension) to a remote or local file path.

    Also optionally change extension of new file path.
    """
    src_extension = src_fpath.split(".")[-1]
    dst_fpath = src_fpath.replace(f".{src_extension}", f"_{suffix}.{src_extension}")
    if dst_extension:
        if src_extension != dst_extension:
            dst_fpath = dst_fpath.replace(src_extension, dst_extension)
    return dst_fpath
