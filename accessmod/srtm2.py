"""
 - Download SRTM tiles, from NASA EarthData servers
 - merge them, compute slope and reproject them
 - upload everything in an output_dir

Notes:
 - Info about datasets here: https://lpdaac.usgs.gov/
 - EarthData credentials are required. Registration is free, here:
   https://urs.earthdata.nasa.gov/users/new
"""

import logging
import os
from io import BytesIO
from typing import List
from zipfile import ZipFile

import click
import geopandas as gpd
import processing
import production  # noqa
import rasterio
import rasterio.merge
import requests
import utils
from appdirs import user_cache_dir
from bs4 import BeautifulSoup
from processing import RASTERIO_DEFAULT_PROFILE
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


# folder containing the pipeline source (assume RO access)
SRC_DIR = os.path.dirname(__file__)

# folder to store temporary files, ... (assume RW access)
WORK_DIR = os.path.join(user_cache_dir("accessmod"), "SRTM")

# if you want to use a cache server, update the url here
# LPDAAC_DOWNLOAD_URL = "http://127.0.0.1:8000/"
LPDAAC_DOWNLOAD_URL = "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/"

# connection info to get the authentication token here:
EARTHDATA_URL = "https://urs.earthdata.nasa.gov"
EARTHDATA_LOGIN_URL = "https://urs.earthdata.nasa.gov/login"
EARTHDATA_PROFILE_URL = "https://urs.earthdata.nasa.gov/profile"

# LP AAC are slow and error prone -> retry mandatory
TIMEOUT = 60
RETRY_ADAPTER = HTTPAdapter(
    max_retries=Retry(
        total=5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"],
    )
)

# mirror for SRTM tiles
TILE_PATH = os.environ.get(
    "ACCESSMOD_BUCKET_NAME",
    "s3://hexa-demo-accessmod/SRTM-tiles/",
)


def download_src(target_geom: BaseGeometry, username: str, password: str) -> List[str]:
    # get a list of tiles to cover target geometry
    bounding_boxes_file = os.path.join(SRC_DIR, "srtm30m_bounding_boxes.json")
    bounding_boxes = gpd.read_file(bounding_boxes_file, driver="GeoJSON")
    tiles = bounding_boxes[bounding_boxes.intersects(target_geom)]
    if tiles.empty:
        raise ValueError("No SRTM tile found for the area of interest.")
    logger.info(f"download() found {len(tiles)} SRTM tiles to cover target.")

    # log to USGS repo (1/ get auth token 2/ login on earthdata)
    session = requests.Session()
    session.mount("https://", RETRY_ADAPTER)
    session.mount("http://", RETRY_ADAPTER)

    r = session.get(EARTHDATA_URL, timeout=TIMEOUT)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    token = ""
    for element in soup.find_all("input"):
        if element.attrs.get("name") == "authenticity_token":
            token = element.attrs.get("value")
            break
    if not token:
        raise Exception("Token not found in EarthData login page.")

    r = session.post(
        EARTHDATA_LOGIN_URL,
        timeout=TIMEOUT,
        data={
            "username": username,
            "password": password,
            "authenticity_token": token,
        },
    )
    r.raise_for_status()
    logger.info(f"EarthData: Logged in as {username}.")

    # download a list of tiles
    downloaded_tiles = []
    for tile_name in tiles["dataFile"].values:
        tile_url = LPDAAC_DOWNLOAD_URL + tile_name
        file_name = os.path.join(WORK_DIR, tile_name)
        if file_name.endswith(".zip"):
            file_name = file_name[:-4]
        logger.info(f"Download {tile_url} to {file_name}")

        # try/except/continue ?
        r = session.get(tile_url, timeout=TIMEOUT)
        r.raise_for_status()

        if tile_name.endswith(".zip"):
            zip_tile = ZipFile(BytesIO(r.content))
            zip_content = zip_tile.namelist()
            assert len(zip_content) == 1, "incoherent zip tile"
            tile_content = zip_tile.read(zip_content[0])
        else:
            tile_content = r.content

        fd = open(file_name, "wb")
        fd.write(tile_content)
        downloaded_tiles.append(file_name)

    logger.info(f"Downloaded {len(downloaded_tiles)} tiles")
    return downloaded_tiles


def download_mirror(target_geom: BaseGeometry) -> List[str]:
    # get a list of tiles to cover target geometry
    bounding_boxes_file = os.path.join(SRC_DIR, "srtm30m_bounding_boxes.json")
    bounding_boxes = gpd.read_file(bounding_boxes_file, driver="GeoJSON")
    tiles = bounding_boxes[bounding_boxes.intersects(target_geom)]
    if tiles.empty:
        raise ValueError("No SRTM tile found for the area of interest.")
    logger.info(f"download() found {len(tiles)} SRTM tiles to cover target.")

    fs = utils.filesystem("s3://a-bucket/")

    # download a list of tiles
    downloaded_tiles = []
    for tile_code in tiles["dataFile"].values:
        full_remote_path = os.path.join(TILE_PATH, tile_code)
        full_local_path = os.path.join(WORK_DIR, tile_code)
        if full_local_path.endswith(".zip"):
            full_local_path = full_local_path[:-4]
        logger.info(f"Download {tile_code}")

        if not fs.exists(full_remote_path):
            raise Exception(f"tile {tile_code} not found in {TILE_PATH}")

        fd_remote = fs.open(full_remote_path, "rb")
        fd_remote_data = fd_remote.read()

        if tile_code.endswith(".zip"):
            zip_tile = ZipFile(BytesIO(fd_remote_data))
            zip_content = zip_tile.namelist()
            assert len(zip_content) == 1, "incoherent zip tile"
            tile_content = zip_tile.read(zip_content[0])
        else:
            tile_content = fd_remote_data

        fd_local = open(full_local_path, "wb")
        fd_local.write(tile_content)
        downloaded_tiles.append(full_local_path)

    logger.info(f"Downloaded {len(downloaded_tiles)} tiles from mirror")
    return downloaded_tiles


def merge_tiles(tiles: List[str]) -> str:
    dst_file = os.path.join(WORK_DIR, "mosaic.tif")
    with rasterio.open(tiles[0]) as src:
        meta = src.meta.copy()

    mosaic, dst_transform = rasterio.merge.merge(tiles)
    meta.update(RASTERIO_DEFAULT_PROFILE)
    meta.update(transform=dst_transform, height=mosaic.shape[1], width=mosaic.shape[2])

    with rasterio.open(dst_file, "w", **meta) as dst:
        dst.write(mosaic)

    logger.info(f"Merged {len(tiles)} tiles into mosaic {dst_file}.")
    return dst_file


def reproject(
    target_geom: BaseGeometry, raster_file: str, epsg: int, resolution: int
) -> str:
    dst_crs = CRS.from_epsg(int(epsg))
    _, shape, bounds = processing.create_grid(
        geom=target_geom, dst_crs=dst_crs, dst_res=resolution
    )
    raster_reproj_file_p1 = raster_file.replace(".tif", "_reproj_p1.tif")
    raster_reproj_file_p2 = raster_file.replace(".tif", "_reproj_p2.tif")
    processing.reproject(
        raster_file,
        raster_reproj_file_p1,
        dst_crs=dst_crs,
        dtype="int16",
        bounds=bounds,
        xres=resolution,
        yres=-resolution,
        resampling_alg="bilinear",
    )
    processing.mask(raster_reproj_file_p1, raster_reproj_file_p2, target_geom)
    logger.info(f"Reprojected {raster_file} into {raster_reproj_file_p2}")
    return raster_reproj_file_p2


@click.group()
def cli():
    pass


@cli.command()
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
@click.option("--config", type=str, required=True, help="pipeline configuration")
def compute_dem(
    username: str,
    password: str,
    config: str,
):
    logger.info("compute_dem() starting")
    config = utils.parse_config(config)

    # create temporary workdir, if it is not existing
    os.makedirs(WORK_DIR, exist_ok=True)

    # download tiles
    target_geometry = utils.parse_extent(config["extent"], config["crs"])
    # if you don't have access to blsq SRTM mirror: use download_src
    # tiles = download_src(target_geometry, username, password)
    tiles = download_mirror(target_geometry)

    # geo stuff
    dem_file = merge_tiles(tiles)
    dem_proj_file = reproject(
        target_geometry, dem_file, config["crs"], config["spatial_resolution"]
    )
    utils.upload_file(dem_proj_file, config["dem"]["path"], config["overwrite"])
    utils.call_webhook(
        event_type="acquisition_completed",
        data={
            "acquisition_type": "dem",
            "uri": config["dem"]["path"],
            "mime_type": "image/geotiff",
        },
        url=os.environ.get("HEXA_WEBHOOK_URL"),
        token=os.environ.get("HEXA_WEBHOOK_TOKEN"),
    )

    logger.info("compute_dem() finished")


if __name__ == "__main__":
    cli()
