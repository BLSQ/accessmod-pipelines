"""
 - Download CGLS tiles, from BLSQ cache
 - merge them, reproject and mask them to get land cover
 - upload everything in an output_dir

Notes:
 - Info about datasets here: https://lcviewer.vito.be/download
 - Urls are not the same year to year, patterns here:
```
https://s3-eu-west-1.amazonaws.com/vito.landcover.global/v3.0.1/2015/E000N40/E000N40_PROBAV_LC100_global_v3.0.1_2015-base_Discrete-Classification-map_EPSG-4326.tif
https://s3-eu-west-1.amazonaws.com/vito.landcover.global/v3.0.1/2016/E000N40/E000N40_PROBAV_LC100_global_v3.0.1_2016-conso_Discrete-Classification-map_EPSG-4326.tif
https://s3-eu-west-1.amazonaws.com/vito.landcover.global/v3.0.1/2017/E000N40/E000N40_PROBAV_LC100_global_v3.0.1_2017-conso_Discrete-Classification-map_EPSG-4326.tif
https://s3-eu-west-1.amazonaws.com/vito.landcover.global/v3.0.1/2018/E000N40/E000N40_PROBAV_LC100_global_v3.0.1_2018-conso_Discrete-Classification-map_EPSG-4326.tif
https://s3-eu-west-1.amazonaws.com/vito.landcover.global/v3.0.1/2019/E000N40/E000N40_PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif
```

"""

import logging
import os
from typing import List

import click
import geopandas as gpd
import processing
import production  # noqa
import rasterio
import rasterio.merge
import utils
from appdirs import user_cache_dir
from processing import RASTERIO_DEFAULT_PROFILE
from rasterio.crs import CRS
from shapely.geometry.base import BaseGeometry

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


LABELS = {
    "110": "Closed forest",
    "120": "Open forest",
    "20": "Shrubs",
    "30": "Herbaceous vegetation",
    "90": "Herbaceous wetland",
    "100": "Moss and lichen",
    "60": "Sparse vegetation",
    "40": "Cropland",
    "50": "Urban",
    "70": "Snow",
    "80": "Permanent water bodies",
    "200": "Open sea",
}


# folder containing the pipeline source (assume RO access)
SRC_DIR = os.path.dirname(__file__)

# folder to store temporary files, ... (assume RW access)
WORK_DIR = os.path.join(user_cache_dir("accessmod"), "CGLS")

# cache folder containing the tiles
TILE_PATH = "s3://hexa-demo-accessmod/CGLS-tiles/"


def download(target_geom: BaseGeometry, year: int) -> List[str]:
    # get a list of tiles to cover target geometry
    bounding_boxes_file = os.path.join(SRC_DIR, "cgls_bounding_boxes.json")
    bounding_boxes = gpd.read_file(bounding_boxes_file, driver="GeoJSON")
    tiles = bounding_boxes[bounding_boxes.intersects(target_geom)]
    if tiles.empty:
        raise ValueError("No GLC tile found for the area of interest.")
    logger.info(f"download() found {len(tiles)} CGLS tiles to cover target.")

    fs = utils.filesystem("s3://a-bucket/")

    # download a list of tiles
    downloaded_tiles = []
    for tile_code in tiles["dataFile"].values:
        file_name = f"{year}_{tile_code}_PROBAV_LC100_global_map_EPSG-4326.tif"
        full_local_path = os.path.join(WORK_DIR, file_name)
        full_remote_path = os.path.join(TILE_PATH, file_name)
        logger.info(f"Download {file_name}")

        if not fs.exists(full_remote_path):
            raise Exception(f"tile {file_name} not found in {TILE_PATH}")

        fd_remote = fs.open(full_remote_path, "rb")
        fd_local = open(full_local_path, "wb")
        fd_local.write(fd_remote.read())
        downloaded_tiles.append(full_local_path)

    logger.info(f"Downloaded {len(downloaded_tiles)} tiles")
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
        height=shape[0],
        width=shape[1],
        resampling_alg="mode",
    )
    processing.mask(raster_reproj_file_p1, raster_reproj_file_p2, target_geom)
    logger.info(f"Reprojected into {raster_reproj_file_p2}")
    return raster_reproj_file_p2


def reclassify(src_file: str) -> str:
    """Simplify labels."""
    dst_file = src_file.replace(".tif", "_reclassified.tif")
    with rasterio.open(src_file) as src:
        with rasterio.open(dst_file, "w", **src.profile) as dst:
            data = src.read(1)
            # closed forest
            data[(data >= 110) & (data < 120)] = 110
            # open forest
            data[(data >= 120) & (data < 130)] = 120
            dst.write(data, 1)
    return dst_file


@click.group()
def cli():
    pass


@cli.command()
@click.option("--config", type=str, required=True, help="pipeline configuration")
@click.option(
    "--webhook-url",
    type=str,
    help="URL to push a POST request with the acquisition's results",
)
@click.option("--webhook-token", type=str, help="Token to use in the webhook POST")
def generate_land_cover(
    config: str,
    webhook_url: str,
    webhook_token: str,
):
    logger.info("generate_land_cover() starting")
    config = utils.parse_config(config)

    # create temporary workdir, if it is not existing
    os.makedirs(WORK_DIR, exist_ok=True)

    # download tiles
    target_geometry = utils.parse_extent(config["extent"])
    tiles = download(target_geometry, config["land_cover"].get("year", 2019))

    # geo stuff
    land_cover = merge_tiles(tiles)
    land_cover_proj = reproject(
        target_geometry, land_cover, config["crs"], config["spatial_resolution"]
    )
    land_cover_reclass = reclassify(land_cover_proj)

    # get raster statistics (unique values, min, max, percentiles, etc)
    statistics = processing.get_raster_statistics(land_cover_reclass)

    utils.upload_file(
        land_cover_reclass, config["land_cover"]["path"], config.get("overwrite", True)
    )

    metadata = statistics.copy()
    metadata.update(labels=LABELS)

    utils.call_webhook(
        event_type="acquisition_completed",
        data={
            "acquisition_type": "land_cover",
            "uri": config["land_cover"]["path"],
            "mime_type": "image/geotiff",
            "metadata": metadata,
        },
        url=webhook_url,
        token=webhook_token,
    )
    logger.info("generate_land_cover() finished")


if __name__ == "__main__":
    cli()
