import logging
import math
import os

import click
import geopandas as gpd
import production  # noqa
import requests
import utils
from appdirs import user_cache_dir
from shapely.geometry import Point

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# folder to store temporary files, ... (assume RW access)
WORK_DIR = os.path.join(user_cache_dir("accessmod"), "healthsites")


@click.group()
def cli():
    pass


@cli.command()
@click.option("--extent", type=str, required=True, help="boundaries of acquisition")
@click.option("--output-dir", type=str, required=True, help="output data directory")
@click.option(
    "--overwrite", is_flag=True, default=False, help="overwrite existing files"
)
@click.option(
    "--token",
    type=str,
    required=True,
    envvar="HEALTHSITES_TOKEN",
    help="healthsites api token",
)
@click.option(
    "--amenity",
    type=str,
    required=False,
    help="filter health facilities by amenity property",
)
def download_healthsites(
    extent: str, output_dir: str, overwrite: bool, token: str, amenity: str = None
):
    """Download list of health facilities for accessmod analysis"""

    def url_builder(endpoint, filters):
        url = "https://healthsites.io/api/v2" + endpoint + "?"
        url += "&".join([f"{k}={v}" for k, v in filters.items()])
        return url

    logger.info("Test token and download facilities list")
    os.makedirs(WORK_DIR, exist_ok=True)

    r = requests.get(
        url_builder(
            "/facilities/count/",
            {
                "api-key": token,
                "extent": extent,
            },
        )
    )
    r.raise_for_status()
    dataset_len = r.json()
    assert dataset_len >= 0
    logger.info("Ready to download %s healthsites in the extent", dataset_len)

    dataset = []
    for page in range(math.ceil(dataset_len / 100)):
        r = requests.get(
            url_builder(
                "/facilities/",
                {
                    "api-key": token,
                    "extent": extent,
                    "page": page + 1,
                    #   "flat-properties": "false",
                },
            )
        )
        r.raise_for_status()
        dataset.extend(
            [
                {
                    "amenity": i.get("attributes", {}).get("amenity"),
                    "name": i.get("attributes", {}).get("name"),
                    "lastchange": i.get("attributes", {}).get("changeset_timestamp"),
                    "healthcare": i.get("attributes", {}).get("healthcare"),
                    "dispensing": i.get("attributes", {}).get("dispensing"),
                    "uuid": i.get("attributes", {}).get("uuid"),
                    "geo_type": i.get("centroid", {}).get("type"),
                    "osm_id": i.get("osm_id"),
                    "osm_type": i.get("osm_type"),
                    "geometry": Point(
                        i["centroid"]["coordinates"][0],
                        i["centroid"]["coordinates"][1],
                    ),
                }
                for i in r.json()
            ]
        )
        logger.info("Page %s downloaded", page)

    df = gpd.GeoDataFrame(dataset)
    if amenity:
        df = df[df.amenity == amenity]

    # upload results
    local_file = os.path.join(WORK_DIR, "facilities.gpkg")
    dst_file = os.path.join(output_dir, "facilities.gpkg")
    df.to_file(local_file, driver="GPKG")
    utils.upload_file(local_file, dst_file, overwrite)


if __name__ == "__main__":
    cli()
