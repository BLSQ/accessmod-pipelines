import base64
import json
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
@click.option("--config", type=str, required=True, help="pipeline configuration")
@click.option(
    "--token",
    type=str,
    required=True,
    envvar="HEALTHSITES_TOKEN",
    help="healthsites api token",
)
def download_healthsites(
    config: str,
    token: str,
):
    """Download list of health facilities for accessmod analysis"""
    config = json.loads(base64.b64decode(config))

    def url_builder(endpoint, filters):
        url = "https://healthsites.io/api/v2" + endpoint + "?"
        url += "&".join([f"{k}={v}" for k, v in filters.items()])
        return url

    logger.info("Test token and download facilities list")
    os.makedirs(WORK_DIR, exist_ok=True)

    target_geometry = utils.parse_extent(config["extent"])
    str_bounds = ",".join([str(x) for x in target_geometry.bounds])

    r = requests.get(
        url_builder(
            "/facilities/count/",
            {
                "api-key": token,
                "extent": str_bounds,
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
                    "extent": str_bounds,
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
    df = df[df.intersects(target_geometry)]
    if config["health_facilities"]["amenity"]:
        df = df[df.amenity == config["health_facilities"]["amenity"]]
    df = df.reset_index(drop=True)

    # upload results
    local_file = os.path.join(WORK_DIR, "facilities.gpkg")
    df.to_file(local_file, driver="GPKG")
    utils.upload_file(
        local_file, config["health_facilities"]["path"], config["overwrite"]
    )


if __name__ == "__main__":
    cli()
