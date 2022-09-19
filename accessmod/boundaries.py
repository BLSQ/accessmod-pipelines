import json
import logging
import os
import tempfile
from urllib.parse import urlparse

import click
import geopandas
import production  # noqa
import requests
import shapely.wkt
import utils

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--config", type=str, required=True, help="pipeline configuration")
def download(config: str):
    config = utils.parse_config(config)
    country = config["country"]["iso-a2"]
    crs = config["crs"]
    level = config["boundaries"]["administrative_level"]
    output_path = config["boundaries"]["path"]

    u = urlparse(os.environ.get("HEXA_WEBHOOK_URL"))
    url = f"{u.scheme}://{u.netloc}/graphql/"
    query = (
        "query { "
        f'  boundaries(country_code: "{country}", level: "{level}")'
        "   {"
        "     name "
        "     extent "
        "   } "
        "} "
    )
    r = requests.post(
        url, headers={"Content-type": "application/json"}, json={"query": query}
    )
    r.raise_for_status()
    r = json.loads(r.content)

    colset = []
    dataset = []
    for b in r["data"]["boundaries"]:
        name = b["name"]
        data = b["extent"].split(";")
        srid = data[0]
        P = shapely.wkt.loads(data[1])
        colset.append(name)
        dataset.append(P)

    d = {"zone": colset, "geometry": dataset}
    gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
    gdf = gdf.to_crs(crs)

    tmp_dir = tempfile.TemporaryDirectory()
    tmp_file = os.path.join(tmp_dir.name, os.path.basename(output_path))
    gdf.to_file(tmp_file, driver="GPKG")
    fs = utils.filesystem(output_path)
    fs.put(tmp_file, output_path)

    utils.call_webhook(
        event_type="acquisition_completed",
        data={
            "acquisition_type": "boundaries",
            "uri": config["boundaries"]["path"],
            "mime_type": "application/geopackage+sqlite3",
        },
        url=os.environ.get("HEXA_WEBHOOK_URL"),
        token=os.environ.get("HEXA_WEBHOOK_TOKEN"),
    )


if __name__ == "__main__":
    cli()
