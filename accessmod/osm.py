import logging
import os
import subprocess

import click
import geopandas as gpd
import pandas as pd
import production  # noqa
import requests
import utils
from appdirs import user_cache_dir

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# folder to store temporary files, ... (assume RW access)
WORK_DIR = os.path.join(user_cache_dir("accessmod"), "OSM")


def osmium(*args):
    r = subprocess.run(
        ["osmium"] + list(args),
        check=True,
        text=True,
        capture_output=True,
    )
    for line in r.stdout.split("\n"):
        # if it's here, process finished OK
        logger.info(line)


def extract_pbf(pbfs, target_file, target_geo, expressions, properties):
    logger.info("Starting thematic extraction of %s objects for %s", expressions, pbfs)

    geodfs = []

    for pbf in pbfs:
        for expression in expressions:
            filtered_fn = pbf.replace(".osm.pbf", "") + ".filterd.osm.pbf"
            json_fn = pbf.replace(".osm.pbf", "") + ".json"
            # tag filter
            osmium("tags-filter", pbf, expression, "-o", filtered_fn, "--overwrite")
            logger.info("Tags filter %s for %s done", expression, pbf)

            # export to geojson
            osmium("export", filtered_fn, "-o", json_fn, "--overwrite")
            logger.info("Export %s done", pbf)

            # import data + clean up
            geodf = gpd.read_file(json_fn)
            for column in geodf.columns:
                if column not in properties and column != "geometry":
                    tmp_geodf = geodf.drop([column], axis=1)
                    del geodf
                    geodf = tmp_geodf
            geodf_min = geodf[
                (geodf.geom_type == "LineString") & geodf.intersects(target_geo)
            ]
            del geodf
            geodfs.append(geodf_min)

    # concat all df into one big thing
    ndf = pd.concat(geodfs, ignore_index=True)
    del geodfs
    geodf = gpd.GeoDataFrame(ndf)
    del ndf
    logger.info("Consolidation done, generated %s objects", len(geodf))

    # post process geodf
    geodf_final = geodf.reset_index(drop=True)
    del geodf
    geodf_final.crs = {"init": "epsg:4326"}
    geodf_final.to_file(target_file, driver="GPKG")
    del geodf_final
    logger.info("Geodf dumped to %s", target_file)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--extent", type=str, required=True, help="boundaries of acquisition")
@click.option("--output-dir", type=str, required=True, help="output data directory")
@click.option(
    "--overwrite", is_flag=True, default=False, help="overwrite existing files"
)
def extract_from_osm(
    extent: str,
    output_dir: str,
    overwrite: bool,
):
    logger.info("extract_from_osm() make work dir")
    os.makedirs(WORK_DIR, exist_ok=True)

    target_geometry = utils.parse_extent(extent)
    all_countries = gpd.read_file("countries_pbf.json")
    all_countries["localpath"] = ""
    all_countries["localpbf"] = all_countries["pbf"].apply(
        lambda p: p.replace("/", "-")
    )

    logger.info("extract_from_osm() download source data")
    countries = all_countries[all_countries.intersects(target_geometry)]
    for i, country in countries.iterrows():
        localpath = os.path.join(WORK_DIR, country["localpbf"])
        countries.loc[i, "localpath"] = localpath
        r = requests.get("http://download.geofabrik.de/" + country["pbf"])
        r.raise_for_status()
        open(localpath, "wb").write(r.content)
        logger.info("Downloaded %s to %s", country["pbf"], localpath)

    transport_file = os.path.join(WORK_DIR, "transport.gpkg")
    extract_pbf(
        list(countries.localpath),
        transport_file,
        target_geometry,
        ["w/highway", "w/route=ferry"],
        [
            "highway",
            "smoothness",
            "surface",
            "tracktype",
            "route",
            "duration",
            "motor_vehicle",
            "motorcar",
            "motorcycle",
            "bicycle",
            "foot",
        ],
    )

    water_file = os.path.join(WORK_DIR, "water.gpkg")
    extract_pbf(
        list(countries.localpath),
        water_file,
        target_geometry,
        ["nwr/natural=water", "nwr/waterway", "nwr/water"],
        ["waterway", "natural", "water", "wetland", "boat"],
    )

    logger.info("extract_from_osm() upload")

    rem_transport_file = os.path.join(output_dir, "transport.gpkg")
    rem_water_file = os.path.join(output_dir, "water.gpkg")
    utils.upload_file(transport_file, rem_transport_file, overwrite)
    utils.upload_file(water_file, rem_water_file, overwrite)
    logger.info("extract_from_osm() finished")


if __name__ == "__main__":
    cli()
