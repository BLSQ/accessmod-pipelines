import base64
import json
import logging
import os
import subprocess
import tempfile
from collections import OrderedDict
from typing import Sequence

import click
import fiona
import geopandas as gpd
import processing
import production  # noqa
import requests
import utils
from appdirs import user_cache_dir
from pyproj import CRS
from shapely.geometry import Polygon, shape

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# folder to store temporary files, ... (assume RW access)
WORK_DIR = os.path.join(user_cache_dir("accessmod"), "OSM")


def osmium(*args):
    cmd = ["osmium"] + list(args)
    r = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
    )
    logger.info(" ".join(cmd))
    for line in r.stdout.split("\n"):
        # if it's here, process finished OK
        if line:
            logger.info(line)


def extract_pbf(
    src_files: Sequence[str],
    dst_file: str,
    area_of_interest: Polygon,
    expressions: Sequence[str],
    properties: Sequence[str],
    geom_type: str = "LineString",
) -> str:
    """Filter records from multiple .osm.pbf files based on location and tags.

    Records from each .osm.pbf file are filtered based on their tags and the
    osmium tag expressions provided by the user. Records are then exported to
    GeoJSON files and processed with Fiona to drop records outside of the area
    of interest.pbfs

    All valid records are merged into a final Geopackage.

    Parameters
    ----------
    src_files : list of str
        Input .osm.pbf files.
    dst_file : str
        Output .gpkg gile.
    area_of_interest : shapely polygon
        Geographic area of interest.
    expressions : list of str
        OSM tag filter expression (e.g. "w/highway").
    properties : list of str
        List of OSM tags to keep as column in the output .gpkg.
    geom_type : str, optional
        Geometry type (e.g. LineString or Polygon)

    Return
    ------
    str
        Path to output .gpkg file.
    """
    # minimal `osmium export` json config to drop unwanted attributes
    osmium_export_config = {
        "linear_tags": True,
        "area_tags": True,
        "include_tags": properties,
    }

    # schema of the output .gpkg file
    dst_schema = {
        "properties": OrderedDict({tag: "str" for tag in properties}),
        "geometry": geom_type,
    }

    # we will convert Polygon to MultiPolygon anyway
    if dst_schema.get("geometry") == "Polygon":
        dst_schema.update(geometry="MultiPolygon")

    fs_in = utils.filesystem(src_files[0])
    fs_out = utils.filesystem(dst_file)

    with tempfile.TemporaryDirectory(prefix="AccessMod_") as tmp_dir:

        dst_file_tmp = os.path.join(tmp_dir, os.path.basename(dst_file))

        for src_file_i, src_file in enumerate(src_files):

            src_file_tmp = os.path.join(tmp_dir, os.path.basename(src_file))
            fs_in.get(src_file, src_file_tmp)

            for expression_i, expression in enumerate(expressions):

                filtered_fp = src_file_tmp.replace(".osm.pbf", "_filtered.osm.pbf")
                geojson_fp = src_file_tmp.replace(".osm.pbf", ".geojson")

                osmium(
                    "tags-filter",
                    "--overwrite",
                    "-o",
                    filtered_fp,
                    src_file_tmp,
                    expression,
                )
                logger.info(f"Filtered {src_file_tmp} based on expression {expression}")

                # osmium export needs some config options to be provided in a
                # json file so we temporarily create one
                with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
                    with open(tmp_file.name, "w") as f:
                        json.dump(osmium_export_config, f)
                    osmium(
                        "export",
                        filtered_fp,
                        "-f",
                        "json",
                        f"--geometry-types={geom_type.lower()}",
                        "-c",
                        tmp_file.name,
                        "-o",
                        geojson_fp,
                        "--overwrite",
                    )
                logger.info(f"Exported {filtered_fp} to {geojson_fp}")

                with fiona.open(geojson_fp, driver="GeoJSON") as src:

                    # create a new gpkg if we are processing the first file, then
                    # append to existing one
                    mode = "w" if src_file_i == 0 and expression_i == 0 else "a"

                    def match_schema(record: dict, dst_schema: dict) -> dict:
                        """Ensure that the Fiona record matches the target schema.

                        If a property is in the record but not available in the
                        target schema, then the property is dropped. If the property
                        is in the target schema but not in the record, then it is
                        added to the record with None as value.
                        """
                        dst_properties = OrderedDict(
                            {
                                prop: record["properties"].get(prop)
                                for prop in dst_schema["properties"]
                            }
                        )
                        record.update(properties=dst_properties)
                        return record

                    def to_multipolygon(record: dict) -> dict:
                        """Convert polygon record to multipolygon."""
                        if record["geometry"].get("type") == "MultiPolygon":
                            return record
                        record["geometry"]["type"] = "MultiPolygon"
                        record["geometry"]["coordinates"] = [
                            record["geometry"]["coordinates"]
                        ]
                        return record

                    # we filter and write feature per feature in the destination
                    # gpkg to avoid memory issues in the largest areas of interest
                    n_records = 0
                    with fiona.open(
                        dst_file_tmp, mode, driver="GPKG", schema=dst_schema
                    ) as dst:
                        for record in src:
                            if "geometry" in record:
                                if geom_type == "Polygon":
                                    record = to_multipolygon(record)
                                geom = shape(record["geometry"])
                                if geom.intersects(area_of_interest):
                                    record = match_schema(record, dst_schema)
                                    dst.write(record)
                                    n_records += 1

                    logger.info(
                        f"Written {n_records} records from {geojson_fp} into {dst_file_tmp}"
                    )

        fs_out.put(dst_file_tmp, dst_file)
        logger.info(f"Put {dst_file_tmp} into {dst_file}")

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
def extract_from_osm(config: str, webhook_url: str, webhook_token: str):
    """Download and extract transport network and water data from OpenStreetMap."""
    logger.info("extract_from_osm() make work dir")

    # config is a json file
    if config.endswith(".json"):
        fs = utils.filesystem(config)
        with fs.open(config) as f:
            config = json.load(f)
    # config is base64 encoded json string
    else:
        config = json.loads(base64.b64decode(config))

    os.makedirs(WORK_DIR, exist_ok=True)

    target_geometry = utils.parse_extent(config["extent"], config["crs"])
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

    if config["transport_network"]["auto"]:
        logger.info("extract_from_osm() transport_network")
        transport_fp = os.path.join(WORK_DIR, "transport_latlon.gpkg")
        transport_reproj_fp = os.path.join(WORK_DIR, "transport.gpkg")
        extract_pbf(
            list(countries.localpath),
            transport_fp,
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

        src_crs = CRS.from_epsg(4326)
        dst_crs = CRS.from_epsg(config.get("crs"))
        processing.reproject_vector(
            src_file=transport_fp,
            dst_file=transport_reproj_fp,
            src_crs=src_crs,
            dst_crs=dst_crs,
        )

        utils.upload_file(
            transport_reproj_fp,
            config["transport_network"]["path"],
            config["overwrite"],
        )

        # save a geojson copy of the dataset for dataviz purposes
        with tempfile.NamedTemporaryFile(suffix=".geojson") as tmp_file:
            geojson_uri = utils.fpath_suffix(
                src_fpath=config["transport_network"]["path"],
                suffix="web",
                dst_extension="geojson",
            )
            geojson_tmp = processing.generate_geojson(
                transport_reproj_fp, tmp_file.name
            )
            utils.upload_file(geojson_tmp, geojson_uri, config.get("overwrite", True))

        # columns and unique values
        metadata = {
            "columns": ["highway", "smoothness", "surface", "tracktype"],
            "values": {
                "highway": []  # todo: for now these values are hardcoded in the front-end
            },
            "category_column": "highway",
            "geojson_uri": geojson_uri,
        }

        utils.call_webhook(
            event_type="acquisition_completed",
            data={
                "acquisition_type": "transport_network",
                "uri": config["transport_network"]["path"],
                "mime_type": "application/geopackage+sqlite3",
                "metadata": metadata,
            },
            url=webhook_url,
            token=webhook_token,
        )

    if config.get("water"):
        if config["water"].get("auto"):
            logger.info("extract_from_osm() water")
            water_fp = os.path.join(WORK_DIR, "water_latlon.gpkg")
            water_reproj_fp = os.path.join(WORK_DIR, "water.gpkg")
            extract_pbf(
                list(countries.localpath),
                water_fp,
                target_geometry,
                ["nwr/natural=water", "nwr/waterway", "nwr/water"],
                ["waterway", "natural", "water", "wetland", "boat"],
            )

            src_crs = CRS.from_epsg(4326)
            dst_crs = CRS.from_epsg(config.get("crs"))
            processing.reproject_vector(
                src_file=water_fp,
                dst_file=water_reproj_fp,
                src_crs=src_crs,
                dst_crs=dst_crs,
            )

            utils.upload_file(
                water_reproj_fp, config["water"]["path"], config["overwrite"]
            )
            utils.call_webhook(
                event_type="acquisition_completed",
                data={
                    "acquisition_type": "water",
                    "uri": config["water"]["path"],
                    "mime_type": "application/geopackage+sqlite3",
                },
                url=webhook_url,
                token=webhook_token,
            )
    logger.info("extract_from_osm() finished")


if __name__ == "__main__":
    cli()
