"""Zonal statistics pipeline."""

import logging
import os
from typing import List

import click
import numpy as np
import pandas as pd
from layer import BoundariesLayer, PopulationLayer, TravelTimesLayer
from processing import enforce_crs
from rasterio.warp import Resampling, reproject
from rasterstats import zonal_stats
from utils import filesystem, parse_config, status_update

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


APP_NAME = "AccessMod"
APP_AUTHOR = "Bluesquare"


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config",
    type=str,
    required=True,
    help="analysis config as a b64 encoded json string",
)
def zonalstats(config: str):
    """Compute zonal statistics."""

    status_update(
        status="RUNNING",
        data={},
        url=os.environ.get("HEXA_WEBHOOK_URL"),
        token=os.environ.get("HEXA_WEBHOOK_TOKEN"),
    )

    config = parse_config(config)

    boundaries = BoundariesLayer(filepath=config["boundaries"]["path"])
    population = PopulationLayer(filepath=config["population"]["path"])
    travel_times = TravelTimesLayer(filepath=config["travel_times"]["path"])

    pop = population_counts(boundaries, population)
    pop = pop.rename("PopTotal").round()
    pop.fillna(0, inplace=True)

    time_thresholds = [int(threshold) for threshold in config["time_thresholds"]]
    pop_time = time_stats(travel_times, boundaries, population, levels=time_thresholds)

    report = boundaries.read().copy()
    report = report.join(pop)

    for mn, count in pop_time.items():

        count = count.rename(f"PopTravelTime_{mn}mn")
        count = count.round()
        count = count.fillna(0)
        report = report.join(count)
        report[f"PopCoveredRatio_{mn}mn"] = (
            report[f"PopTravelTime_{mn}mn"] / report["PopTotal"]
        ).round(4)

    fs = filesystem(config["output_dir"])
    fs.makedirs(config["output_dir"], exist_ok=True)

    gpkg = os.path.join(config["output_dir"], "zonal_stats.gpkg")
    with fs.open(gpkg, "wb") as f:
        report.to_file(f, driver="GPKG")

    csv = os.path.join(config["output_dir"], "zonal_stats.csv")
    with fs.open(csv, "wb") as f:
        report.drop(["geometry"], axis=1).to_csv(f, index=False)

    status_update(
        status="SUCCESS",
        data={"outputs": {"zonal_statistics_geo": gpkg, "zonal_statistics_table": csv}},
        url=os.environ.get("HEXA_WEBHOOK_URL"),
        token=os.environ.get("HEXA_WEBHOOK_TOKEN"),
    )


def population_counts(
    boundaries: BoundariesLayer, population: PopulationLayer
) -> pd.Series:
    """Count population in each boundary.

    Parameters
    ----------
    boundaries : BoundariesLayer
        Input boundaries vector layer.
    population : PopulationLayer
        Input population raster layer (population count per pixel).

    Return
    -------
    Serie
        Population counts per area as a pandas Serie.
    """
    areas = enforce_crs(boundaries.read(), crs=population.meta["crs"])
    shapes = [area.__geo_interface__ for area in areas.geometry]
    stats = zonal_stats(
        shapes,
        population.read(),
        affine=population.meta["transform"],
        stats=["sum"],
        nodata=population.meta["nodata"],
    )
    data = pd.Series(data=[stat["sum"] for stat in stats], index=areas.index)
    logger.info(f"Counted population for {len(shapes)} areas (total={data.sum()})")
    return data


def time_stats(
    travel_times: TravelTimesLayer,
    boundaries: BoundariesLayer,
    population: PopulationLayer,
    levels: List[int] = [30, 60, 90, 120, 150, 180, 240, 300, 360],
):
    """Aggregate travel times statistics.

    Compute accessibility statistics at the boundary level based on travel time
    and population. Calculate population covered for multiple time thresholds.

    Parameters
    ----------
    travel_times : TravelTimesLayer
        Input travel times raster layer (in minutes).
    boundaries : BoundariesLayer
        Input boundaries vector layer.
    population : PopulationLayer
        Input population raster layer (population count per pixel).
    levels : list of int, optional
        Travel times thresholds in minutes.

    Return
    ------
    metrics : dict
        Accessibility statistics for each travel time threshold and for each
        boundary.
    """
    metrics = {}
    time = np.zeros(shape=population.meta["shape"], dtype="int32")
    reproject(
        source=travel_times.read(),
        destination=time,
        src_crs=travel_times.meta["crs"],
        src_transform=travel_times.meta["transform"],
        dst_crs=population.meta["crs"],
        dst_transform=population.meta["transform"],
        src_nodata=travel_times.meta["nodata"],
        dst_nodata=-1,
        resampling=Resampling.bilinear,
    )
    logger.info("Reprojected travel times to population CRS")

    areas = enforce_crs(boundaries.read(), crs=population.meta["crs"])
    shapes = [area.__geo_interface__ for area in areas.geometry]

    ppp_src = population.read()

    for lvl in levels:
        ppp = ppp_src.copy()
        ppp[time >= lvl] = 0
        ppp[time < 0] = 0
        stats = zonal_stats(
            shapes,
            ppp,
            affine=population.meta["transform"],
            stats=["sum"],
            nodata=population.meta["nodata"],
        )
        metrics[lvl] = pd.Series(
            data=[stat["sum"] for stat in stats], index=areas.index
        )
        logger.info(
            f"Counted population with access in less than {lvl}mn (total={metrics[lvl].sum()})"
        )

    return metrics


if __name__ == "__main__":
    cli()
