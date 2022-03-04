"""Accessibility analysis pipeline."""

import logging
import os
from typing import List, Tuple

import click
import geopandas as gpd
import grasshelper
import numpy as np
import pandas as pd
import processing
import rasterio
import rasterio.features
import utils
from appdirs import user_cache_dir
from pyproj import CRS

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
@click.option("--output-dir", type=str, required=True, help="output data directory")
@click.option("--dem", type=str, help="digital elevation model")
@click.option("--slope", type=str, help="slope raster")
@click.option("--max-slope", type=float, help="max passable slope in percents")
@click.option("--land-cover", type=str, help="land cover raster")
@click.option("--transport-network", type=str, help="transport network layer")
@click.option("--priority-roads", is_flag=True, help="roads have priority over water")
@click.option(
    "--priority-land-cover",
    type=str,
    help="land cover classes with priority over water",
)
@click.option("--barrier", type=str, help="barrier layer")
@click.option("--water", type=str, help="water layer")
@click.option("--water-all-touched", is_flag=True, help="mask all cells touching water")
@click.option("--moving-speeds", type=str, help="path to scenario table")
@click.option(
    "--category-column",
    type=str,
    default="highway",
    help="category column in transport layer",
)
@click.option(
    "--algorithm",
    type=click.Choice(["isotropic", "anisotropic"], case_sensitive=False),
    help="cost distance algorithm",
)
@click.option(
    "--knight-move",
    is_flag=True,
    default=False,
    help="use knight's move in cost distance analysis",
)
@click.option(
    "--invert-direction", is_flag=True, help="compute travel times from target"
)
@click.option("--max-travel-time", type=int, help="max travel time in minutes")
@click.option(
    "--overwrite", is_flag=True, default=False, help="overwrite existing files"
)
@click.option(
    "--webhook-url",
    type=str,
    help="URL to push a POST request with the analysis' results",
)
@click.option("--webhook-token", type=str, help="Token to use in the webhook POST")
@click.argument("health-facilities")
def accessibility(
    health_facilities: str,
    output_dir: str,
    dem: str,
    slope: str,
    max_slope: float,
    land_cover: str,
    transport_network: str,
    priority_roads: bool,
    priority_land_cover: str,
    barrier: str,
    water: str,
    water_all_touched: bool,
    moving_speeds: str,
    category_column: str,
    algorithm: str,
    knight_move: bool,
    invert_direction: bool,
    max_travel_time: int,
    overwrite: bool,
    webhook_url: str,
    webhook_token: str,
):
    """Perform an accessibility analysis."""
    utils.status_update(
        status="RUNNING",
        data={},
        url=webhook_url,
        token=webhook_token,
    )

    fs = utils.filesystem(output_dir)
    fs.makedirs(output_dir, exist_ok=True)

    fs = utils.filesystem(dem)
    with fs.open(dem) as f:
        with rasterio.open(f) as src:
            dst_crs = src.crs
            dst_shape = src.shape
            dst_transform = src.transform

    fs = utils.filesystem(moving_speeds)
    with fs.open(moving_speeds) as f:
        scenario_table = pd.read_csv(moving_speeds)
        # temporary hack to divide the scenario table into two separate
        # dicts (one for land cover and one for transport network)
        scenario_table["class"] = scenario_table["class"].astype(str)
        class_is_int = scenario_table["class"].apply(lambda cls: cls.isnumeric())
        landcover_speeds = {
            int(row["class"]): row["speed"]
            for _, row in scenario_table[class_is_int].iterrows()
        }
        transport_speeds = {
            row["class"]: row["speed"]
            for _, row in scenario_table[~class_is_int].iterrows()
        }

    # temporary hack to support empty forms
    if barrier == "None":
        barrier = None
    if water == "None":
        water = None

    friction = friction_surface(
        dst_file=os.path.join(output_dir, "friction_surface.tif"),
        src_landcover=land_cover,
        src_landcover_speeds=landcover_speeds,
        dst_crs=dst_crs,
        dst_shape=dst_shape,
        dst_transform=dst_transform,
        src_transport=transport_network,
        src_transport_speeds=transport_speeds,
        src_transport_column=category_column,
        src_slope=slope,
        max_slope=max_slope,
        src_water_vector=water,
        src_water_all_touched=water_all_touched,
        src_barrier=[barrier] if barrier else None,
        unit_meters=algorithm == "anisotropic",
        overwrite=overwrite,
    )

    cost, catchment = isotropic_costdistance(
        src_friction=friction,
        src_targets=health_facilities,
        dst_dir=output_dir,
        knight_move=knight_move,
        overwrite=overwrite,
    )

    utils.status_update(
        status="SUCCESS",
        data={
            "travel_times": cost,
            "friction_surface": friction,
            "catchment_areas": catchment,
        },
        url=webhook_url,
        token=webhook_token,
    )


def speed_from_raster(src_raster: str, moving_speeds: dict) -> np.ndarray:
    """Compute speed raster from a categorical raster.

    In a speed raster, the value of each pixel is equal to speed
    in km/h associated with the location.

    Parameters
    ----------
    src_raster : str
        Path to input categorical raster.
    moving_speeds : dict
        Moving speed in km/h for each categorical value in the
        source raster.

    Return
    ------
    ndarray
        Speed raster as a 2d numpy array.
    """
    fs = utils.filesystem(src_raster)
    with fs.open(src_raster) as f:
        with rasterio.open(f) as src:
            src_shape = (src.height, src.width)
            src_array = src.read(1)
            dst_array = np.zeros(shape=src_shape, dtype=np.int16)
            for category, speed in moving_speeds.items():
                dst_array[src_array == category] = speed
    return dst_array


def speed_from_vector(
    src_vector: str,
    dst_crs: CRS,
    dst_shape: Tuple[int],
    dst_transform: rasterio.Affine,
    moving_speeds: dict,
    category_column: str,
    all_touched: bool = True,
) -> np.ndarray:
    """Compute speed raster from a vector file.

    In a speed raster, the value of each pixel is equal to speed
    in km/h associated with the location.

    Parameters
    ----------
    src_vector : str
        Path to input vector file.
    dst_crs : pyproj crs
        Target coordinate reference system.
    dst_shape : tuple of int
        Target raster shape (nrows, ncols).
    dst_transform : rasterio affine
        Target affine transform.
    moving_speeds : dict
        Moving speed in km/h for each category of feature in the
        input vector file.
    category_column : str
        Column in the input vector file containing category info.
    all_touched : bool, optional
        Burn all pixels touching the geometries (default=True).

    Return
    ------
    ndarray
        Speed raster as a 2d numpy array.
    """
    src_features = gpd.read_file(src_vector)
    src_features = processing.enforce_crs(src_features, dst_crs)

    src_features["speed"] = src_features[category_column].apply(
        lambda cat: moving_speeds.get(cat)
    )
    src_features = src_features[~pd.isna(src_features["speed"])]

    # sorting order of input features is important as it determines
    # which features are going to have the priority when burning
    # the speed value into the output raster
    src_features.sort_values(by="speed", ascending=True)

    shapes = [
        (feature.geometry, feature.speed) for _, feature in src_features.iterrows()
    ]
    logger.debug(f"Found {len(shapes)} features in {src_vector}.")

    return rasterio.features.rasterize(
        shapes=shapes,
        out_shape=dst_shape,
        fill=0,
        transform=dst_transform,
        all_touched=True,
        dtype="int16",
    )


def water_mask(
    dst_crs: CRS,
    dst_shape: Tuple[int],
    dst_transform: rasterio.Affine,
    src_vector: str = None,
    src_raster: str = None,
    all_touched: bool = True,
) -> np.ndarray:
    """Compute the water mask from vector or raster input layer.

    At least one input file (src_vector, src_raster or both) must be provided.
    If both are provided, vector and raster features are merged into the final
    layer.

    Parameters
    ----------
    dst_crs : pyproj crs
        Target coordinate reference system.
    dst_shape : tuple of int
        Target raster shape (nrows, ncols).
    dst_transform : rasterio affine
        Target affine transform.
    src_vector : str, optional
        Path to input vector file with water bodies and rivers.
        Features can be polygons or polylines (default=None).
    src_raster : str, optional
        Path to input raster file with water bodies and rivers.
        Non-null values are considered to be water (default=None).
    all_touched : bool, optional
        Burn all pixels touching the geometries (default=True).

    Return
    ------
    ndarray
        Water mask as a boolean 2d numpy array.
    """
    if not any((src_vector, src_raster)):
        raise ValueError("At least one water data source must be provided.")

    water = np.zeros(shape=dst_shape, dtype=np.uint8)

    if src_raster:
        fs = utils.filesystem(src_raster)
        with fs.open(src_raster) as f:
            with rasterio.open(f) as src:
                water[src.read(1, masked=True) > 0] = 1

    if src_vector:
        fs = utils.filesystem(src_vector)
        with fs.open(src_vector) as f:
            src_features = gpd.read_file(f)
        src_features = processing.enforce_crs(src_features, dst_crs)
        features = rasterio.features.rasterize(
            shapes=[geom.__geo_interface__ for geom in src_features.geometry],
            out_shape=dst_shape,
            transform=dst_transform,
            fill=0,
            default_value=1,
            all_touched=all_touched,
            dtype=np.uint8,
        )
        water[features == 1] = 1

    return water == 1


def slope_mask(src_slope: str, max_slope: float = 45) -> np.ndarray:
    """Compute the mask for high slope locations.

    Parameters
    ----------
    src_slope : str
        Path to input source slope raster.
    max_slope : float, optional
        Max slope value in percents (default=45%).

    Return
    ------
    ndarray
        Slope mask as a boolean 2d numpy array.
    """
    fs = utils.filesystem(src_slope)
    with fs.open(src_slope) as f:
        with rasterio.open(f) as src:
            return src.read(1, masked=True) > max_slope


def apply_barriers(
    src_speed: np.ndarray,
    barriers: List[str],
    dst_crs: CRS,
    dst_shape: Tuple[int],
    dst_transform: rasterio.Affine,
    all_touched: bool = True,
) -> np.ndarray:
    """Assign zero speed to pixels located on an obstacle.

    Barrier features can be either polygons, polylines or both.

    Parameters
    ----------
    src_speed : ndarray
        Input speed raster in km/h.
    barriers : list of str
        Paths to vector filles corresponding to barriers.
    dst_crs : pyproj crs
        Target coordinate reference system.
    dst_shape : tuple of int
        Target raster shape (nrows, ncols).
    dst_transform : rasterio affine
        Target affine transform.
    all_touched : bool, optional
        Apply barrier on all intersecting pixels (default=True).

    Return
    ------
    ndarray
        Output raster as 2d numpy array.
    """
    speed = src_speed.copy()
    for barrier in barriers:
        src_features = gpd.read_file(barrier)
        src_features = processing.enforce_crs(src_features, dst_crs)
        mask = rasterio.features.rasterize(
            shapes=[geom.__geo_interface__ for geom in src_features.geometry if geom],
            fill=0,
            default_value=1,
            all_touched=all_touched,
            out_shape=dst_shape,
            transform=dst_transform,
            dtype="uint8",
        )
        speed[mask == 1] = 0
    return speed


def friction_surface(
    dst_file: str,
    src_landcover: str,
    src_landcover_speeds: dict,
    dst_crs: CRS,
    dst_shape: Tuple[int],
    dst_transform: rasterio.Affine,
    src_transport: str = None,
    src_transport_speeds: dict = None,
    src_transport_column: str = None,
    src_slope: str = None,
    max_slope: float = None,
    src_water_vector: str = None,
    src_water_raster: str = None,
    src_water_all_touched: bool = True,
    src_barrier: List[str] = None,
    unit_meters: bool = False,
    overwrite: bool = False,
) -> np.ndarray:
    """Compute the friction surface.

    Parameters
    ----------
    dst_file : str
        Path to output raster.
    src_landcover : str
        Path to input land cover raster.
    src_landcover_speeds : dict
        Moving speeds in km/h for each land cover category.
        Land cover categories not provided in the dict will
        have a speed value of 0 km/h.
    dst_crs : pyproj crs
        Target coordinate reference system.
    dst_shape : tuple of int
        Target raster shape (nrows, ncols).
    dst_transform : rasterio affine
        Target affine transform.
    src_transport : str, optional
        Path to input transport network vector file.
    src_transport_speeds : dict, optional
        Moving speeds in km/h for transport network category.
        Transport network categories not provided in the dict
        are ignored.
    src_transport_column : str, optional
        Column in src_transport with category information.
    src_slope : str, optional
        Path to input slope raster. Required if max_slope is set.
    max_slope : float, optional
        Max. passable slope value in percents (default=None).
    src_water_vector : str, optional
        Path to input vector file with water bodies and rivers.
        All features are considered as impassable surface water.
    src_water_raster : str, optional
        Path to input raster file with water bodies and rivers.
        All non-null pixels are considered as impassable surface
        water.
    src_water_all_touched : bool, optional
        All pixels intersecting source water features are considered
        as impassable surface water (default=True).
    src_barrier : list of str, optional
        Paths to input vector files with barriers. All features
        are considered as impassable areas.
    unit_meters : bool, optional
        Compute time to cross *one meter* (expected by r.walk) instead of
        time to cross *one pixel* (expected by r.cost) (default=False).
    overwrite : bool, optional
        Overwrite existing files (default=False).


    Return
    ------
    ndarray
        Friction surface as 2d numpy array.
    """
    fs = utils.filesystem(dst_file)
    if fs.exists(dst_file) and not overwrite:
        raise FileExistsError(f"File {dst_file} already exists.")

    off_road = speed_from_raster(src_landcover, src_landcover_speeds)

    if src_water_raster or src_water_vector:
        water = water_mask(
            dst_crs=dst_crs,
            dst_shape=dst_shape,
            dst_transform=dst_transform,
            src_vector=src_water_vector,
            src_raster=src_water_raster,
            all_touched=src_water_all_touched,
        )
        off_road[water] = 0

    if src_slope and max_slope:
        slope = slope_mask(src_slope, max_slope)
        off_road[slope] = 0

    speed = off_road

    if src_transport:
        on_road = speed_from_vector(
            src_vector=src_transport,
            dst_crs=dst_crs,
            dst_shape=dst_shape,
            dst_transform=dst_transform,
            moving_speeds=src_transport_speeds,
            category_column=src_transport_column,
            all_touched=True,
        )
        speed = np.maximum(off_road, on_road)

    if src_barrier:
        speed = apply_barriers(
            src_speed=speed,
            barriers=src_barrier,
            dst_crs=dst_crs,
            dst_shape=dst_shape,
            dst_transform=dst_transform,
            all_touched=True,
        )

    # from km/h to m/s
    speed = speed.astype(np.float32)
    speed = speed / 3.6

    friction = np.empty(shape=dst_shape, dtype=np.float32)
    if unit_meters:
        # time to cross one meter in seconds
        friction[speed != 0] = 1 / speed[speed != 0]
    else:
        # time to cross one pixel in seconds
        friction[speed != 0] = dst_transform.a / speed[speed != 0]

    friction[np.isinf(friction)] = np.nan

    dst_profile = rasterio.profiles.default_gtiff_profile
    dst_profile.update(
        count=1,
        dtype="float32",
        transform=dst_transform,
        crs=dst_crs,
        height=dst_shape[0],
        width=dst_shape[1],
        compress="zstd",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        nodata=-1,
    )
    with fs.open(dst_file, "wb") as f:
        with rasterio.open(f, "w", **dst_profile) as dst:
            dst.write(friction, 1)

    return dst_file


def isotropic_costdistance(
    src_friction: str,
    src_targets: str,
    dst_dir: str,
    knight_move: bool = True,
    overwrite: bool = False,
):
    """Isotropic cost distance analysis.

    Based on r.cost GRASS GIS module, see
    <https://grass.osgeo.org/grass78/manuals/r.cost.html>.

    Parameters
    ----------
    src_friction : str
        Path to friction surface raster.
    src_targets : str
        Path to destination points layer.
    dst_dir : str
        Output directory.
    knight_move : bool, optional
        Use the "Knight's move" (default=True).
        See r.cost documentation for more info.
    """
    grass_datadir = os.path.join(
        user_cache_dir(appname=APP_NAME, appauthor=APP_AUTHOR),
        f"grassdata_{utils.random_string(16)}",
    )
    os.makedirs(grass_datadir, exist_ok=True)
    logger.debug(f"GRASS data directory: {grass_datadir}.")

    fs = utils.filesystem(src_friction)
    tmp_friction = os.path.join(grass_datadir, os.path.basename(src_friction))
    fs.get(src_friction, tmp_friction)
    logger.debug(f"Local copy from {src_friction} to {tmp_friction}.")

    fs = utils.filesystem(src_targets)
    tmp_targets = os.path.join(grass_datadir, os.path.basename(src_targets))
    fs.get(src_targets, tmp_targets)
    logger.debug(f"Local copy from {src_targets} to {tmp_targets}.")

    with rasterio.open(tmp_friction) as src:
        src_crs = src.crs

    grasshelper.setup_environment(grass_datadir, src_crs)
    grasshelper.grass_execute("r.in.gdal", input=tmp_friction, output="friction")
    grasshelper.grass_execute("g.region", raster="friction")
    grasshelper.grass_execute("v.in.ogr", input=tmp_targets, output="targets")

    grasshelper.grass_execute(
        "r.cost",
        flags=f'{"k" if knight_move else ""}n',
        input="friction",
        output="cost",
        nearest="nearest",
        outdir="backlink",
        start_points="targets",
    )
    logger.info("Finished cost distance analysis.")

    fs = utils.filesystem(dst_dir)
    fs.makedirs(dst_dir, exist_ok=True)

    cost_fp = os.path.join(dst_dir, "cumulative_cost.tif")
    tmp = os.path.join(grass_datadir, os.path.basename(cost_fp))
    if fs.exists(cost_fp) and not overwrite:
        raise FileExistsError(f"File {cost_fp} already exists.")
    grasshelper.grass_execute(
        "r.out.gdal",
        flags="f",
        input="cost",
        output=tmp,
        format="GTiff",
        nodata=-1,
        type="Float32",
        createopt="COMPRESS=ZSTD,PREDICTOR=2",
        overwrite=overwrite,
    )
    fs.put(tmp, cost_fp)
    logger.info(f"Exported cumulative cost raster to {cost_fp}.")

    nearest_fp = os.path.join(dst_dir, "catchment_areas.tif")
    tmp = os.path.join(grass_datadir, os.path.basename(nearest_fp))
    if fs.exists(nearest_fp) and not overwrite:
        raise FileExistsError(f"File {nearest_fp} already exists.")
    grasshelper.grass_execute(
        "r.out.gdal",
        input="nearest",
        output=tmp,
        format="GTiff",
        nodata=65535,
        type="UInt16",
        createopt="COMPRESS=ZSTD,PREDICTOR=2",
        overwrite=overwrite,
    )
    fs.put(tmp, nearest_fp)
    logger.info(f"Exported catchment areas raster to {nearest_fp}.")

    return cost_fp, nearest_fp


def anisotropic_costdistance(
    src_friction: str,
    src_targets: str,
    src_dem: str,
    dst_dir: str,
    knight_move: bool = True,
    max_cost: float = 0,
    walk_coeff: Tuple[float] = (0.72, 6.0, 1.9998, -1.9998),
    lambda_: float = 1.0,
    slope_factor: float = -0.2125,
    overwrite: bool = False,
):
    """Anisotropic cost distance analysis.

    Based on r.walk GRASS GIS module, see
    <https://grass.osgeo.org/grass78/manuals/r.walk.html>.

    Parameters
    ----------
    src_friction : str
        Path to friction surface raster.
    src_targets : str
        Path to destination points layer.
    src_dem : str
        Path to digital elevation model.
    dst_dir : str
        Output directory.
    knight_move : bool, optional
        Use the "Knight's move" (default=True).
        See r.cost documentation for more info.
    max_cost : float, optional
        Max cumulative cost (default=0).
    walk_coeff : tuple of float, optional
        Coefficients for walking energy formula parameters a,b,c,d
        (default=(0.72, 6.0, 1.9998, -1.9998)).
    lambda_ : float, optional
        Lambda coefficients for combining walking energy and friction
        cost (default=1.0).
    slope_factor : float, optional
        Slope factor determines travel energy cost per height step
        (default=-0.2125).
    overwrite : bool, optional
        Overwrite existing files (default=False).

    Return
    ------
    str
        Path to cumulative cost raster.
    str
        Path to catchment areas raster.
    """
    grass_datadir = os.path.join(
        user_cache_dir(appname=APP_NAME, appauthor=APP_AUTHOR),
        f"grassdata_{utils.random_string(16)}",
    )
    os.makedirs(grass_datadir, exist_ok=True)
    logger.debug(f"GRASS data directory: {grass_datadir}.")

    fs = utils.filesystem(src_friction)
    tmp_friction = os.path.join(grass_datadir, os.path.basename(src_friction))
    fs.get(src_friction, tmp_friction)
    logger.debug(f"Local copy from {src_friction} to {tmp_friction}.")

    fs = utils.filesystem(src_dem)
    tmp_dem = os.path.join(grass_datadir, os.path.basename(src_dem))
    fs.get(src_dem, tmp_dem)
    logger.debug(f"Local copy from {src_dem} to {tmp_dem}.")

    fs = utils.filesystem(src_targets)
    tmp_targets = os.path.join(grass_datadir, os.path.basename(src_targets))
    fs.get(src_targets, tmp_targets)
    logger.debug(f"Local copy from {src_targets} to {tmp_targets}.")

    with rasterio.open(tmp_friction) as src:
        src_crs = src.crs

    grasshelper.setup_environment(grass_datadir, src_crs)
    grasshelper.grass_execute("r.in.gdal", input=tmp_friction, output="friction")
    grasshelper.grass_execute("r.in.gdal", input=tmp_dem, output="dem")
    grasshelper.grass_execute("g.region", raster="friction")
    grasshelper.grass_execute("v.in.ogr", input=tmp_targets, output="targets")

    grasshelper.grass_execute(
        "r.walk",
        flags=f'{"k" if knight_move else ""}n',
        elevation="dem",
        friction="friction",
        output="cost",
        nearest="nearest",
        outdir="backlink",
        start_points="targets",
    )
    logger.info("Finished cost distance analysis.")

    fs = utils.filesystem(dst_dir)
    fs.makedirs(dst_dir, exist_ok=True)

    cost_fp = os.path.join(dst_dir, "cumulative_cost.tif")
    tmp = os.path.join(grass_datadir, os.path.basename(cost_fp))
    if fs.exists(cost_fp) and not overwrite:
        raise FileExistsError(f"File {cost_fp} already exists.")
    grasshelper.grass_execute(
        "r.out.gdal",
        flags="f",
        input="cost",
        output=tmp,
        format="GTiff",
        nodata=-1,
        type="Float32",
        createopt="COMPRESS=ZSTD,PREDICTOR=2",
        overwrite=overwrite,
    )
    fs.put(tmp, cost_fp)
    logger.info(f"Exported cumulative cost raster to {cost_fp}.")

    nearest_fp = os.path.join(dst_dir, "catchment_areas.tif")
    tmp = os.path.join(grass_datadir, os.path.basename(nearest_fp))
    if fs.exists(nearest_fp) and not overwrite:
        raise FileExistsError(f"File {nearest_fp} already exists.")
    grasshelper.grass_execute(
        "r.out.gdal",
        input="nearest",
        output=tmp,
        format="GTiff",
        nodata=65535,
        type="UInt16",
        createopt="COMPRESS=ZSTD,PREDICTOR=2",
        overwrite=overwrite,
    )
    fs.put(tmp, nearest_fp)
    logger.info(f"Exported catchment areas raster to {nearest_fp}.")

    return cost_fp, nearest_fp


if __name__ == "__main__":
    cli()
