"""Accessibility analysis pipeline."""

import logging
import os
import shutil
import tempfile
from enum import Enum
from functools import cached_property
from typing import Dict, Tuple

import click
import grasshelper
import numpy as np
import production  # noqa
import rasterio
from appdirs import user_cache_dir
from errors import AccessModError
from layer import (
    BarrierLayer,
    ElevationLayer,
    LandCoverLayer,
    StackLayer,
    TransportNetworkLayer,
    WaterLayer,
)
from utils import filesystem, parse_config, random_string, status_update, upload_file

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


APP_NAME = "AccessMod"
APP_AUTHOR = "Bluesquare"

DEFAULT_LANDCOVER_LABELS = {
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


DEFAULT_SPEEDS = {
    "110": 2,
    "120": 2,
    "20": 3,
    "30": 3,
    "90": 1,
    "100": 3,
    "60": 3,
    "40": 3,
    "50": 4,
    "70": 0,
    "80": 0,
    "200": 0,
    "primary": 70,
    "primary_link": 70,
    "secondary": 50,
    "secondary_link": 50,
    "tertiary": 30,
    "tertiary_link": 30,
    "trunk": 60,
    "trunk_link": 60,
    "unclassified": 15,
    "residential": 15,
    "living_street": 10,
    "service": 10,
    "track": 10,
    "path": 10,
}


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
def accessibility(config: str):
    """Perform an accessibility analysis."""

    status_update(
        status="RUNNING",
        data={},
        url=os.environ.get("HEXA_WEBHOOK_URL"),
        token=os.environ.get("HEXA_WEBHOOK_TOKEN"),
    )

    config = parse_config(config)
    logger.info("Parsed JSON configuration")

    dem = ElevationLayer(filepath=config["dem"]["path"])
    logger.info(f"Using DEM from {config['dem']['path']}")

    # support null stack entry in config
    if not config.get("stack", False):
        config["stack"] = {
            "auto": True,
            "path": os.path.join(config["output_dir"], "stack.tif"),
        }

    # set default category column for transport network layer
    if "transport_network" in config:
        if config["transport_network"]:
            if not config["transport_network"].get("category_column") and config[
                "transport_network"
            ].get("auto"):
                config["transport_network"]["category_column"] = "highway"

    # set default land cover labels
    if "land_cover" in config:
        if config["land_cover"]:
            if not config["land_cover"].get("labels") and config["land_cover"].get(
                "auto"
            ):
                config["land_cover"]["labels"] = DEFAULT_LANDCOVER_LABELS

    # set default moving speeds
    if not config.get("moving_speeds"):
        # transport_network and land_cover are enabled
        if config.get("transport_network") and config.get("land_cover"):
            # and both layers are set to "auto"
            if config["transport_network"].get("auto") and config["land_cover"].get(
                "auto"
            ):
                # we already know the labels in that case so we can set default speeds
                # if they have not been provided by the user
                config["moving_speeds"] = DEFAULT_SPEEDS

    layer = config.get("stack")

    if not layer:

        # stack cannot be null, it must be provided by the user or generated
        # automatically
        raise AccessModError("Stack layer is required")

    else:

        # stack is generated automatically from various input geographic
        # variables
        if layer.get("auto"):

            # stack output path must be provided
            if not layer.get("path"):
                raise AccessModError("Missing output path for stack")

            layers = []

            # land cover (required if stack is not provided)
            layer = config.get("land_cover")
            if not layer:
                raise AccessModError("Missing land cover layer")
            layers.append(
                LandCoverLayer(
                    filepath=layer["path"],
                    labels=layer["labels"],
                    name=layer.get("name", "Land cover"),
                )
            )
            logger.info(f"Using land cover from {layer['path']}")

            # transport network (optional)
            layer = config.get("transport_network")
            if layer:
                layers.append(
                    TransportNetworkLayer(
                        filepath=layer["path"],
                        category_column=layer["category_column"],
                        name=layer.get("name", "Transport network"),
                    )
                )
                logger.info(f"Using transport network from {layer['path']}")

            # water (optional)
            layer = config.get("water")
            if layer:
                layers.append(
                    WaterLayer(
                        filepath=layer["path"],
                        all_touched=layer.get("all_touched", True),
                        name=layer.get("name", "water"),
                    )
                )
                logger.info(f"Using water from {layer['path']}")

            # barriers (optional)
            if "barriers" in config:
                for barrier in config.get("barriers"):
                    layers.append(
                        BarrierLayer(
                            filepath=barrier["path"],
                            all_touched=barrier.get("all_touched", False),
                            name=barrier.get("name"),
                        )
                    )
                    logger.info(f"Using barrier from {barrier['path']}")

            stack = StackLayer(
                filepath=config["stack"]["path"],
                layers=layers,
                priorities=config["priorities"],
                moving_speeds=config["moving_speeds"],
            )
            logger.info(f"Generated stack from {len(layers)} layers")

            stack.write(overwrite=config.get("overwrite"))

        # stack is provided by the user
        else:

            # stack input path must be provided
            if not layer.get("path"):
                raise AccessModError("Missing input path for stack")

            if not layer.get("labels"):
                logger.warn("Missing labels for stack layer")
            stack = StackLayer(filepath=layer["path"], labels=layer["labels"])
            logger.info(f"Using stack from {layer['path']}")

    analysis = AccessibilityAnalysis(
        dem=dem, stack=stack, output_dir=config["output_dir"]
    )

    if config["algorithm"].lower() == "isotropic":
        algorithm = CostDistanceAlgorithm(1)
    elif config["algorithm"].lower() == "anisotropic":
        algorithm = CostDistanceAlgorithm(2)
    else:
        raise AccessModError(
            f"{config['algorithm']} is not a supported cost distance algorithm."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:

        logger.info("Started calculation of friction surface...")
        dst_file = os.path.join(tmp_dir, "friction.tif")
        friction = analysis.friction_surface(
            dst_file=dst_file,
            moving_speeds=config["moving_speeds"],
            unit_meters=algorithm.value == 2,
        )
        logger.info(f"Friction surface written to {dst_file}")

        logger.info("Started cost distance analysis")
        cost, nearest = analysis.cost_distance(
            friction=friction,
            health_facilities=config["health_facilities"]["path"],
            dst_dir=config["output_dir"],
            algorithm=algorithm,
            knight_move=config.get("knight_move"),
            max_cost=config.get("max_travel_time", 360) * 60,
            overwrite=config.get("overwrite"),
        )
        logger.info(f"Travel times written into {config['output_dir']}")

        friction_uri = os.path.join(config["output_dir"], "friction.tif")
        upload_file(friction, friction_uri, overwrite=True)

    status_update(
        status="SUCCESS",
        data={
            "outputs": {
                "travel_times": cost,
                "stack": stack.filepath,
                "stack_labels": stack.labels,
                "friction_surface": friction_uri,
            }
        },
        url=os.environ.get("HEXA_WEBHOOK_URL"),
        token=os.environ.get("HEXA_WEBHOOK_TOKEN"),
    )


class CostDistanceAlgorithm(Enum):
    ISOTROPIC = 1
    ANISOTROPIC = 2


class AccessibilityAnalysis:
    """Accessibility analysis."""

    def __init__(
        self,
        dem: ElevationLayer,
        stack: StackLayer,
        output_dir: str,
    ):
        """Accessibility analysis.

        Parameters
        ----------
        dem : ElevationLayer
            Digital elevation model.
        stack : StackLayer
            Land cover merge.
        health_facilities : HealthFacilitiesLayer
            Layer with target health facilities.
        output_dir : str
            Path to output directory.
        stack_order : list of tuple of (Layer, str), optional
            Priority between stack layers and classes.
        moving_speeds : dict, optional
            Moving speeds (in km/h) for each class in the stack.
        """
        self.dem = dem
        self.stack = stack
        self.output_dir = output_dir
        self.fs = filesystem(output_dir)

    @cached_property
    def meta(self) -> dict:
        """Read raster metadata from stack."""
        return self.stack.meta

    def friction_surface(
        self, dst_file: str, moving_speeds: Dict[str, float], unit_meters: bool = False
    ) -> str:
        """Compute the friction surface.

        Parameters
        ----------
        dst_file : str
            Path to output raster.
        moving_speeds : dict
            Moving speeds (in km/h) for each class in the stack layer.
        unit_meters : bool, optional
            Compute time to cross *one meter* (expected by r.walk) instead of
            time to cross *one pixel* (expected by r.cost) (default=False).

        Return
        ------
        str
            Path to output raster.
        """
        merge = self.stack.read()
        speed = np.zeros(self.stack.meta["shape"], dtype="float32")

        for class_value, class_label in self.stack.labels.items():

            class_value = int(class_value)

            # in the moving speeds dictionary, classes from land cover layer are
            # referred to through their original class ID (e.g. 120, 200, etc.)
            if class_value > 0 and class_value < 1000:
                if str(class_value) in moving_speeds:
                    class_speed = moving_speeds[str(class_value)]
                # if no speed is provided for the land cover class, use a
                # default value and raise a warning
                else:
                    class_speed = 2
                    logger.warn(
                        f"No speed provided for land cover class {class_value}:{class_label}"
                    )

            # in the moving speeds dictionary, classes from transport network
            # layer are reffered to with their original category value (e.g.
            # primary, secondary, residential, etc.)
            elif class_value >= 1000 and class_value < 2000:
                if class_label in moving_speeds:
                    class_speed = moving_speeds[class_label]
                else:
                    # if no speed provided for transport network, raise a
                    # warning and ignore
                    logger.warn(
                        f"No speed provided for transport network {class_label}"
                    )
                    continue

            # set classes from water and barrier layers to null speed
            elif class_value >= 2000:
                class_speed = 0

            else:
                class_speed = np.nan

            speed[merge == class_value] = class_speed

        speed /= 3.6  # from km/h to m/s

        friction = np.empty(shape=merge.shape, dtype="float32")
        if unit_meters:
            # time to cross one meter in seconds
            friction[speed != 0] = 1 / speed[speed != 0]
        else:
            # time to cross one pixel in seconds
            friction[speed != 0] = self.meta.get("transform").a / speed[speed != 0]

        # assign nodata value to invalid pixels
        friction[np.isinf(friction)] = -9999
        friction[friction < 0] = -9999
        friction[speed == 0] = -9999

        # write friction surface to disk as a geotiff
        dst_profile = rasterio.profiles.default_gtiff_profile
        dst_profile.update(
            count=1,
            dtype="float32",
            transform=self.meta.get("transform"),
            crs=self.meta.get("crs"),
            height=self.meta.get("height"),
            width=self.meta.get("width"),
            compress="zstd",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            nodata=-9999,
        )
        fs = filesystem(dst_file)
        with fs.open(dst_file, "wb") as f:
            with rasterio.open(f, "w", **dst_profile) as dst:
                dst.write(friction, 1)

        return dst_file

    def cost_distance(
        self,
        friction: str,
        health_facilities: str,
        dst_dir: str,
        algorithm: CostDistanceAlgorithm = CostDistanceAlgorithm.ISOTROPIC,
        knight_move: bool = True,
        max_cost: float = 21600,
        walk_coeff: Tuple[float] = (0.72, 6.0, 1.9998, -1.9998),
        lambda_: float = 1.0,
        slope_factor: float = -0.2125,
        overwrite: bool = False,
    ) -> Tuple[str, str]:
        """Perform cost distance analysis.

        Based on GRASS GIS modules r.cost
        <https://grass.osgeo.org/grass78/manuals/r.cost.html>
        and r.walk
        <https://grass.osgeo.org/grass78/manuals/r.walk.html>.

        Parameters
        ----------
        friction : str
            Path to friction surface raster.
        health_facilities : str
            Path to destination points layer.
        dst_dir : str
            Output directory.
        algorithm : CostDistanceAlgorithm, optional
            Isotropic (r.cost) or anisotropic (r.walk). Isotropic
            by default.
        knight_move : bool, optional
            Use the "Knight's move" (default=True).
            See r.cost documentation for more info.
        max_cost : float, optional
            Max cumulative cost (seconds). Set to 0 to ignore (default=21600).
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
            f"grassdata_{random_string(16)}",
        )
        os.makedirs(grass_datadir, exist_ok=True)
        logger.debug(f"GRASS data directory: {grass_datadir}.")

        if algorithm.value == CostDistanceAlgorithm.ANISOTROPIC.value and not self.dem:
            raise AccessModError("A DEM must be provided for anisotropic modeling.")

        fs = filesystem(friction)
        tmp_friction = os.path.join(grass_datadir, os.path.basename(friction))
        fs.get(friction, tmp_friction)
        logger.debug(f"Local copy from {friction} to {tmp_friction}.")

        fs = filesystem(health_facilities)
        tmp_targets = os.path.join(grass_datadir, os.path.basename(health_facilities))
        fs.get(health_facilities, tmp_targets)
        logger.debug(f"Local copy from {health_facilities} to {tmp_targets}.")

        with rasterio.open(tmp_friction) as src:
            src_crs = src.crs
            xres = src.transform.a

        grasshelper.setup_environment(grass_datadir, src_crs)
        grasshelper.grass_execute("r.in.gdal", input=tmp_friction, output="friction")
        grasshelper.grass_execute("g.region", raster="friction", align="friction")
        grasshelper.grass_execute("v.in.ogr", input=tmp_targets, output="targets")

        logger.info("Loaded input data into GRASS environment")

        if algorithm.value == CostDistanceAlgorithm.ANISOTROPIC.value:

            fs = filesystem(self.dem.filepath)
            tmp_dem = os.path.join(grass_datadir, os.path.basename(self.dem.filepath))
            fs.get(self.dem.filepath, tmp_dem)
            logger.debug(f"Local copy from {self.dem} to {tmp_dem}.")

            grasshelper.grass_execute("r.in.gdal", input=tmp_dem, output="dem")

            logger.info("Started anisotropic cost distance analysis")

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

        elif algorithm.value == CostDistanceAlgorithm.ISOTROPIC.value:

            logger.info("Started isotropic cost distance analysis")

            grasshelper.grass_execute(
                "r.cost",
                flags=f'{"k" if knight_move else ""}n',
                input="friction",
                output="cost",
                nearest="nearest",
                outdir="backlink",
                start_points="targets",
            )

        else:
            raise AccessModError(f"Algorithm not supported: {algorithm.name}.")

        logger.info("Finished cost distance analysis.")

        fs = filesystem(dst_dir)
        fs.makedirs(dst_dir, exist_ok=True)

        cost_fp = os.path.join(dst_dir, "cumulative_cost.tif")
        cost_tmp = os.path.join(grass_datadir, "cumulative_cost.tif")
        if fs.exists(cost_fp) and not overwrite:
            raise FileExistsError(f"File {cost_fp} already exists.")

        grasshelper.grass_execute(
            "r.out.gdal",
            flags="f",
            input="cost",
            output=cost_tmp,
            format="GTiff",
            nodata=-1,
            type="Float32",
            createopt="COMPRESS=ZSTD,PREDICTOR=2",
            overwrite=overwrite,
        )

        # convert travel times to minutes and add max cost threshold
        with rasterio.open(cost_tmp) as src:
            meta = src.profile.copy()
            cost = src.read(1)
        cost /= 60
        cost[cost < 0] = -1
        cost[cost > (max_cost / 60)] = -1
        meta.update(dtype="int16")
        cost_tmp_minutes = os.path.join(grass_datadir, "cumulative_cost_minutes.tif")
        with rasterio.open(cost_tmp_minutes, "w", **meta) as dst:
            dst.write(cost.astype("int16"), 1)

        fs.put(cost_tmp_minutes, cost_fp)
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

        shutil.rmtree(grass_datadir)

        return cost_fp, nearest_fp


if __name__ == "__main__":
    cli()
