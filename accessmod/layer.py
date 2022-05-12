"""Input, intermediary and output geographic layers."""
import logging
import os
import tempfile
from enum import Enum
from functools import cached_property
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import production  # noqa
import rasterio
from errors import AccessModError
from processing import enforce_crs
from pyproj import CRS
from rasterio.features import rasterize as rio_rasterize
from utils import filesystem

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Format(Enum):
    """Layer file format."""

    VECTOR = 1
    RASTER = 2
    TABULAR = 3


class Role(Enum):
    """Layer role."""

    BARRIER = 1
    DEM = 2
    HEALTH_FACILITIES = 3
    LAND_COVER = 4
    POPULATION = 5
    TRANSPORT_NETWORK = 6
    TRAVEL_TIMES = 7
    WATER = 8
    STACK = 9
    BOUNDARIES = 10


class Layer:
    """Geographic layer."""

    def __init__(
        self, filepath: str, role: Role, format: Format, name: str, cache: bool = True
    ):
        """Geographic layer.

        Parameters
        ----------
        filepath : str
            Path to data file (existing or not).
        role : Role
            Layer role.
        format : Format
            Layer format.
        name : str
            Layer name.
        cache : bool, optional
            Use a cached filesystem (default=True).
        """
        self.filepath = filepath
        self.role = role
        self.format = format
        self.name = name
        self.fs = filesystem(self.filepath, cache=cache)

    def __str__(self):
        return f"Layer({self.role}, {self.filepath})"

    def __eq__(self, other):
        return self.filepath == other.filepath and self.role == other.role

    def exists(self):
        """File exists."""
        return self.fs.exists(self.filepath)

    def read_metadata(self) -> dict:
        """Read raster metadata."""
        if not self.exists():
            raise FileNotFoundError(f"File {self.filepath} does not exist.")
        if not self.format == Format.RASTER:
            raise ValueError(f"File {self.filepath} is not a raster.")

        with self.fs.open(self.filepath) as f:
            with rasterio.open(f) as src:
                meta = src.meta
                meta["shape"] = (meta["height"], meta["width"])
        return meta

    def read(self) -> Union[np.ndarray, pd.DataFrame, gpd.GeoDataFrame]:
        """Read raster, vector or tabular data."""
        if not self.exists():
            raise FileNotFoundError(f"File {self.filepath} does not exist.")

        if self.format == Format.VECTOR:
            extension = self.filepath.split(".")[-1]
            with tempfile.NamedTemporaryFile(suffix=extension) as tmp:
                with self.fs.open(self.filepath) as f:
                    tmp.write(f.read())
                return gpd.read_file(tmp.name)

        with self.fs.open(self.filepath, "rb") as f:

            if self.format == Format.TABULAR:
                extension = os.path.basename(self.filepath).split(".")[-1].lower()
                if extension == "csv":
                    return pd.read_csv(f)
                elif extension in ("xls", "xlsx"):
                    return pd.read_excel(f)

            elif self.format == Format.RASTER:
                with rasterio.open(f) as src:
                    return src.read(1)

            else:
                raise ValueError("Unrecognized file format.")


class ElevationLayer(Layer):
    """Elevation/DEM layer."""

    def __init__(self, filepath: str, name: str = "dem", **kwargs):
        super().__init__(
            filepath=filepath, role=Role.DEM, format=Format.RASTER, name=name, **kwargs
        )


class TransportNetworkLayer(Layer):
    """Transport network layer."""

    def __init__(
        self,
        filepath: str,
        category_column: str,
        name: str = "transport-network",
        **kwargs,
    ):
        super().__init__(
            filepath=filepath,
            role=Role.TRANSPORT_NETWORK,
            format=Format.VECTOR,
            name=name,
            **kwargs,
        )
        self.category_column = category_column

    @cached_property
    def data(self):
        """Read vector data."""
        return self.read()

    @cached_property
    def unique(self):
        """Get unique values in category column."""
        return list(self.data[self.category_column].unique())

    @property
    def default_labels(self) -> Dict[int, str]:
        """Default labels from unique values."""
        return {i + 1: category for i, category in enumerate(self.unique)}

    @property
    def labels(self):
        return self.default_labels

    def class_value(self, class_label: str) -> int:
        """Get transport network category integer value based on a label."""
        for value, label in self.labels.items():
            if label.lower() == class_label.lower():
                return value

    def rasterize(
        self,
        category: str,
        dst_transform: rasterio.Affine,
        dst_shape: Tuple[int, int],
        dst_crs: CRS,
        all_touched: bool = True,
    ) -> np.ndarray:
        """Rasterize transport network vector layer.

        Parameters
        ----------
        category : str
            Network category to rasterize (e.g. "residential").
        dst_transform : rasterio Affine
            Target affine transform.
        dst_shape : tuple of int
            Target raster shape.
        dst_crs : pyproj CRS
            Target CRS.
        order : list of str, optional
            Rasterization order. First categories have priority.
        all_touched : bool, optional
            Burn all pixels intersecting the vector features (default=True).

        Return
        ------
        ndarray
            Output raster.
        """
        network = self.data.copy()
        network = enforce_crs(network, dst_crs)
        network = network[network[self.category_column] == category]
        shapes = [geom.__geo_interface__ for geom in network.geometry]

        if len(shapes) == 0:
            return None

        return rio_rasterize(
            shapes=shapes,
            out_shape=dst_shape,
            fill=0,
            transform=dst_transform,
            all_touched=all_touched,
            default_value=1,
            dtype="uint8",
        )


class LandCoverLayer(Layer):
    """Land cover layer."""

    def __init__(
        self,
        filepath: str,
        labels: Dict[int, str] = None,
        name: str = "land-cover",
        **kwargs,
    ):
        """Land cover layer."""
        super().__init__(
            filepath=filepath,
            role=Role.LAND_COVER,
            format=Format.RASTER,
            name=name,
            **kwargs,
        )
        if labels:
            self.labels = {int(v): label for v, label in labels.items()}
        else:
            self.labels = self.default_labels

    @cached_property
    def unique(self) -> List[int]:
        """Get unique values in raster."""
        with self.fs.open(self.filepath) as f:
            with rasterio.open(f) as src:
                unique_values = list(np.unique(src.read(1)))
                unique_values = [v for v in unique_values if v != src.nodata]
        return unique_values

    @property
    def default_labels(self) -> Dict[int, str]:
        """Default labels from unique values."""
        return {value: str(value) for value in self.unique}

    def initialize_labels(self, labels) -> Dict[int, str]:
        """Initialize land cover labels.

        If labels have been provided by the user, they are used as they are
        after removing classes not found in the raster data. If no labels have
        been provided, create default ones from the default numeric values in
        the raster.
        """
        with self.fs.open(self.filepath) as f:
            with rasterio.open(f) as src:
                unique_values = list(np.unique(src.read(1)))
                unique_values = [v for v in unique_values if v != src.nodata]

        if labels:
            return {id_: label for id_, label in labels.items() if id_ in unique_values}
        else:
            return {id_: str(id_) for id_ in unique_values}

    def class_value(self, class_label: str) -> int:
        """Get land cover category integer value based on a label."""
        for value, label in self.labels.items():
            if label == class_label:
                return value
        return None

    @cached_property
    def meta(self) -> dict:
        """Read raster metadata."""
        with self.fs.open(self.filepath) as f:
            with rasterio.open(f) as src:
                return src.meta

    def read(self) -> np.ndarray:
        """Read raster layer as a numpy ndarray."""
        with self.fs.open(self.filepath) as f:
            with rasterio.open(f) as src:
                return src.read(1)


class BarrierLayer(Layer):
    """Barrier layer."""

    def __init__(
        self, filepath: str, all_touched: bool = True, name: str = "barrier", **kwargs
    ):
        """Barrier layer."""
        super().__init__(
            filepath=filepath,
            name=name,
            role=Role.BARRIER,
            format=Format.VECTOR,
            **kwargs,
        )
        self.all_touched = all_touched

    def read(self):
        """Read vector layer."""
        if not self.exists():
            raise FileNotFoundError(f"File {self.filepath} does not exist.")
        with self.fs.open(self.filepath) as f:
            return gpd.read_file(f)

    def rasterize(
        self,
        dst_transform: rasterio.Affine,
        dst_shape: Tuple[int, int],
        dst_crs: CRS,
    ) -> np.ndarray:
        """Rasterize barrier vector layer.

        Parameters
        ----------
        dst_transform : rasterio Affine
            Target affine transform.
        dst_shape : tuple of int
            Target raster shape.
        dst_crs : pyproj CRS
            Target CRS.

        Return
        ------
        ndarray
            Output raster.
        """
        features = self.read()
        features = enforce_crs(features, dst_crs)
        shapes = [geom.__geo_interface__ for geom in features.geometry]
        if len(shapes) == 0:
            return None

        return rio_rasterize(
            shapes=shapes,
            out_shape=dst_shape,
            fill=0,
            transform=dst_transform,
            all_touched=self.all_touched,
            default_value=1,
            dtype="uint8",
        )


class WaterLayer(Layer):
    """Water layer."""

    def __init__(
        self, filepath: str, all_touched: bool = True, name: str = "water", **kwargs
    ):
        """Water layer."""
        super().__init__(
            filepath=filepath,
            role=Role.WATER,
            format=Format.VECTOR,
            name=name,
            **kwargs,
        )
        self.all_touched = all_touched

    def read(self):
        """Read vector layer."""
        if not self.exists():
            raise FileNotFoundError(f"File {self.filepath} does not exist.")
        with self.fs.open(self.filepath) as f:
            return gpd.read_file(f)

    def rasterize(
        self,
        dst_transform: rasterio.Affine,
        dst_shape: Tuple[int, int],
        dst_crs: CRS,
    ) -> np.ndarray:
        """Rasterize water vector layer.

        Parameters
        ----------
        dst_transform : rasterio Affine
            Target affine transform.
        dst_shape : tuple of int
            Target raster shape.
        dst_crs : pyproj CRS
            Target CRS.

        Return
        ------
        ndarray
            Output raster.
        """
        features = self.read()
        features = enforce_crs(features, dst_crs)
        shapes = [geom.__geo_interface__ for geom in features.geometry]

        if len(shapes) == 0:
            return None

        return rio_rasterize(
            shapes=shapes,
            out_shape=dst_shape,
            fill=0,
            transform=dst_transform,
            all_touched=self.all_touched,
            default_value=1,
            dtype="uint8",
        )


class StackLayer(Layer):
    """Land cover stack layer."""

    def __init__(
        self,
        filepath: str,
        layers: List[Layer] = None,
        priorities: List[dict] = None,
        labels: Dict[int, str] = None,
        name: str = "stack",
        **kwargs,
    ):
        """Land cover stack layer.

        A stack layer can be initialized from an existing file. In this case, a
        dictionnary with class labels must also be provided.

        A stack layer can also be initialized from multiple existing layers. In
        that case, at least a land cover layer must be provided and a priorities
        dictionnary must be provided to order priorities between layers and
        classes.

        Parameters
        ----------
        filepath : str
            Path to output raster.
        name : str
            Arbitrary layer name.
        layers : list of Layer, optional
            List of input geographic layers. Only barrier layers may be provided
            multiple times and at least the land cover layer is required. Not
            required if the stack layer already exists at file path.
        priorities : list of dict, optional
            Rasterization priorities between layers and classes. Required if the
            stack layer is initialized from existing layers. Each element of the
            priorities list is a dict such as:
            `{"name": <layer_name>, "class": <class_label>}`
        labels : dict, optional
            Class labels of the provided stack layer. Not required if the stack
            layer is initialized from existing layers.
        """
        super().__init__(
            filepath=filepath,
            role=Role.STACK,
            format=Format.RASTER,
            name=name,
            **kwargs,
        )
        self.fs = filesystem(filepath)

        if not self.exists and not layers:
            raise AccessModError(
                "Stack layer does not exist and no layer have been provided."
            )
        if not self.exists and not priorities:
            raise AccessModError(
                "Stack layer does not exist and no stack priorities have been provided."
            )
        if self.exists and not labels:
            raise AccessModError("Missing class labels for stack layer.")

        self.layers = layers
        self.priorities = self.expand_priorities(priorities)
        if labels:
            self.labels = {int(v): label for v, label in labels.items()}
        else:
            self.labels = {}

    @property
    def exists(self) -> bool:
        """Stack layer exists at the provided file path."""
        return self.fs.exists(self.filepath)

    @cached_property
    def meta(self):
        """Raster metadata."""
        # if stack layers exists on disk, read metadata from raster file
        if self.exists:
            with self.fs.open(self.filepath) as f:
                with rasterio.open(f) as src:
                    meta = src.meta

        # if stack layer does not exist, read metadata from first available
        # layer
        else:
            if not self.layers:
                raise AccessModError("Stack does not have any layer.")
            fp = None
            for layer in self.layers:
                if layer.format == Format.RASTER:
                    fp = layer.filepath
            if not fp:
                raise AccessModError("No raster layer found in stack.")
            fs = filesystem(fp)
            with fs.open(fp) as f:
                with rasterio.open(f) as src:
                    meta = src.meta

        meta["shape"] = (meta.get("height"), meta.get("width"))
        return meta

    @property
    def roles(self):
        """List of available layer roles."""
        return [layer.role for layer in self.layers]

    @property
    def names(self):
        """List of layer names."""
        return [layer.name for layer in self.layers]

    def find(self, role: Role) -> List[str]:
        """Find layers by role."""
        return [layer.name for layer in self.layers if layer.role.value == role.value]

    def get(self, name: str) -> Layer:
        """Get layer by name."""
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    def expand_priorities(self, priorities: List[dict]) -> List[dict]:
        """Expand class priorities dictionary.

        If an element of the priorities list refers to the land cover or
        transport network layer without providing a class label, then it is
        automatically expanded to all the missing class labels for the specified
        layer.
        """
        # layer is not available in the current stack
        for stack_class in priorities:
            if stack_class["name"] not in self.names:
                priorities.remove(stack_class)

        # class is not available in the specified layer
        for stack_class in priorities:
            layer = self.get(stack_class["name"])
            if (
                layer.role.value
                in (Role.LAND_COVER.value, Role.TRANSPORT_NETWORK.value)
                and "class" in stack_class
            ):
                if stack_class["class"] not in layer.labels.values():
                    priorities.remove(stack_class)

        # expand land cover and transport network class labels if needed
        for i, stack_class in enumerate(priorities):
            layer = self.get(stack_class["name"])
            if layer.role.value in (
                Role.LAND_COVER.value,
                Role.TRANSPORT_NETWORK.value,
            ) and not stack_class.get("class"):
                # class labels that are already present in the priorities
                present = [
                    stack_class["class"]
                    for stack_class in priorities
                    if stack_class["name"] == layer.name and "class" in stack_class
                ]
                # class labels that are missing from the priorities
                missing = [
                    label for label in layer.labels.values() if label not in present
                ]
                for label in reversed(missing):
                    priorities.insert(i, {"name": layer.name, "class": label})
                priorities.remove(stack_class)

        return priorities

    def class_value(self, class_label: str) -> int:
        """Get class ID (i.e. pixel value) from class label.

        It is assumed that class label is not present in multiple
        layers at the same time. The function looks for a matching string
        in the land cover and transport network layers. It is assumed that
        class label is not present in multiple layers at the same time.

        Search is case-insensitive.

        Parameters
        ----------
        class_label : str
            Class label (e.g. Bare, Built-Up, primary, residential...).

        Return
        ------
        int
            Class ID (pixel value).
        """
        for value, label in self.labels.items():
            if label.lower() == class_label.lower():
                return value
        raise ValueError(f"Class {class_label} not found in layer.")

    def merge(self) -> np.ndarray:
        """Merge geographic layers into a single raster.

        Merging priorities between overlapping layers is provided through the
        `priorities` attribute at the class level.

        Raster values:
            * 0: no data
            * 1-999 range: land cover classes
            * 1000-1999: transport network classes
            * 2000: water cells
            * 3000-2999: barrier classes
        """
        logger.info("Started merge of input layers")
        land_cover = self.get(self.find(Role.LAND_COVER)[0]).read()
        stack = np.zeros(shape=self.meta.get("shape"), dtype="int16")
        self.labels = {}

        # 1 will be added each time we iterate over a barrier layer
        i_barrier = 0

        # populate the stack raster class per class starting from the lowest
        # priority
        for stack_class in reversed(self.priorities):

            layer = self.get(stack_class["name"])
            label = stack_class.get("class")
            if not layer:
                raise AccessModError(f"Layer '{layer}' not found.")

            if layer.role.value == Role.LAND_COVER.value:
                class_value = layer.class_value(label)
                stack[land_cover == class_value] = land_cover[land_cover == class_value]
                self.labels[class_value] = label

            elif layer.role.value == Role.TRANSPORT_NETWORK.value:
                data = layer.rasterize(
                    label,
                    self.meta.get("transform"),
                    self.meta.get("shape"),
                    self.meta.get("crs"),
                )
                if data is not None:
                    class_value = layer.class_value(label) + 1000
                    stack[data == 1] = class_value
                    self.labels[class_value] = label

            elif layer.role.value == Role.WATER.value:
                data = layer.rasterize(
                    self.meta.get("transform"),
                    self.meta.get("shape"),
                    self.meta.get("crs"),
                )
                if data is not None:
                    stack[data == 1] = 2000
                    self.labels[2000] = layer.name

            elif layer.role.value == Role.BARRIER.value:
                data = layer.rasterize(
                    self.meta.get("transform"),
                    self.meta.get("shape"),
                    self.meta.get("crs"),
                )
                if data is not None:
                    stack[data == 1] = 3000 + i_barrier
                    self.labels[3000 + i_barrier] = layer.name
                    i_barrier += 1

            else:
                raise AccessModError("Unrecognized layer.")

        logger.info(f"Merged {len(self.priorities)} classes into stack layer")

        return stack

    def write(self, overwrite=False):
        """Write stack layer to disk."""
        self.fs.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if self.fs.exists(self.filepath) and not overwrite:
            raise FileExistsError(f"File {self.filepath} already exists.")

        data = self.merge()
        meta = self.meta.copy()
        meta.update(dtype="int16", compress="zstd", nodata=0)

        with self.fs.open(self.filepath, "wb") as f:
            with rasterio.open(f, "w", **meta) as dst:
                dst.write(data, 1)

        logger.info(f"Written stack layer to {self.filepath}")

    def read(self) -> np.ndarray:
        """Read raster layer as a numpy ndarray."""
        with self.fs.open(self.filepath) as f:
            with rasterio.open(f) as src:
                return src.read(1)


class HealthFacilitiesLayer(Layer):
    """Health facilities layer."""

    def __init__(self, filepath: str, name: str = "health-facilities", **kwargs):
        """Health facilities layer."""
        super().__init__(
            filepath=filepath,
            role=Role.HEALTH_FACILITIES,
            format=Format.VECTOR,
            name=name,
            **kwargs,
        )

    def read(self):
        """Read vector layer."""
        if not self.exists():
            raise FileNotFoundError(f"File {self.filepath} does not exist.")
        with self.fs.open(self.filepath) as f:
            return gpd.read_file(f)


class TravelTimesLayer(Layer):
    """Travel times layer."""

    def __init__(self, filepath: str, name: str = "travel-times", **kwargs):
        """Travel times layer."""
        super().__init__(
            filepath=filepath,
            role=Role.TRAVEL_TIMES,
            format=Format.RASTER,
            name=name,
            **kwargs,
        )

    def read(self) -> np.ndarray:
        """Read raster layer as a numpy ndarray."""
        with self.fs.open(self.filepath) as f:
            with rasterio.open(f) as src:
                return src.read(1)

    @cached_property
    def meta(self) -> dict:
        """Read raster metadata."""
        with self.fs.open(self.filepath) as f:
            with rasterio.open(f) as src:
                meta = src.meta
        meta["shape"] = (meta["height"], meta["width"])
        return meta


class BoundariesLayer(Layer):
    """Boundaries layer."""

    def __init__(self, filepath: str, name: str = "boundaries", **kwargs):
        """Boundaries layer."""
        super().__init__(
            filepath=filepath,
            role=Role.BOUNDARIES,
            format=Format.RASTER,
            name=name,
            **kwargs,
        )

    def read(self):
        """Read vector layer."""
        if not self.exists():
            raise FileNotFoundError(f"File {self.filepath} does not exist.")
        with self.fs.open(self.filepath) as f:
            return gpd.read_file(f)


class PopulationLayer(Layer):
    """Population layer."""

    def __init__(self, filepath: str, name: str = "population", **kwargs):
        """Population layer."""
        super().__init__(
            filepath=filepath,
            role=Role.POPULATION,
            format=Format.RASTER,
            name=name,
            **kwargs,
        )

    def read(self) -> np.ndarray:
        """Read raster layer as a numpy ndarray."""
        with self.fs.open(self.filepath) as f:
            with rasterio.open(f) as src:
                return src.read(1, masked=True)

    @cached_property
    def meta(self) -> dict:
        """Read raster metadata."""
        with self.fs.open(self.filepath) as f:
            with rasterio.open(f) as src:
                meta = src.meta
        meta["shape"] = (meta["height"], meta["width"])
        return meta
