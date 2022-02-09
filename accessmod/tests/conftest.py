import json
import os

import pytest
from shapely.geometry import shape


@pytest.fixture
def djibouti_geom():
    with open(
        os.path.join(
            os.path.dirname(__file__), "data", "DJ-region-10m-4326-2020.geojson"
        )
    ) as f:
        return shape(json.load(f)["features"][0]["geometry"])
