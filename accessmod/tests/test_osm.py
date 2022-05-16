import base64
import json
import os
import tempfile

import geopandas as gpd
from click.testing import CliRunner
from osm import cli


def test_osm():

    data_dir = os.path.join(os.path.dirname(__file__), "data", "dji")

    with open(os.path.join(data_dir, "config_osm.json")) as f:
        config = json.load(f)

    with tempfile.TemporaryDirectory(prefix="AccessMod_") as tmp_dir:

        # update file paths in config
        for layer in ("transport_network", "water"):
            config[layer]["path"] = os.path.join(data_dir, "osm", config[layer]["path"])

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "extract-from-osm",
                "--config",
                base64.b64encode(json.dumps(config).encode()).decode(),
            ],
        )

        assert result.exit_code == 0

        roads = gpd.read_file(config["transport_network"]["path"])
        assert len(roads) > 10

        rivers = gpd.read_file(config["water"]["path"])
        assert len(rivers) > 10
