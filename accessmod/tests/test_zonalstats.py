import base64
import json
import os
import tempfile

from click.testing import CliRunner
from zonalstats import cli


def test_zonalstats():

    data_dir = os.path.join(os.path.dirname(__file__), "data", "malawi")

    with open(os.path.join(data_dir, "zonal_stats.json")) as f:
        config = json.load(f)

    with tempfile.TemporaryDirectory(prefix="AccessMod_") as tmp_dir:

        config["output_dir"] = tmp_dir

        # update file paths in config
        for layer in ("travel_times", "population", "boundaries"):
            config[layer]["path"] = os.path.join(
                data_dir, "input", config[layer]["path"]
            )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "zonalstats",
                "--config",
                base64.b64encode(json.dumps(config).encode()).decode(),
            ],
        )

        assert result.exit_code == 0
