import base64
import json
import os
import tempfile

import numpy as np
import pytest
import rasterio
from accessibility import cli
from click.testing import CliRunner


def test_accessibility():

    data_dir = os.path.join(os.path.dirname(__file__), "data", "malawi")

    with open(os.path.join(data_dir, "accessibility.json")) as f:
        config = json.load(f)

    with tempfile.TemporaryDirectory(prefix="AccessMod_") as tmp_dir:

        config["output-dir"] = tmp_dir

        # update file paths in config
        for layer in ("land-cover", "transport-network", "water", "health-facilities"):
            config[layer]["path"] = os.path.join(
                data_dir, "input", config[layer]["path"]
            )
        config["stack"]["path"] = os.path.join(tmp_dir, config["stack"]["path"])
        config["barriers"][0]["path"] = os.path.join(
            data_dir, "input", config[layer]["path"]
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "accessibility",
                "--config",
                base64.b64encode(json.dumps(config).encode()).decode(),
            ],
        )

        assert result.exit_code == 0

        with rasterio.open(os.path.join(tmp_dir, "stack.tif")) as src:

            data = src.read(1)
            assert list(np.unique(data)) == [0, 1, 2, 3, 1001, 1002, 1003, 2000, 3000]
            assert np.count_nonzero(data) > 20000

        with rasterio.open(os.path.join(tmp_dir, "cumulative_cost.tif")) as src:

            data = src.read(1, masked=True)
            assert data.min() == 0
            assert data.max() == 359
            assert data.mean() == pytest.approx(150, abs=50)
