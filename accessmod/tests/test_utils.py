import os
import shutil
import tempfile

import pytest
import rasterio
import responses
import utils


@pytest.mark.parametrize("a3,a2", [("BFA", "BF"), ("COD", "CD")])
def test_to_iso_a2(a3, a2):
    assert utils.to_iso_a2(a3) == a2


@responses.activate
def test_country_geometry():

    with open(
        os.path.join(
            os.path.dirname(__file__), "responses", "DJ-region-10m-4326-2010.geojson"
        )
    ) as f:
        responses.add(
            responses.GET,
            url="https://gisco-services.ec.europa.eu/distribution/v2/countries/distribution/DJ-region-01m-4326-2020.geojson",
            body=f.read(),
            status=200,
        )

    geom = utils.country_geometry("DJI", use_cache=False)
    assert geom.centroid.x == pytest.approx(42.579, rel=0.01)
    assert geom.centroid.y == pytest.approx(11.731, rel=0.01)


def test_unzip():

    with tempfile.TemporaryDirectory(prefix="AccessMod_") as tmp_dir:

        # copy zip tile to tmp dir
        FNAME = "N10E041.SRTMGL1.zip"
        src_dir = os.path.join(os.path.dirname(__file__), "data", "srtm")
        fp = os.path.join(tmp_dir, FNAME)
        shutil.copyfile(os.path.join(src_dir, FNAME), fp)

        files = utils.unzip(
            src=fp, dst_dir=tmp_dir, remove_archive=False, overwrite=False
        )

        # check if output raster is readable
        with rasterio.open(files[0]) as src:
            assert src.read(1).any()


def test_unzip_all():

    with tempfile.TemporaryDirectory(prefix="AccessMod_") as tmp_dir:

        # copy zip tiles to tmp dir
        src_dir = os.path.join(os.path.dirname(__file__), "data", "srtm")
        for fn in os.listdir(src_dir):
            if fn.endswith(".zip"):
                src_fp = os.path.join(src_dir, fn)
                fp = os.path.join(tmp_dir, fn)
                shutil.copyfile(src_fp, fp)

        files = utils.unzip_all(src_dir=tmp_dir, dst_dir=tmp_dir, remove_archive=False)

        assert len(files) == 6
        with rasterio.open(files[0]) as src:
            assert src.read(1).any()


def test__human_readable_size():

    assert utils._human_readable_size(1024) == "1.0 KB"
    assert utils._human_readable_size(650850471) == "650.9 MB"
