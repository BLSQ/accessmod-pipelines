import os
import re
import tempfile

import pytest
import rasterio
import responses
import srtm
import utils


@pytest.fixture(autouse=True)
def mocked_responses():

    with open(
        os.path.join(os.path.dirname(__file__), "responses", "srtm", "earthdata.html"),
        "rb",
    ) as f:

        responses.add(
            responses.GET,
            url="https://urs.earthdata.nasa.gov",
            body=f.read(),
            status=200,
        )

    responses.add(
        responses.POST, url="https://urs.earthdata.nasa.gov/login", body=b"", status=200
    )

    # SRTM tiles for Djibouti
    for tile in ["N10E041", "N10E042", "N11E041", "N11E042", "N11E043", "N12E042"]:
        fp = os.path.join(
            os.path.dirname(__file__), "data", "srtm", f"{tile}.SRTMGL1.zip"
        )
        with open(fp, "rb") as f:
            responses.add(
                responses.GET,
                url=re.compile(f".+{tile}.+"),
                body=f.read(),
                status=200,
                headers={"Content-Length": str(os.path.getsize(fp))},
            )


@pytest.fixture
def catalog():
    return srtm.SRTM()


@responses.activate
def test_srtm_token(catalog):
    assert (
        catalog._token
        == "rSmsq2/UJPPDrlzrsodWNBXhH6lkLOC1s7Uzs+a2bfDoMpdzChBqJUnhQuJht6ZOWHvRcDgeO0h3RiXvBueIGQ=="
    )


@responses.activate
def test_srtm_login(catalog):
    catalog.login("user", "pass")


def test_srtm_get_bounding_boxes(catalog):
    assert len(catalog.bounding_boxes == 14295)


def test_srtm_find(catalog, djibouti_geom):
    tiles = catalog.find(djibouti_geom)
    assert len(tiles) == 7
    for tile in tiles:
        assert tile.startswith("https://") and tile.endswith(".hgt.zip")


@responses.activate
def test_srtm_download(catalog):

    url = "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/N11E042.SRTMGL1.hgt.zip"
    with tempfile.TemporaryDirectory(prefix="AccessMod_") as tmp_dir:

        # 1st download
        catalog.download(url, tmp_dir, use_cache=False, overwrite=False)

        # overwrite previous file
        catalog.download(url, tmp_dir, use_cache=False, overwrite=True)

        # cache the downloaded file
        catalog.download(url, tmp_dir, use_cache=True, overwrite=True)

        # use the cache from previous download
        catalog.download(url, tmp_dir, use_cache=True, overwrite=True)


def test_merge_tiles():

    # the 6 zipped tiles used to test mosaicking
    # they have been downsampled to speed up tests
    tiles = [
        os.path.join(os.path.dirname(__file__), "data", "srtm", f"{tile}.SRTMGL1.zip")
        for tile in ["N10E041", "N10E042", "N11E041", "N11E042", "N11E043", "N12E042"]
    ]

    with tempfile.TemporaryDirectory(prefix="AccessMod_") as tmp_dir:

        # unzip .zip tiles from test data directory
        tiles_unzipped = []
        for tile in tiles:
            img = utils.unzip(tile, tmp_dir, remove_archive=False)[0]
            tiles_unzipped.append(img)

        dem = os.path.join(tmp_dir, "dem.tif")
        srtm.merge_tiles(tiles_unzipped, dem, overwrite=False)

        with rasterio.open(dem) as src:
            data = src.read(1, masked=True)
            assert data.any()
            assert data.min() >= -500
            assert data.max() <= 3000


def test_compute_slope():

    dem = os.path.join(os.path.dirname(__file__), "data", "srtm", "dem.tif")
    slope = srtm.compute_slope(dem, "slope.tif")
    with rasterio.open(slope) as src:
        data = src.read(1, masked=True)
        assert data.any()
        assert data.min() >= 0
        assert data.max() <= 100
