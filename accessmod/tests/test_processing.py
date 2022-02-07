import json
import os
import tempfile

import processing
import pytest
import rasterio
from rasterio.crs import CRS
from shapely.geometry import shape


@pytest.fixture
def djibouti_geom():
    with open(
        os.path.join(
            os.path.dirname(__file__), "data", "DJ-region-10m-4326-2020.geojson"
        )
    ) as f:
        return shape(json.load(f)["features"][0]["geometry"])


def test_create_grid(djibouti_geom):

    # create a raster grid in UTM projection with a pixel size of 100 m
    # across Djibouti
    transform, shape_, bounds = processing.create_grid(
        djibouti_geom, dst_crs=CRS.from_epsg(32637), dst_res=100
    )

    assert transform.a == 100
    assert shape_ == (2002, 1808)
    assert bounds == pytest.approx((801536, 1209021, 982226, 1409134), abs=1)


def test_reproject():

    with tempfile.TemporaryDirectory(prefix="AccessMod_") as tmp_dir:

        src_fp = os.path.join(os.path.dirname(__file__), "data", "srtm", "N11E042.hgt")
        dst_fp = os.path.join(tmp_dir, "N11E042.hgt")

        xres, yres = 100, 100
        bounds = (
            801536.1040236845,
            1209021.3748652723,
            982226.4919753706,
            1409134.388683451,
        )

        processing.reproject(
            src_raster=src_fp,
            dst_raster=dst_fp,
            dst_crs=CRS.from_epsg(32637),
            dtype="int16",
            xres=xres,
            yres=yres,
            bounds=bounds,
            resampling_alg="near",  # fastest algorithm
        )

        with rasterio.open(dst_fp) as src:

            data = src.read(1, masked=True)
            assert data.any()
            assert data.min() > -250  # assal lake
            assert data.max() < 2000
            assert src.transform.a == xres
            assert src.transform.e == -yres


def test_mask(djibouti_geom):

    with tempfile.TemporaryDirectory(prefix="AccessMod_") as tmp_dir:

        src_fp = os.path.join(os.path.dirname(__file__), "data", "srtm", "N11E042.hgt")
        dst_fp = os.path.join(tmp_dir, "N11E042.hgt")

        processing.mask(
            src_raster=src_fp, dst_raster=dst_fp, geom=djibouti_geom, src_crs=None
        )

        with rasterio.open(dst_fp) as src:

            data = src.read(1, masked=True)
            assert data.any()
            assert data.mask.any()
            assert data.min() > -250
            assert data.max() < 2000
