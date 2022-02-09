import tempfile

import responses
import worldpop


def test_build_url():

    constrained = "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/maxar_v1/DJI/dji_ppp_2020_UNadj_constrained.tif"
    unconstrained = "https://data.worldpop.org/GIS/Population/Global_2000_2020/2015/DJI/dji_ppp_2015_UNadj.tif"
    no_adj = "https://data.worldpop.org/GIS/Population/Global_2000_2020/2015/DJI/dji_ppp_2015.tif"

    assert worldpop.build_url("dji", 2020, un_adj=True, constrained=True) == constrained
    assert (
        worldpop.build_url("dji", 2015, un_adj=True, constrained=False) == unconstrained
    )
    assert worldpop.build_url("dji", 2015, un_adj=False, constrained=False) == no_adj


@responses.activate
def test_download_raster():

    responses.add(
        responses.GET,
        url="https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/maxar_v1/DJI/dji_ppp_2020_UNadj_constrained.tif",
        body=b"",
        status=200,
        headers={"Content-Length": "0"},
    )

    with tempfile.TemporaryDirectory(prefix="AccessMod_") as tmp_dir:
        worldpop.download_raster(
            country="dji",
            output_dir=tmp_dir,
            year=2020,
            un_adj=True,
            constrained=True,
            resolution=100,
        )
