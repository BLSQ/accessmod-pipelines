import logging
import os

import click
import requests
import utils
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


BASE_URL = "https://data.worldpop.org/GIS/Population/Global_2000_2020"
PERIOD_CONSTRAINED = (2020, 2020)
PERIOD_UNCONSTRAINED = (2000, 2020)


def build_url(
    country: str, year: int = 2020, un_adj: bool = True, constrained: bool = True
) -> str:
    """Build download URL.

    Parameters
    ----------
    country : str
        ISO A3 country code.
    year : int
        Year of interest.
    un_adj : bool, optional
        Use UN adjusted population counts (default=True).
    constrained : bool, optional
        Constrained vs. unconstrained dataset (default=True).

    Return
    ------
    str
        Public download URL.
    """
    if constrained:
        return (
            f"{BASE_URL}_Constrained/{year}/maxar_v1/{country.upper()}/"
            f"{country.lower()}_ppp_{year}{'_UNadj' if un_adj else ''}_constrained.tif"
        )
    else:
        return (
            f"{BASE_URL}/{year}/{country.upper()}/"
            f"{country.lower()}_ppp_{year}{'_UNadj' if un_adj else ''}.tif"
        )


def download_raster(
    country: str,
    output_dir: str,
    year: int = 2020,
    un_adj: bool = True,
    constrained: bool = True,
    resolution: int = 100,
    timeout: int = 30,
    overwrite: bool = False,
) -> str:
    """Download a WorldPop population dataset.

    Four types of datasets are supported:
      - Unconstrained (100 m)
      - Unconstrained and UN adjusted (100 m)
      - Constrained (100 m)
      - Constrained and UN adjusted (100 m)

    See Worldpop website for more details:
    <https://www.worldpop.org/project/categories?id=3>

    Parameters
    ----------
    country : str
        ISO A3 country code.
    output_dir : str
        Path to output directory.
    year : int, optional
        Year of interest (default=2020).
    un_adj : bool, optional
        Use UN adjusted population counts (default=True)
    constrained : bool, optional
        Constrained or unconstrained dataset (default=True).
    resolution : int, optional
        Spatial resolution in meters (default=100).
        Either 100 or 1,000 m.
    timeout : int, optional
        Request timeout in seconds (default=30).
    overwrite : bool, optional
        Overwrite existing files (default=False).

    Return
    ------
    str
        Path to output GeoTIFF file.
    """
    retry_adapter = HTTPAdapter(
        max_retries=Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET"],
            # method_whitelist=["HEAD", "GET"],
        )
    )

    s = requests.Session()
    s.mount("https://", retry_adapter)
    s.mount("http://", retry_adapter)

    url = build_url(country=country, year=year, un_adj=un_adj, constrained=constrained)
    logger.info(f"WorldPop URL: {url}.")

    os.makedirs(output_dir, exist_ok=True)
    fp = os.path.join(output_dir, url.split("/")[-1])
    if os.path.isfile(fp) and not overwrite:
        raise FileExistsError(f"File {fp} already exists.")

    with s.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        size_http = int(r.headers.get("content-length"))
        with open(fp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    logger.info(f"Downloaded WorldPop data into {fp}.")

    size_local = os.path.getsize(fp)
    if size_local != size_http:
        raise IOError(
            f"Remote ({utils._human_readable_size(size_http)}) and "
            f"local ({utils._human_readable_size(size_local)}) sizes "
            f"of file {fp} differ."
        )

    return fp


@click.group()
def cli():
    pass


@cli.command()
@click.option("--country", type=str, required=True, help="country code")
@click.option("--year", type=int, required=True, default=2020, help="year of interest")
@click.option(
    "--constrained/--unconstrained",
    is_flag=True,
    default=True,
    help="constrained vs. unconstrained",
)
@click.option(
    "--un-adj/--no-adj",
    is_flag=True,
    default=True,
    help="UN-adjusted population counts",
)
@click.option("--resolution", type=int, default=100, help="spatial resolution (m)")
@click.option("--output-dir", type=str, required=True, help="output data directory")
@click.option(
    "--overwrite", is_flag=True, default=False, help="overwrite existing files"
)
def download(
    country: str,
    year: int,
    constrained: bool,
    un_adj: bool,
    resolution: int,
    output_dir: str,
    overwrite: bool,
):
    """Download WorldPop population dataset."""
    if not utils.country_is_valid(country):
        raise ValueError(f"{country} is not a valid country code.")

    # only 2020 data available for constrained datasets
    if constrained:
        ymin, ymax = PERIOD_CONSTRAINED
        if year < ymin or year > ymax:
            raise ValueError(
                f"Year {year} not supported for Worldpop constrained datasets."
            )
    else:
        ymin, ymax = PERIOD_UNCONSTRAINED
        if year < ymin or year > ymax:
            raise ValueError(
                f"Year {year} not supported for Worldpop unconstrained datasets."
            )

    # only 100 m resolution available for constrained datasets
    if (resolution not in (100, 1000)) or (resolution == 1000 and constrained):
        raise ValueError(f"Spatial resolution of {resolution} m is not available.")

    download_raster(
        country=country,
        output_dir=output_dir,
        year=year,
        un_adj=un_adj,
        constrained=constrained,
        resolution=resolution,
        timeout=30,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    cli()
