import logging
import os

import click
import production  # noqa
from fsspec import AbstractFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from s3fs import S3FileSystem

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def filesystem(target_path: str) -> AbstractFileSystem:
    """Guess filesystem based on path"""
    client_kwargs = {}
    if "://" in target_path:
        target_protocol = target_path.split("://")[0]
        if target_protocol == "s3":
            fs_class = S3FileSystem
            client_kwargs = {"endpoint_url": os.environ.get("AWS_S3_ENDPOINT")}
        elif target_protocol == "gcs":
            fs_class = GCSFileSystem
        elif target_protocol == "http" or target_protocol == "https":
            fs_class = HTTPFileSystem
        else:
            raise ValueError(f"Protocol {target_protocol} not supported.")
    else:
        fs_class = LocalFileSystem

    return fs_class(client_kwargs=client_kwargs)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--extent", type=str, required=True, help="boundaries of acquisition")
@click.option("--output-dir", type=str, required=True, help="output data directory")
@click.option(
    "--overwrite", is_flag=True, default=False, help="overwrite existing files"
)
def download(
    extent: str,
    year: int,
    output_dir: str,
    overwrite: bool,
):
    print("DOWNLOAD OK")


@cli.command()
@click.option("--extent", type=str, required=True, help="boundaries of acquisition")
@click.option("--output_dir", type=str, help="output data directory")
@click.option("--input-dir", type=str, required=True, help="input data directory")
@click.option(
    "--overwrite", is_flag=True, default=False, help="overwrite existing files"
)
def extract_water_layer(
    extent: str,
    output_dir: str,
    input_dir: str,
    overwrite: bool,
):
    pass


@cli.command()
@click.option("--extent", type=str, required=True, help="boundaries of acquisition")
@click.option("--output_dir", type=str, help="output data directory")
@click.option("--input-dir", type=str, required=True, help="input data directory")
@click.option(
    "--overwrite", is_flag=True, default=False, help="overwrite existing files"
)
def extract_transport_layer(
    extent: str,
    output_dir: str,
    input_dir: str,
    overwrite: bool,
):
    pass


if __name__ == "__main__":
    cli()
