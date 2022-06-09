import logging
import os
import click
import production  # noqa
import requests
import utils

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


BASE_FILE = "/app/boundaries.gpkg"

@click.group()
def cli():
    pass


@cli.command()
@click.option("--config", type=str, required=True, help="pipeline configuration")
@click.option(
    "--webhook-url",
    type=str,
    help="URL to push a POST request with the acquisition's results",
)
@click.option("--webhook-token", type=str, help="Token to use in the webhook POST")

def download(config: str, webhook_url: str, webhook_token: str):

    config = utils.parse_config(config)
    output_path = config["boundaries"]["path"]

    fs = utils.filesystem(output_path)
    fs.put(BASE_FILE, output_path)

if __name__ == "__main__":
    cli()
