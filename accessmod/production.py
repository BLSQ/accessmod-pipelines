# Custom setup to inject sentry & credentials info if executed in an openhexa environnement

import logging
import logging.config
import os

import requests

logger = logging.getLogger(__name__)
logger.info("Execute common")

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "standard": {"format": "%(asctime)s %(levelname)s: %(message)s"},
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
        },
        "loggers": {
            "": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }
)

if "SENTRY_DSN" in os.environ:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration

    # inject sentry into logging config. set level to ERROR, we don't really want the rest?
    sentry_logging = LoggingIntegration(level=logging.ERROR, event_level=logging.ERROR)
    sentry_sdk.init(
        dsn=os.environ["SENTRY_DSN"],
        integrations=[sentry_logging],
        traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "1")),
        send_default_pii=True,
        environment=os.environ.get("SENTRY_ENVIRONMENT"),
    )

if "HEXA_PIPELINE_TOKEN" in os.environ:
    token = os.environ["HEXA_PIPELINE_TOKEN"]
    r = requests.post(
        os.environ["HEXA_CREDENTIALS_URL"],
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()

    os.environ.update(data["env"])

    for name, encoded_content in data["files"].items():
        content = encoded_content.encode().decode("base64")
        with (open(name, "w")) as f:
            f.write(content)

    # to help debug...
    print("Hexa env update, variables:")
    for var in sorted(os.environ):
        new_var = "(from hexa)" if var in data["env"] else ""
        print(f"Var {var} {new_var}")

    if data["files"]:
        print("Hexa files injection:")
        for path in sorted(data["files"]):
            print(f"File {path} added")
