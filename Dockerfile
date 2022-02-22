FROM python:3.9.7

LABEL maintainer="yforget@bluesquarehub.com"

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN apt-get update && apt-get install -y \
    gdal-bin \
    proj-bin \
    grass-core \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY docker-entrypoint.sh \
  processing.py \
  utils.py \
  srtm.py \
  srtm30m_bounding_boxes.json \
  worldpop.py \
  countries.csv \
  /app/

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["srtm"]
