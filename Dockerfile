FROM ubuntu:focal-20220113

LABEL maintainer="yforget@bluesquarehub.com"

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN apt-get update && apt-get install -y \
  python3-pip \
  gdal-bin \
  proj-bin \
  grass-core \
  grass-dev \
  libgdal-dev \
  locales \
  python3-six \
  osmium-tool \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY docker-entrypoint.sh \
  accessmod/*.py \
  accessmod/srtm30m_bounding_boxes.json \
  accessmod/osmconf.ini \
  accessmod/geofabrik.json \
  accessmod/countries.csv \
  accessmod/cgls_bounding_boxes.json \
  accessmod/countries_pbf.json \
  accessmod/tests/data/Zonal_Stats_Sample_Data/boundaries.gpkg \
  /app/

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["srtm"]
