FROM ubuntu:jammy-20220130

LABEL maintainer="yforget@bluesquarehub.com"

RUN apt-get update && apt-get install -y \
    gdal-bin \
    proj-bin \
    grass-core \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY processing.py utils.py srtm.py srtm30m_bounding_boxes.json /app/

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["srtm"]
