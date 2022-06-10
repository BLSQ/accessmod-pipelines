#!/bin/bash
set -e


command=$1
arguments=${*:2}

show_help() {
  echo """
  Available commands:

  srtm             : start SRTM subsystem (DEM)
  healthsites      : start healthsites subsystem (health facilities)
  osm              : start OSM subsystem (water/transport)
  copernicus_glc   : start Coppernicus GLC subsystem (land cover)
  worldpop         : start worldpop subsystem (population)
  accessibility    : start accessibility analysis
  python           : run arbitrary python code
  bash             : launch bash session
  test             : launch tests using Pytest

  Any arguments passed will be forwarded to the executed command
  """
}

case "$command" in
"srtm")
  python3 -m srtm2 $arguments
  ;;
"worldpop")
  python3 -m worldpop $arguments
  ;;
"healthsites")
  python3 -m healthsites $arguments
  ;;
"osm")
  python3 -m osm $arguments
  ;;
"copernicus_glc")
  python3 -m copernicus_glc $arguments
  ;;
"accessibility")
  python3 -m accessibility $arguments
  ;;
"boundaries")
  python3 -m boundaries $arguments
  ;;
"zonalstats")
  python3 -m zonalstats $arguments
  ;;
"test")
  pytest -s $arguments
  ;;
"python")
  python3 $arguments
  ;;
"bash")
  bash $arguments
  ;;
*)
  show_help
  ;;
esac
