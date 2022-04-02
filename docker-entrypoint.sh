#!/bin/bash
set -e


command=$1
arguments=${*:2}

show_help() {
  echo """
  Available commands:

  srtm             : start SRTM subsystem (DEM/Slope)
  ghf              : start GHF subsystem (health facilities)
  osm              : start OSM subsystem (water/transport)
  coppernicus_glc  : start Coppernicus GLC subsystem (land cover)
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
"ghf")
  python3 -m ghf $arguments
  ;;
"osm")
  python3 -m osm $arguments
  ;;
"coppernicus_glc")
  python3 -m coppernicus_glc $arguments
  ;;
"accessibility")
  python3 -m accessibility $arguments
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
