#!/bin/bash
set -e


command=$1
arguments=${*:2}

show_help() {
  echo """
  Available commands:

  srtm             : start srtm subsystem
  worldpop         : start worldpop subsystem
  accessibility    : start accessibility analysis
  python           : run arbitrary python code
  bash             : launch bash session
  test             : launch tests using Pytest

  Any arguments passed will be forwarded to the executed command
  """
}

case "$command" in
"srtm")
  python3 -m srtm $arguments
  ;;
"worldpop")
  python3 -m worldpop $arguments
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
