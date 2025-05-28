#!/bin/sh
export PYTHONPATH=.:$PYTHONPATH 
pytest ./tests/ $*
