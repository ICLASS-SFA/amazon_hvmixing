#!/bin/bash

source activate /global/common/software/m1867/python/py310

python /global/homes/f/feng045/program/iclass/re2/amazon_hvmixing/regrid_wrf/regrid_W.py $1 $2
