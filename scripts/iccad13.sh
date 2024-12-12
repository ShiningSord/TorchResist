#!/bin/bash

python3 -m examples.iccad13 --mask ./data/MetalSet/1nm/images --outpath ./data/MetalSet/iccad13/1nm/litho --config ./simulator/lithobench/config/lithosimple

# python3 -m examples.iccad13 --mask /research/d5/gds/zxwang22/storage/resist/cells/png/1nm --outpath /research/d5/gds/zxwang22/storage/resist/iccad13/1nm/litho --config ./simulator/lithobench/config/lithosimple