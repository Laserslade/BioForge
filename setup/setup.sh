#!/bin/bash
# setup.sh - Installs dependencies and downloads the artifacts the pipeline needs.
set -e

pip install biopython networkx requests torch cobra pymoo matplotlib pandas

mkdir -p ./bio_pipeline
cd ./bio_pipeline

if [ ! -d "./ProteinMPNN" ]; then
    git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git ./ProteinMPNN
else
    echo "ProteinMPNN already present."
fi

mkdir -p ./models
if [ ! -f "./models/iML1515.xml" ]; then
    wget -q -O ./models/iML1515.xml http://bigg.ucsd.edu/static/models/iML1515.xml
    echo "Downloaded iML1515.xml"
else
    echo "iML1515.xml already present."
fi
