#!/usr/bin/env bash

MODALITY=${1:-T1}  # "T1" is the default value
IXI_DIR="data/IXI"
DATASET_NAME="IXI-$MODALITY"
MODALITY_DIR=$IXI_DIR/$DATASET_NAME
RAW_IMAGES_DIR="${MODALITY_DIR}/raw"
mkdir -p $RAW_IMAGES_DIR
cd $RAW_IMAGES_DIR
TAR_NAME="${DATASET_NAME}.tar"
wget "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/${TAR_NAME}"
tar xf $TAR_NAME
rm $TAR_NAME
