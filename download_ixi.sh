#!/usr/bin/env bash
# Usage:
# $ modality="T2"
# $ ./download_ixi.sh $modality
#
# Possible modalities are T1, T2, PD, MRA and DTI.
# If nothing is passed, the T1 images will be downloaded:
# $ ./download_ixi.sh

MODALITY=${1:-T1}
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
