"""
This script is for performing skull stripping on the affine-aligned datasets.

The data are stored in multiple folders:
- matrices, storing the affine matrices for the affine registration
- mri, storing the MR images
- gif_parcellation, storing the parcellation of the MR images
- reference, storing one image parcellation pair

The preprocessed files will be saved under
- preprocessed/images
- preprocessed/labels
- preprocessed/reference
"""
import glob
import os

import torchio as tio

SMALLEST_BRAIN_LABEL = 24  # from colour table
data_folder_path = "/raid/candi/Yunguan/DeepReg/neuroimaging"
output_folder_path = f"{data_folder_path}/preprocessed"

# get file paths
image_file_paths = glob.glob(f"{data_folder_path}/mri/*.nii.gz")
label_file_paths = glob.glob(f"{data_folder_path}/gif_parcellation/*.nii.gz")
matrix_file_paths = glob.glob(f"{data_folder_path}/matrices/*.txt")

assert len(image_file_paths) == len(label_file_paths) == len(matrix_file_paths)
num_images = len(image_file_paths)

image_file_paths = sorted(image_file_paths)
label_file_paths = sorted(label_file_paths)
matrix_file_paths = sorted(matrix_file_paths)

# get unique IDs
image_file_names = [
    os.path.split(x)[1].replace(".nii.gz", "") for x in image_file_paths
]
label_file_names = [
    os.path.split(x)[1].replace(".nii.gz", "") for x in label_file_paths
]
matrix_file_names = [os.path.split(x)[1].replace(".txt", "") for x in matrix_file_paths]

# images have suffix "_t1_pre_on_mni"
# labels have suffix "_t1_pre_NeuroMorph_Parcellation" or "-T1_NeuroMorph_Parcellation"
# matrices have suffix "_t1_pre_to_mni"
# verify sorted filenames are matching
for i in range(num_images):
    image_fname = image_file_names[i]
    label_fname = label_file_names[i]
    label_fname = label_fname.replace(
        "_t1_pre_NeuroMorph_Parcellation", "_t1_pre_on_mni"
    )
    label_fname = label_fname.replace("-T1_NeuroMorph_Parcellation", "_t1_pre_on_mni")
    matrix_fname = matrix_file_names[i]
    matrix_fname = matrix_fname.replace("_t1_pre_to_mni", "_t1_pre_on_mni")
    assert image_fname == label_fname == matrix_fname


def preprocess(image_path: str, label_path: str, matrix_path: str):
    """
    Preprocess one data sample.

    Args:
        image_path: file path for image
        label_path: file path for parcellation
        matrix_path: file path for affine matrix
    """
    name = os.path.split(image_path)[1].replace("_pre_on_mni.nii.gz", "")
    out_image_path = f"{output_folder_path}/images/{name}.nii.gz"
    out_label_path = f"{output_folder_path}/labels/{name}.nii.gz"

    # resample parcellation to MNI
    matrix = tio.io.read_matrix(matrix_path)
    parcellation = tio.LabelMap(label_path, to_mni=matrix)
    resample = tio.Resample(image_path, pre_affine_name="to_mni")
    parcellation_mni = resample(parcellation)
    parcellation_mni.save(out_label_path)

    # get brain mask
    extract_brain = tio.Lambda(lambda x: (x >= SMALLEST_BRAIN_LABEL))
    brain_mask = extract_brain(parcellation_mni)

    # skull-stripping
    mri = tio.ScalarImage(image_path)
    mri.data[~brain_mask.data.bool()] = 0
    mri.save(out_image_path)


for image_path, label_path, matrix_path in zip(
    image_file_paths, label_file_paths, matrix_file_paths
):
    preprocess(image_path=image_path, label_path=label_path, matrix_path=matrix_path)
