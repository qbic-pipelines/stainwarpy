# stainwarpy

<p align="center">
  <img src="https://raw.githubusercontent.com/tckumarasekara/stainwarpy/main/stainwarpy_logo.png" width="600" alt="Stainwarpy Logo">
</p>

[![Docker](https://img.shields.io/badge/Docker-stainwarpy--cli-blue)](https://hub.docker.com/repository/docker/tckum/stainwarpy-cli)
[![PyPI version](https://img.shields.io/pypi/v/stainwarpy)](https://pypi.org/project/stainwarpy/)
[![Bioconda version](https://img.shields.io/conda/vn/bioconda/stainwarpy.svg)](https://anaconda.org/bioconda/stainwarpy)

**stainwarpy** is a command-line tool and a Python package for registering H&E stained and multiplexed tissue images. It provides a feature based registration pipeline, saving registered images, transformation maps and evaluation metrics.


## Features

- Register H&E images and multiplexed images (after extracting DAPI channel) using transformations.
- Supports feature-based registration.
- Outputs registered images, transformation maps and evaluation metrics (TRE and Mutual Information).
- Transforms segmentation masks based on the computed transformations


## Recommendations

- For most cases, it is recommended to register **H&E images onto multiplexed images** (H&E as moving image).  
- The **default similarity transformation** usually works well and stable, therefore recommended.
- ⚠️Note: This package is under active development. Frequent updates may introduce breaking changes. Use new versions with caution.

## Installation

You can install **stainwarpy** using pip:

```bash
pip install stainwarpy
```

or

Bioconda (for Linux/macOS):

```bash
conda install -c bioconda stainwarpy
```

---

## Usage as a command-line tool

### Register Images

```bash
stainwarpy register <fixed_path> <moving_path> <output_folder> <final_img_sz> [options]
```

#### Examples:

```bash
stainwarpy register data/fixed_img.ome.tiff data/moving_img.ome.tiff ../output multiplexed multiplexed
```
```bash
stainwarpy register data/fixed_img.tif data/moving_img.tif ../output multiplexed multiplexed --multiplexed-px-sz 0.21 --hne-px-sz 0.52
```

#### Arguments:

- **multiplexed_path**: Path to the multiplexed image (.tif/.tiff./.ome.tif/.ome.tiff)
- **hne_path**: Path to the H&E image (.tif/.tiff./.ome.tif/.ome.tiff)
- **output_folder**: Folder to save the registered images and metrics  
- **fixed_img**: Which one to be taken as fixed image `multiplexed` or `hne`
- **final_img_sz**: Final moving image pixel size to be kept in the size of `multiplexeed` or `hne` image pixel size  

#### Options:

- `--multiplexed-px-sz` : Pixel size of the multiplexed image (no need to provide for ome.tiff, so default: None)
- `--hne-px-sz` : Pixel size of the H&E image (no need to provide for ome.tiff, so default: None)
- `--feature-tform` : Feature transformation method: `similarity` or`affine` or `projective` (default: `similarity`)
- `--channel-idx` : Channel index (DAPI) to extract if channel extraction not done beforehand for multiplexed image (default: `0`), not used if already extracted.

#### Output

After running registration, the following files/folders will be generated and saved in the specified output folder:

- **registration_metrics_tform_map.json** — TRE and Mutual Information and transformation map in an user friendly file format  
- **0_final_channel_image.ome.tif** — Registered image (in the pixel size of moving image)
- **feature_based_transformation_map.npy** — Transformation map 


### Extract a Channel (DAPI can be extracted for registration)

```bash
stainwarpy extract-channel <file_path> <output_folder_path> [--channel-idx N]
```

#### Arguments
- **file-path** : Path to multichannel image (.tif/.tiff/.ome.tif/.ome.tiff)
- **output-folder-path** : Folder to save the extracted channel image

#### Options
- `--channel-idx`: Channel index to extract (default: 0 for DAPI)
 
#### Output

- **multiplexed_single_channel_img.ome.tif** - Image with the extracted channel saved in the specified output folder


### Transform segmentation Masks

Transform segmentation masks based on the transformation maps produced with the command `register`. 

```bash
stainwarpy transform-seg-mask <mask_path> <fixed_path> <moving_path> <output_folder_path> <tform_map_path> <multiplexed/hne> <multiplexed/hne> [options]
```

#### Arguments

- **mask_path** : Path to the segmentation mask of the moving image (.ome.tif/.ome.tiff/.tif/.tiff/.npy)
- **multiplexed_path** : Path to the multiplexed image (.tif/.tiff/.ome.tif/.ome.tiff)
- **hne_path** : Path to the H&E image (.tif/.tiff/.ome.tif/.ome.tiff)
- **output_folder_path** : Folder to save the transformed segmentation mask
- **tform_map_path** : Path to the transformation map
- **fixed_img** : Which image taken as fixed image `multiplexed` or `hne`
- **final_mask_sz** : Pixel size for final mask: `multiplexed` or `hne`

#### Options

- `--multiplexed-px-sz` : Pixel size of the multiplexed image (no need to provide for ome.tiff, so default: None)
- `--hne-px-sz` : Pixel size of the H&E image (no need to provide for ome.tiff, so default: None)

#### Output

- **transformed_segmentation_mask.ome.tif** : The segmentation mask transformed to the fixed image coordinate space saved in the specified output folder


---

## Usage as a Python Library

Although **stainwarpy** is mainly a command-line tool, its functions can also be used directly in Python for scripting.

### Example: Running the Registration Pipeline

```python
from stainwarpy.regPipeline import registration_pipeline

# run registration pipeline
tform_map, final_img, tre, mi = registration_pipeline(
    fixed_path="fixed_image.tif",
    moving_path="moving_image.tif",
    fixed_px_sz=0.5,
    moving_px_sz=0.5,
    fixed_img="multiplexed",
    final_img_sz="fixed",
    feature_tform="affine"        # to use a transformation other than default "similarity"
)

print("TRE:", tre)
print("Mutual Information:", mi)
```
---

## License

This project is licensed under the **MIT License**. 

This project includes portions of code in stainwarpy/preprocess.py adapted from **HistomicsTK**
(https://github.com/DigitalSlideArchive/HistomicsTK/), which is licensed under **Apache License 2.0**.
See [LICENSE_HISTOMICSTK.txt](LICENSE_HISTOMICSTK.txt) for the full license text.





