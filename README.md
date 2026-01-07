# stainwarpy

<p align="center">
  <img src="https://raw.githubusercontent.com/tckumarasekara/stainwarpy/main/stainwarpy_logo.png" width="600" alt="Stainwarpy Logo">
</p>

[![Docker](https://img.shields.io/badge/Docker-stainwarpy--cli-blue)](https://hub.docker.com/repository/docker/tckum/stainwarpy-cli)
[![PyPI version](https://img.shields.io/pypi/v/stainwarpy)](https://pypi.org/project/stainwarpy/)

**stainwarpy** is a command-line tool and a Python package for registering H&E stained and multiplexed tissue images. It provides a feature based registration pipeline, saving registered images, transformation maps and evaluation metrics.


## Features

- Register H&E images and multiplexed images (after extracting DAPI channel) using transformations.
- Supports feature-based registration.
- Outputs registered images (in the pixel size of moving image), transformation maps and evaluation metrics (TRE and Mutual Information).
- Transforms segmentation masks based on the computed transformations


## Recommendations

- For most cases, it is recommended to register **H&E images onto multiplexed images** (H&E as moving image).  
- The **default similarity transformation** usually works well and stable, therefore recommended.


## Installation

You can install **stainwarpy** using pip:

```bash
pip install stainwarpy
```

---

## Usage as a command-line tool

### Register Images

```bash
stainwarpy register <fixed_path> <moving_path> <output_folder> <fixed_img> [options]
```

#### Examples:

```bash
stainwarpy register data/fixed_img.ome.tiff data/moving_img.ome.tiff ../output multiplexed
```
```bash
stainwarpy register data/fixed_img.tif data/moving_img.tif ../output multiplexed --fixed-px-sz 0.21 --moving-px-sz 0.52
```

#### Arguments:

- **fixed_path**: Path to the fixed image (H&E or DAPI or Multiplexed image path (.tif/.tiff./.ome.tif/.ome.tiff))
- **moving_path**: Path to the moving image (H&E or DAPI or Multiplexed image path (.tif/.tiff./.ome.tif/.ome.tiff))
- **output_folder**: Folder to save the registered images and metrics  
- **fixed_img**: Type of fixed image: `multiplexed` or `hne`  

#### Options:

- `--fixed-px-sz` : Pixel size of the fixed image (no need to provide for ome.tiff, so default: None)
- `--moving-px-sz` : Pixel size of the moving image (no need to provide for ome.tiff, so default: None)
- `--feature-tform` : Feature transformation method: `similarity` or`affine` or `projective` (default: `similarity`)

#### Output

After running registration, the following files/folders will be generated and saved in the specified output folder:

- **registration_metrics.json** — TRE and Mutual Information  
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

- **multiplexed_channel_{channel_idx}.tif** - Image with the extracted channel saved in the specified output folder


### Transform segmentation Masks

Transform segmentation masks based on the transformation maps produced with the command `register`. 

```bash
stainwarpy transform-seg-mask <mask_path> <fixed_path> <output_folder_path> <tform_map_path> <moving_px_sz> [--fixed-px-sz]
```

#### Arguments

- **mask_path** : Path to the segmentation mask of the moving image (.npy)
- **fixed_path** : Path to the fixed image (.tif/.tiff/.ome.tif/.ome.tiff)
- **output_folder_path** : Folder to save the transformed segmentation mask
- **tform_map_path** : Path to the transformation map
- **moving_px_sz** : Path to moving image if .ome.tiff or Pixel size of the moving image

#### Options

- `--fixed-px-sz` : Pixel size of the fixed image (no need to provide for ome.tiff, so default: None)

#### Output

- **transformed_segmentation_mask.npy** : The segmentation mask transformed to the fixed image coordinate space saved in the specified output folder


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





