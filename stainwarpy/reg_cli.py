import typer
import os
import numpy as np
import json
from tifffile import TiffFile
from skimage.transform import AffineTransform, resize
from stainwarpy import __version__
from .regPipeline import registration_pipeline
from .preprocess import extract_channel, load_image_data, get_pixel_size_ome_tiff, get_image_size_ome_tiff, save_ome_tiff, save_ome_mask
from .reg import transform_seg_mask


app = typer.Typer(help="Register H&E stained images to multiplexed images using a feature based registration pipeline.")


@app.callback(invoke_without_command=True)
def main_callback(
    version: bool = typer.Option(False, "--version", help="Show stainwarpy version and exit.", is_eager=True)
):
    if version:
        typer.echo(f"stainwarpy {__version__}")
        raise typer.Exit()
    

@app.command(name="register")
def register(
    multiplexed_path: str = typer.Argument(..., help="Path to the multiplexed image (.tif/.tiff/.ome.tif/.ome.tiff)"),
    hne_path: str = typer.Argument(..., help="Path to the hne image (.tif/.tiff/.ome.tif/.ome.tiff)"),
    output_folder: str = typer.Argument(..., help="Folder to save the registered images and metrics"),
    fixed_img: str = typer.Argument(..., help="Which image to use as fixed image: ['multiplexed', 'hne']"),
    final_img_sz: str = typer.Argument(..., help="Pixel size for final image: ['multiplexed', 'hne']"),
    multiplexed_px_sz: float = typer.Option(None, help="Pixel size of the multiplexed image (if image is not .ome.tif)"),
    hne_px_sz: float = typer.Option(None, help="Pixel size of the hne image (if image is not .ome.tif)"),
    feature_tform: str = typer.Option('similarity', help="Feature transformation method ['similarity', 'affine', 'projective']. 'similarity' by default and recommended.", show_default=True),
    channel_idx: int = typer.Option(0, help="Channel index (DAPI) to extract if channel extraction not done beforehand for multiplexed image", show_default=True)
):
    """
    Sub-command to register H&E stained images to multiplexed images using feature based registration. Saves registered image, transformation maps and registration metrics to the specified output folder.

    Parameters:
    - multiplexed_path (str): Path to the multiplexed image (.tif/.tiff/.ome.tif/.ome.tiff)
    - hne_path (str): Path to the hne image (.tif/.tiff/.ome.tif/.ome.tiff)
    - output_folder (str): Folder to save the registered images and metrics
    - fixed_img (str): Which image to use as fixed image: ['multiplexed', 'hne']
    - final_img_sz (str): Pixel size for final image: ['multiplexed', 'hne']
    - multiplexed_px_sz (float, optional): Pixel size of the multiplexed image (if image is not .ome.tif)
    - hne_px_sz (float, optional): Pixel size of the hne image (if image is not .ome.tif)
    - feature_tform (str, optional): Feature transformation method ['similarity', 'affine', 'projective']. 'similarity' by default and recommended. 
    - channel_idx (int, optional): Channel index (DAPI) to extract if channel extraction not done beforehand for multiplexed image

    Returns:
    - None
    """

    if fixed_img == 'multiplexed':
        fixed_path = multiplexed_path
        moving_path = hne_path
        fixed_px_sz = multiplexed_px_sz
        moving_px_sz = hne_px_sz
        if final_img_sz not in ['multiplexed', 'hne']:
            raise ValueError("final_img_sz must be either 'multiplexed' or 'hne'.")
        final_img_sz = 'fixed' if final_img_sz == 'multiplexed' else 'moving'

    elif fixed_img == 'hne':
        fixed_path = hne_path
        moving_path = multiplexed_path
        fixed_px_sz = hne_px_sz
        moving_px_sz = multiplexed_px_sz
        if final_img_sz not in ['multiplexed', 'hne']:
            raise ValueError("final_img_sz must be either 'multiplexed' or 'hne'.")
        final_img_sz = 'fixed' if final_img_sz == 'hne' else 'moving'
        
    else:
        raise ValueError("fixed_img must be either 'multiplexed' or 'hne'.")


    # run the pipeline
    transformation_map, final_img, tre, mi = registration_pipeline(
        fixed_path,
        moving_path,
        fixed_px_sz,
        moving_px_sz,
        fixed_img,
        final_img_sz,
        feature_tform=feature_tform,
        chnl_idx=channel_idx
    )

    os.makedirs(output_folder, exist_ok=True)

    # save registration metrics and tfrom map in json
    metrics_output_path = os.path.join(output_folder, "registration_metrics_tform_map.json")

    params_list = transformation_map.params.tolist()
    with open(metrics_output_path, "w") as f:
        json.dump({"TRE": tre, "Mutual Information": mi, "Transformation Map": params_list}, f)
    print(f"Registration metrics saved to {metrics_output_path}")

    # save registered image
    ome_xml = None
    try:
        with TiffFile(moving_path) as ref:
            ome_xml = ref.ome_metadata
    except:
        pass

    final_img_path = os.path.join(output_folder, "0_final_channel_image.ome.tif")

    if final_img_sz == 'fixed':
        save_ome_tiff(final_img, final_img_path, physical_size_x=fixed_px_sz, physical_size_y=fixed_px_sz, source_ome_xml=ome_xml)
    elif final_img_sz == 'moving':
        save_ome_tiff(final_img, final_img_path, physical_size_x=moving_px_sz, physical_size_y=moving_px_sz, source_ome_xml=ome_xml)

    print(f"Registered image saved to {final_img_path}")

    # save transformation map

    np.save(os.path.join(output_folder, f"feature_based_transformation_map.npy"), transformation_map.params)

    print(f"Transformation maps saved to {output_folder}/feature_based_transformation_map.npy")



@app.command(name="extract-channel")
def extract_channel_cmd(
    file_path: str = typer.Argument(..., help="Path to the input image (.tif/.tiff/.ome.tif/.ome.tiff)"),
    output_folder_path: str = typer.Argument(..., help="Folder to save the image with extracted channel"),
    channel_idx: int = typer.Option(0, help="Channel index to extract (Default = 0 for DAPI)", show_default=True),
):
    """
    Sub-command to extract a specific channel from a multi-channel image and save it as a separate image.

    Parameters:
    - file_path (str): Path to the input image (.tif/.tiff/.ome.tif/.ome.tiff)
    - output_folder_path (str): Folder to save the image with extracted channel
    - channel_idx (int, optional): Channel index to extract (Default = 0 for DAPI)

    Returns:
    - None
    """

    img = load_image_data(file_path)
    img_ch = extract_channel(img, channel_idx)

    os.makedirs(output_folder_path, exist_ok=True)

    try:
        px, py = get_pixel_size_ome_tiff(file_path)
    except:
        px, py = None, None

    img_path = os.path.join(output_folder_path, f"multiplexed_single_channel_img.ome.tif")
    save_ome_tiff(img_ch, img_path, physical_size_x=px, physical_size_y=py)
    print(f"Image with extracted channel saved to {img_path}")



@app.command(name="transform-seg-mask")
def transform_seg_mask_cmd(
    mask_path: str = typer.Argument(..., help="Path to the segmentation mask of the moving image (.ome.tif/.ome.tiff/.tif/.tiff/.npy)"),
    multiplexed_path: str = typer.Argument(..., help="Path to the multiplexed image (.tif/.tiff/.ome.tif/.ome.tiff)"),
    hne_path: str = typer.Argument(..., help="Path to the hne image (.tif/.tiff/.ome.tif/.ome.tiff)"),
    output_folder_path: str = typer.Argument(..., help="Folder to save the transformed segmentation mask"),
    tform_map_path: str = typer.Argument(..., help="Path to the transformation map"),
    fixed_img: str = typer.Argument(..., help="Which image used as fixed image: ['multiplexed', 'hne']"),
    final_mask_sz: str = typer.Argument(..., help="Pixel size for final mask: ['multiplexed', 'hne']"),
    multiplexed_px_sz: float = typer.Option(None, help="Pixel size of the multiplexed image (if image is not .ome.tif)", show_default=True),
    hne_px_sz: float = typer.Option(None, help="Pixel size of the hne image (if image is not .ome.tif)", show_default=True)
):
    """
    Sub-command to transform a segmentation mask from the moving image space to the fixed image space using the provided transformation map.
    
    Parameters:
    - mask_path (str): Path to the segmentation mask of the moving image (.ome.tif/.ome.tiff/.tif/.tiff/.npy)
    - multiplexed_path (str): Path to the multiplexed image (.tif/.tiff/.ome.tif/.ome.tiff)
    - hne_path (str): Path to the hne image (.tif/.tiff/.ome.tif/.ome.tiff)
    - output_folder_path (str): Folder to save the transformed segmentation mask
    - tform_map_path (str): Path to the transformation map
    - fixed_img (str): Which image used as fixed image: ['multiplexed', 'hne']
    - final_mask_sz (str): Pixel size for final mask: ['multiplexed', 'hne']
    - multiplexed_px_sz (float, optional): Pixel size of the multiplexed image (if image is not .ome.tif)
    - hne_px_sz (float, optional): Pixel size of the hne image (if image is not .ome.tif)

    Returns:
    - None
    """

    if fixed_img == 'multiplexed':
        fixed_path = multiplexed_path
        moving_path = hne_path
        fixed_px_sz = multiplexed_px_sz
        moving_px_sz = hne_px_sz
        if final_mask_sz not in ['multiplexed', 'hne']:
            raise ValueError("final_mask_sz must be either 'multiplexed' or 'hne'.")
        final_mask_sz = 'fixed' if final_mask_sz == 'multiplexed' else 'moving'

    elif fixed_img == 'hne':
        fixed_path = hne_path
        moving_path = multiplexed_path
        fixed_px_sz = hne_px_sz
        moving_px_sz = multiplexed_px_sz
        if final_mask_sz not in ['multiplexed', 'hne']:
            raise ValueError("final_mask_sz must be either 'multiplexed' or 'hne'.")
        final_mask_sz = 'fixed' if final_mask_sz == 'hne' else 'moving'

    else:
        raise ValueError("fixed_img must be either 'multiplexed' or 'hne'.")
    
    
    # load mask
    if mask_path.endswith('.tif') or mask_path.endswith('.tiff'):
        mask = load_image_data(mask_path)
    elif mask_path.endswith('.npy'):
        mask = np.load(mask_path) 
    else:
        raise ValueError("Segmentation mask must be a .npy or .tif/.tiff/.ome.tif/.ome.tiff file.")
    print(f"Loaded segmentation mask.")

    # load and create transformation parameter objects
    transformation_maps= AffineTransform(matrix=np.load(tform_map_path))

    print("Loaded transformation map.")

    if fixed_px_sz is None:
        try:
            fixed_px_sz, _ = get_pixel_size_ome_tiff(fixed_path)
        except Exception:
            fixed_px_sz = None
        
        if fixed_px_sz is None:
            raise ValueError("Pixel size information not found in metadata for fixed image. Please provide fixed_px_sz.")
    
    if moving_px_sz is None:
        try:
            moving_px_sz, _ = get_pixel_size_ome_tiff(moving_path)
        except:
            moving_px_sz = None

        if moving_px_sz is None:
            raise ValueError("Pixel size information not found in metadata for moving image. Please provide moving_px_sz.")
    
    try:
        scale = moving_px_sz / fixed_px_sz
    except:
        raise ValueError("Could not determine scaling factor. Please check the provided pixel sizes or image paths (ome.tiff).")
    

    # get fixed image size for output shape
    fixed_img_shape_bfr = get_image_size_ome_tiff(fixed_path)
    fixed_img_shape = (int(fixed_img_shape_bfr[0]/scale), int(fixed_img_shape_bfr[1]/scale))

    # warp moving image
    moved_mask = transform_seg_mask(mask, transformation_maps, output_shape=fixed_img_shape)

    # resize the mask to match provided pixel size
    target_shape = fixed_img_shape_bfr

    if final_mask_sz == 'fixed':
        moved_mask = resize(moved_mask, target_shape, order=0, preserve_range=True, anti_aliasing=False).astype(moved_mask.dtype)
    elif final_mask_sz == 'moving':
        moved_mask = moved_mask


    # save transformed mask
    ome_xml = None
    try:
        with TiffFile(mask_path) as ref:
            ome_xml = ref.ome_metadata
    except:
        pass

    os.makedirs(output_folder_path, exist_ok=True)
    
    save_ome_mask(moved_mask, 
                  os.path.join(output_folder_path, "transformed_segmentation_mask.ome.tif"),
                  physical_size_x=fixed_px_sz if final_mask_sz == 'fixed' else moving_px_sz,
                  physical_size_y=fixed_px_sz if final_mask_sz == 'fixed' else moving_px_sz,
                  source_ome_xml=ome_xml)
    print(f"Transformed segmentation mask saved to {output_folder_path}/transformed_segmentation_mask.ome.tif")


def main():
    app()


if __name__ == "__main__":
    main()