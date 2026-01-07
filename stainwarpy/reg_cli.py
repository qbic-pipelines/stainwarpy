import typer
import os
import numpy as np
import json
from tifffile import imwrite, TiffFile
from skimage.transform import AffineTransform, resize
from .regPipeline import registration_pipeline
from .preprocess import extract_channel, load_image_data, get_pixel_size_ome_tiff, save_ome_tiff
from .reg import transform_seg_mask


app = typer.Typer(help="Register H&E stained images to multiplexed images using a feature based registration pipeline.")


@app.command(name="register")
def register(
    fixed_path: str = typer.Argument(..., help="Path to the fixed image (.tif/.tiff/.ome.tif/.ome.tiff)"),
    moving_path: str = typer.Argument(..., help="Path to the moving image (.tif/.tiff/.ome.tif/.ome.tiff)"),
    output_folder: str = typer.Argument(..., help="Folder to save the registered images and metrics"),
    fixed_img: str = typer.Argument(..., help="Type of fixed image: ['multiplexed', 'hne']"),
    fixed_px_sz: float = typer.Option(None, help="Pixel size of the fixed image (if image is not .ome.tif)"),
    moving_px_sz: float = typer.Option(None, help="Pixel size of the moving image (if image is not .ome.tif)"),
    feature_tform: str = typer.Option('similarity', help="Feature transformation method ['similarity', 'affine', 'projective']. 'similarity' by default and recommended.", show_default=True)
):
    """
    Sub-command to register H&E stained images to multiplexed images using feature based registration. Saves registered image, transformation maps and registration metrics to the specified output folder.

    Parameters:
    - fixed_path (str): Path to the fixed image (.tif/.tiff/.ome.tif/.ome.tiff)
    - moving_path (str): Path to the moving image (.tif/.tiff/.ome.tif/.ome.tiff)
    - output_folder (str): Folder to save the registered images and metrics
    - fixed_img (str): Type of fixed image: ['multiplexed', 'hne']
    - fixed_px_sz (float, optional): Pixel size of the fixed image (if image is not .ome.tif)
    - moving_px_sz (float, optional): Pixel size of the moving image (if image is not .ome.tif)
    - feature_tform (str, optional): Feature transformation method ['similarity', 'affine', 'projective']. 'similarity' by default and recommended. 

    Returns:
    - None
    """
    
    # run the pipeline
    transformation_map, final_img, tre, mi = registration_pipeline(
        fixed_path,
        moving_path,
        fixed_px_sz,
        moving_px_sz,
        fixed_img,
        feature_tform=feature_tform
    )

    os.makedirs(output_folder, exist_ok=True)

    # save registration metrics
    metrics_output_path = os.path.join(output_folder, "registration_metrics.json")

    with open(metrics_output_path, "w") as f:
        json.dump({"TRE": tre, "Mutual Information": mi}, f)
    print(f"Registration metrics saved to {metrics_output_path}")

    # save registered image
    ome_xml = None
    try:
        with TiffFile(moving_path) as ref:
            ome_xml = ref.ome_metadata
    except:
        pass

    final_img_path = os.path.join(output_folder, "0_final_channel_image.ome.tif")
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
    img_path = os.path.join(output_folder_path, f"multiplexed_channel_{channel_idx}.tif")
    imwrite(img_path, img_ch)
    print(f"Image with extracted channel saved to {img_path}")



@app.command(name="transform-seg-mask")
def transform_seg_mask_cmd(
    mask_path: str = typer.Argument(..., help="Path to the segmentation mask of the moving image (.npy)"),
    fixed_path: str = typer.Argument(..., help="Path to the fixed image (.tif/.tiff/.ome.tif/.ome.tiff)"),
    output_folder_path: str = typer.Argument(..., help="Folder to save the transformed segmentation mask"),
    tform_map_path: str = typer.Argument(..., help="Path to the transformation map"),
    moving_px_sz: str = typer.Argument(..., help="Path to moving image if .ome.tiff or Pixel size of the moving image"),
    fixed_px_sz: float = typer.Option(None, help="Pixel size of the fixed image (if image is not .ome.tif)", show_default=True)
):
    """
    Sub-command to transform a segmentation mask from the moving image space to the fixed image space using the provided transformation map.
    
    Parameters:
    - mask_path (str): Path to the segmentation mask of the moving image (.npy)
    - fixed_path (str): Path to the fixed image (.tif/.tiff/.ome.tif/.ome.tiff)
    - output_folder_path (str): Folder to save the transformed segmentation mask
    - tform_map_path (str): Path to the transformation map
    - moving_px_sz (str): Path to moving image if .ome.tiff or Pixel size of the moving image
    - fixed_px_sz (float, optional): Pixel size of the fixed image (if image is not .ome.tif)

    Returns:
    - None
    """
    
    # load mask
    mask = np.load(mask_path) # will need to change according to mask format
    print(f"Loaded segmentation mask.")

    # load and create transformation parameter objects
    transformation_maps= AffineTransform(matrix=np.load(os.path.join(tform_map_path)))

    print("Loaded transformation map.")

    fixed_init = load_image_data(fixed_path)

    if fixed_px_sz is None:
        try:
            fixed_px_sz, _ = get_pixel_size_ome_tiff(fixed_path)
        except Exception:
            fixed_px_sz = None
        
        if fixed_px_sz is None:
            raise ValueError("Pixel size information not found in metadata for fixed image. Please provide fixed_px_sz.")
    
    try:
        moving_px_sz, _ = get_pixel_size_ome_tiff(moving_px_sz)
    except:
        pass

    try:
        scale = float(moving_px_sz) / fixed_px_sz
    except:
        raise ValueError("Could not determine moving image pixel sizes for scaling. Please check the provided pixel size or moving image path (ome.tiff).")
    
    if len(fixed_init.shape) == 2:
        fixed_init_sc = resize(fixed_init, (int(fixed_init.shape[0]/scale), int(fixed_init.shape[1]/scale)), anti_aliasing=True)
    else:
        fixed_init_sc = resize(fixed_init, (int(fixed_init[:, :, 0].shape[0]/scale), int(fixed_init[:, :, 0].shape[1]/scale)), anti_aliasing=True)
    fixed_img_shape = (int(fixed_init_sc.shape[0]), int(fixed_init_sc.shape[1]))

    moved_mask = transform_seg_mask(mask, transformation_maps, output_shape=fixed_img_shape)

    os.makedirs(output_folder_path, exist_ok=True)
    np.save(os.path.join(output_folder_path, "transformed_segmentation_mask.npy"), moved_mask)
    print(f"Transformed segmentation mask saved to {output_folder_path}/transformed_segmentation_mask.npy")



def main():
    app()


if __name__ == "__main__":
    main()