from skimage.transform import warp, resize
import numpy as np
from .preprocess import load_and_scale_images, colour_deconvolusion_preprocessing_HnE, extract_channel, get_image_size_ome_tiff
from .reg import register_DAPI_HnE
from .metrics import compute_TRE, compute_mutual_information

def registration_pipeline(fixed_path, moving_path, fixed_px_sz, moving_px_sz, fixed_img, final_img_sz, feature_tform='similarity', chnl_idx=0):
    """
    Pipeline for registering images. Loads and scales images, preprocesses them, performs registration, and computes registration 
    metrics.

    Parameters:
    - fixed_path (str): Path to the fixed image
    - moving_path (str): Path to the moving image
    - fixed_px_sz (float): Pixel size of the fixed image (if image is not .ome.tif)
    - moving_px_sz (float): Pixel size of the moving image (if image is not .ome.tif)
    - fixed_img (str): Type of fixed image: ['multiplexed', 'hne']
    - final_img_sz (str): Pixel size for final image: ['fixed', 'moving']
    - feature_tform (str): Type of transformation to estimate ('similarity', 'affine', 'projective')
    - chnl_idx (int): Channel index (DAPI) to extract if channel extraction not done beforehand for multiplexed image

    Returns:
    - transformation_maps (skimage.transform): Estimated transformation object
    - moved_img (ndarray): Registered moving image
    - tre (dict or None): Dictionary with TRE values before and after registration if computed, else None
    - mi (dict or None): Dictionary with Mutual Information (MI) values before and after registration if computed, else None
    """
    
    # load and scale images 
    fixed_init, moving_init = load_and_scale_images(fixed_path, moving_path, fixed_px_sz, moving_px_sz, chnl_idx)
    print("Images loaded.")

    # preprocess HnE image
    if fixed_img == 'multiplexed':
        moving_prepr = colour_deconvolusion_preprocessing_HnE(moving_init)
        fixed_prepr = fixed_init if len(fixed_init.shape) == 2 else extract_channel(fixed_init, chnl_idx)
    elif fixed_img == 'hne':
        fixed_prepr = colour_deconvolusion_preprocessing_HnE(fixed_init)
        moving_prepr = moving_init if len(moving_init.shape) == 2 else extract_channel(moving_init, chnl_idx)
    else:
        raise ValueError("At least one of the images must be an HnE stained image with 3 channels and fixed_img parameter must be either 'multiplexed' or 'hne'.")
    print("Preprocessing completed.")
    
    
    # registration
    if len(fixed_init.shape) == 2:
        h, w = fixed_init.shape
    else:
        h, w, c = fixed_init.shape

    transformation_maps, registered_imgs, tre_pts = register_DAPI_HnE(fixed_prepr, moving_prepr, feature_tform)
    moved_img = warp(moving_init, transformation_maps.inverse, output_shape=(h, w, moving_init.shape[2]) if len(moving_init.shape) == 3 else (h, w))


    # set final image size
    try:
        if final_img_sz == 'fixed':
            target_shape = get_image_size_ome_tiff(fixed_path)

            if len(moved_img.shape) == 2:
                moved_img = resize(moved_img, (target_shape[0], target_shape[1]), anti_aliasing=True)

            elif len(moved_img.shape) == 3:
                moved_img = resize(moved_img, (target_shape[0], target_shape[1], moved_img.shape[2]), anti_aliasing=True)

        elif final_img_sz == 'moving':
            moved_img = moved_img

    except:
        print("Final image resizing skipped and kept at original moving image size. 'final_img_sz' parameter should be either 'fixed' or 'moving'.")


    # set final image data type
    try: 
        if fixed_img == 'multiplexed':
            moved_img = np.clip(moved_img, 0, 1)
            moved_img = np.rint(moved_img * 255).astype(np.uint8) # 0-255 for HnE
        else:
            moved_img = np.clip(moved_img, 0, 1)
            moved_img = (moved_img * 65535).astype(np.uint16)
    except:
        moved_img = np.clip(moved_img, 0, 1)
        moved_img = (moved_img * 65535).astype(np.uint16)


    # evaluate registration with metrics
    try:
        tre = compute_TRE(transformation_maps, tre_pts, fixed_prepr)
    except ValueError as e:
        print("TRE computation skipped:", e)
        tre = None  
    except Exception as e:
        print("An unexpected error occurred during TRE computation:", e)
        tre = None

    try:
        mi = compute_mutual_information(fixed_prepr, moving_prepr, registered_imgs)
    except Exception as e:
        print("An unexpected error occurred during mutual information computation:", e)
        mi = None

    return transformation_maps, moved_img, tre, mi


