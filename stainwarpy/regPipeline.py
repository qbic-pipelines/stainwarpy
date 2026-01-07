from skimage.transform import warp
from .preprocess import load_and_scale_images, colour_deconvolusion_preprocessing_HnE, extract_channel
from .reg import register_DAPI_HnE
from .metrics import compute_TRE, compute_mutual_information

def registration_pipeline(fixed_path, moving_path, fixed_px_sz, moving_px_sz, fixed_img, feature_tform='similarity'):
    """
    Pipeline for registering images. Loads and scales images, preprocesses them, performs registration, and computes registration 
    metrics.

    Parameters:
    - fixed_path (str): Path to the fixed image
    - moving_path (str): Path to the moving image
    - fixed_px_sz (float): Pixel size of the fixed image (if image is not .ome.tif)
    - moving_px_sz (float): Pixel size of the moving image (if image is not .ome.tif)
    - fixed_img (str): Type of fixed image: ['multiplexed', 'hne']
    - feature_tform (str): Type of transformation to estimate ('similarity', 'affine', 'projective')

    Returns:
    - transformation_maps (skimage.transform): Estimated transformation object
    - moved_img (ndarray): Registered moving image
    - tre (dict or None): Dictionary with TRE values before and after registration if computed, else None
    - mi (dict or None): Dictionary with Mutual Information (MI) values before and after registration if computed, else None
    """
    
    # load and scale images 
    fixed_init, moving_init = load_and_scale_images(fixed_path, moving_path, fixed_px_sz, moving_px_sz)
    print("Images loaded.")

    # preprocess HnE image
    if fixed_img == 'multiplexed':
        moving_prepr = colour_deconvolusion_preprocessing_HnE(moving_init)
        fixed_prepr = fixed_init
    elif fixed_img == 'hne':
        fixed_prepr = colour_deconvolusion_preprocessing_HnE(fixed_init)
        if len(moving_init.shape) == 2:
            moving_prepr = moving_init
        else:
            moving_prepr = extract_channel(moving_init, 0)
    else:
        raise ValueError("fixed_img must be either 'multiplexed' or 'hne'")
    print("Preprocessing completed.")
    
    
    # registration
    if len(fixed_init.shape) == 2:
        h, w = fixed_init.shape
    else:
        h, w, c = fixed_init.shape

    transformation_maps, registered_imgs, tre_pts = register_DAPI_HnE(fixed_prepr, moving_prepr, feature_tform)
    moved_img = warp(moving_init, transformation_maps.inverse, output_shape=(h, w, moving_init.shape[2]) if len(moving_init.shape) == 3 else (h, w))


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


