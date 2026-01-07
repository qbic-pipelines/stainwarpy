import random
from skimage.transform import resize, estimate_transform, warp, AffineTransform
from skimage.feature import SIFT, match_descriptors
from skimage import measure


def transform_seg_mask(mask, transformation_maps, output_shape):
    """
    Trasform a segmentation mask using the provided transformation maps.

    Parameters:
    - mask (ndarray): Segmentation mask to be transformed
    - transformation_maps (skimage.transform): Transformation object
    - output_shape (tuple): Desired output shape of the transformed mask (shape of fixed image)

    Returns:
    - moved_mask (ndarray): Transformed segmentation mask
    """

    moved_mask = warp(mask, transformation_maps.inverse, output_shape=output_shape, order=0, preserve_range=True)

    return moved_mask



def features_with_SIFT(fixed, moving, max_ratio=0.6, n_octaves=3, n_scales=5):
    """
    Detect and match features between fixed and moving images using SIFT feature detector.

    Parameters:
    - fixed (ndarray): Fixed image
    - moving (ndarray): Moving image
    - max_ratio (float): Maximum ratio for descriptor matching
    - n_octaves (int): Number of octaves for SIFT
    - n_scales (int): Number of scales per octave for SIFT

    Returns:
    - matches (list of ndarray): List containing matched keypoints from moving and fixed images [moving_matches, fixed_matches]
    """
    
    fixedX, fixedY = fixed.shape
    movingX, movingY = moving.shape
    scale_factor = 4

    # Determine scale factor based on image size
    if fixedX > 2000 or fixedY > 2000:
        scale_factor = max(fixedX // 2000, fixedY // 2000) * 4

    elif fixedX < 250 and fixedY < 250:
        scale_factor = 1

    # Resize the images to reduce memory usage
    fixed_scaled = resize(fixed, (fixedX // scale_factor, fixedY // scale_factor), anti_aliasing=True)
    moving_scaled = resize(moving, (movingX // scale_factor, movingY // scale_factor), anti_aliasing=True)

    # Initialize SIFT detector
    descriptor_extractor = SIFT(n_octaves=n_octaves, n_scales=n_scales)

    # Detect and extract features
    descriptor_extractor.detect_and_extract(moving_scaled)
    keypoints1, descriptors1 = descriptor_extractor.keypoints, descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(fixed_scaled)
    keypoints2, descriptors2 = descriptor_extractor.keypoints, descriptor_extractor.descriptors

    # Match descriptors between images
    matches12 = match_descriptors(
        descriptors1, descriptors2, max_ratio=max_ratio, cross_check=True
    )

    if matches12.shape[0] < 3:
        raise ValueError("Not enough matching points found between images for reliable registration.")

    # Extract matched keypoints
    src, dst = keypoints1[matches12[:, 0]], keypoints2[matches12[:, 1]]

    dst, src = dst * scale_factor, src * scale_factor

    # Compute inliers using RANSAC 
    _, inliers = measure.ransac((dst, src),
                               AffineTransform, min_samples=4,
                               residual_threshold=2, max_trials=1000)
    
    # Filter inliers
    movingtemp_matches = src[inliers] 
    fixedtemp_matches = dst[inliers] 
    
    # Prepare output matches changed to (x, y) format
    moving_matches = movingtemp_matches[:, [1, 0]].copy()
    fixed_matches = fixedtemp_matches[:, [1, 0]].copy()

    return [moving_matches, fixed_matches]



def register_feature_based(fixed, moving, feature_tform):
    """
    Perform feature based registration between fixed and moving images, if sufficient matching points are found. Compute transformation map and aligned moving image.
    Compute points for TRE calculation.

    Parameters:
    - fixed (ndarray): Fixed image
    - moving (ndarray): Moving image
    - feature_tform (str): Type of transformation to estimate ('similarity', 'affine', 'projective')

    Returns:
    - tform (skimage.transform): Estimated transformation object  
    - aligned_moving (ndarray): Aligned moving image after applying the transformation
    - tre_points (tuple of ndarray): Tuple of (src_points, dst_points) for TRE calculation
    - reg_points (tuple of ndarray): Tuple of (src_points, dst_points) used for registration  
    """

    [moving_matches, fixed_matches] = features_with_SIFT(fixed, moving)

    num_matches = moving_matches.shape[0]

    if num_matches < 3:
        raise ValueError(f"At least three matching points are required for initial feature based registration, only {num_matches} found.")
    
    num_tre_points = min(6, num_matches - 3, num_matches // 2)
    all_idx = set(range(num_matches))
    tre_idx = random.sample(range(num_matches), num_tre_points)
    other_idx = list(all_idx - set(tre_idx))
    moving_pts_for_reg, fixed_pts_for_reg = moving_matches[other_idx], fixed_matches[other_idx]

    tform = estimate_transform(feature_tform, src=moving_pts_for_reg, dst=fixed_pts_for_reg)
    aligned_moving = warp(moving, tform.inverse, output_shape=fixed.shape)

    return tform, aligned_moving, [moving_matches[tre_idx], fixed_matches[tre_idx]], [moving_pts_for_reg, fixed_pts_for_reg]



def register_DAPI_HnE(fixed, moving, feature_tform='similarity'):
    """
    Main function to register DAPI channel of multiplexed image to H&E stained image using feature based registration.

    Parameters:
    - fixed (ndarray): Fixed image (DAPI channel of multiplexed image)
    - moving (ndarray): Moving image (H&E stained image)
    - feature_tform (str): Type of transformation to estimate ('similarity', 'affine', 'projective')

    Returns:
    - tform_map (skimage.transform): Estimated transformation object
    - aligned_moving (ndarray): Aligned moving image after applying the transformation
    - tre_points (tuple of ndarray): Tuple of (src_points, dst_points) for TRE calculation
    """

    tform_map, moving_img_aligned, [moving_tre_pts, fixed_tre_pts], [moving_reg_pts, fixed_reg_pts] = register_feature_based(fixed, moving, feature_tform)

    print('Feature based registration completed.')

    return tform_map, moving_img_aligned, [moving_tre_pts, fixed_tre_pts]



    