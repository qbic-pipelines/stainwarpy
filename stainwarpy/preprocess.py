import numpy as np
import xml.etree.ElementTree as ET
from tifffile import imread, TiffFile, TiffWriter
from skimage.transform import resize
import collections

"""
This file contains parts of code adapted from HistomicsTK
(https://github.com/DigitalSlideArchive/HistomicsTK/), licensed under Apache License 2.0.
"""

def rgb_to_sda(im_rgb):
    """
    From HistomicsTK.
    Convert RGB image to stain density array (SDA) using Beer-Lambert law. 

    Parameters:
    - im_rgb (ndarray): Input RGB image

    Returns:
    - im_sda (ndarray): Stain density array
    """

    is_matrix = im_rgb.ndim == 2
    if is_matrix:
        im_rgb = im_rgb.T
    
    im_rgb = im_rgb.astype(float) + 1
    I_0 = 256

    im_rgb = np.maximum(im_rgb, 1e-10)

    im_sda = -np.log(im_rgb / (1. * I_0)) * 255 / np.log(I_0)
    im_sda = np.maximum(im_sda, 0)

    return im_sda.T if is_matrix else im_sda



def sda_to_rgb(im_sda):
    """
    From HistomicsTK.
    Convert stain density array (SDA) back to RGB image using inverse Beer-Lambert law.

    Parameters:
    - im_sda (ndarray): Stain density array

    Returns:
    - im_rgb (ndarray): Reconstructed RGB image
    """

    is_matrix = im_sda.ndim == 2
    if is_matrix:
        im_sda = im_sda.T
        
    I_0 = 256

    im_rgb = I_0 ** (1 - im_sda / 255.)
    return (im_rgb.T if is_matrix else im_rgb) - True



def colour_deconvolusion(hne_init, W):
    """
    From HistomicsTK.
    Perform color deconvolution on an image using a given stain matrix.

    Parameters:
    - hne_init (ndarray): Input image
    - W (ndarray): Stain matrix

    Returns:
    - Output (namedtuple): Contains deconvolved stains and related information
    """

    w = np.array(W)

    if w.shape[1] < 3:
        wc = np.zeros((w.shape[0], 3))
        wc[:, :w.shape[1]] = w
        w = wc

    if np.linalg.norm(w[:, 2]) <= 1e-16:
        stain0 = w[:, 0]
        stain1 = w[:, 1]
        stain2 = np.cross(stain0, stain1)
        wc = np.array([stain0, stain1, stain2 / np.linalg.norm(stain2)]).T
    else:
        wc = w

    # normalize stains to unit-norm
    wc = wc / np.sqrt((wc ** 2).sum(0))

    # invert stain matrix
    Q = np.linalg.pinv(wc)

    # transform 3D input image to 2D RGB matrix format
    m_tmp = hne_init if hne_init.ndim == 2 else hne_init.reshape((-1, hne_init.shape[-1])).T
    m = m_tmp[:3]

    # transform input RGB to optical density values and deconvolve,
    # tfm back to RGB
    sda_fwd = rgb_to_sda(m)
    sda_deconv = np.dot(Q, sda_fwd)
    sda_inv = sda_to_rgb(sda_deconv)
                                    

    # reshape output
    StainsFloat = sda_inv if len(hne_init.shape) == 2 else sda_inv.T.reshape(hne_init.shape[:-1] + (sda_inv.shape[0],))

    # transform type
    Stains = StainsFloat.clip(0, 255).astype(np.uint8)

    # return
    Unmixed = collections.namedtuple('Unmixed',
                                     ['Stains', 'StainsFloat', 'Wc'])
    Output = Unmixed(Stains, StainsFloat, wc)

    return Output



def colour_deconvolusion_preprocessing_HnE(hne_init):
    """
    Color deconvolution preprocessing for HnE stained images.

    Parameters:
    - hne_init (ndarray): Input HnE stained image

    Returns:
    - hne_deconv (ndarray): Hematoxylin channel after color deconvolution
    """

    # create stain matrix (columns correspond to stains hematoxylin, eosin, null)
    W = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.0, 0.0, 0.0]]).T

    # perform standard color deconvolution
    imDeconvolved = colour_deconvolusion(hne_init, W)
    hne_deconv = 1 - imDeconvolved.Stains[:, :, 0]

    return hne_deconv



def get_image_size_ome_tiff(file_path):
    """
    Get image size from an OME-TIFF file.

    Parameters:
    - file_path (str): Path to the OME-TIFF file

    Returns:
    - shape (tuple): Shape of the image (height, width)
    """

    with TiffFile(file_path) as tif:
        img = tif.series[0].asarray()
        shape = img.shape[0:2] if img.ndim == 2 or img.shape[0] < img.shape[2] else img.shape[1:3] 
        return shape



def get_pixel_size_ome_tiff(file_path):
    """
    Get pixel size from an OME-TIFF file.

    Parameters:
    - file_path (str): Path to the OME-TIFF file

    Returns:
    - px (float): Physical size of a pixel in X direction (µm)
    - py (float): Physical size of a pixel in Y direction (µm)
    """
    
    with TiffFile(file_path) as tif:
        ome = tif.ome_metadata
        if ome is None:
            raise ValueError(f"Not an OME-TIFF: {file_path}")

        root = ET.fromstring(ome)
        pixels = root.find(".//{*}Pixels")   

        px = pixels.get("PhysicalSizeX")
        py = pixels.get("PhysicalSizeY")

        px = float(px) if px is not None else None
        py = float(py) if py is not None else None

        return px, py



def load_image_data(file_path):
    """
    Load image data from a file and arrange dimensions as (Y, X, C) or (Y, X).

    Parameters:
    - file_path (str): Path to the image file

    Returns:
    - img (ndarray): Loaded image
    """
    
    if file_path.endswith(".tif") or file_path.endswith(".tiff"):
        img_raw = imread(file_path)
        img = np.array(img_raw) 

        return img if (len(img.shape) == 2) or (img.shape[2] < img.shape[0]) else img.transpose(1, 2, 0)
    
    else: 
        raise ValueError("Unsupported file format. Please provide a .tif file.")



def extract_channel(img, channel_index):
    """
    Extract a specific channel from a multi-channel image.

    Parameters:
    - img (ndarray): Input multi-channel image
    - channel_index (int): Index of the channel to extract

    Returns:
    - channel_img (ndarray): Extracted channel image
    """
    
    return img[:, :, channel_index]



def load_and_scale_images(fixed_path, moving_path, fixed_px_sz, moving_px_sz):
    """
    Load and scale fixed and moving images based on their pixel sizes, moving image is scaled to match fixed image.

    Parameters:
    - fixed_path (str): Path to the fixed image file
    - moving_path (str): Path to the moving image file
    - fixed_px_sz (float or None): Pixel size of the fixed image (µm). If None, will attempt to read from metadata.
    - moving_px_sz (float or None): Pixel size of the moving image (µm). If None, will attempt to read from metadata.

    Returns:
    - fixed_init (ndarray): Loaded and scaled fixed image
    - moving_init (ndarray): Loaded moving image
    """

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
        except Exception:
            moving_px_sz = None

        if moving_px_sz is None:
            raise ValueError("Pixel size information not found in metadata for moving image. Please provide moving_px_sz.")

    scale = moving_px_sz / fixed_px_sz

    # load fixed image
    fixed_img = load_image_data(fixed_path)
    if len(fixed_img.shape) == 2:
        fixed_init = resize(fixed_img, (int(fixed_img.shape[0]/scale), int(fixed_img.shape[1]/scale)), anti_aliasing=True)
    elif fixed_img.shape[2] == 3:
        fixed_init = resize(fixed_img, (int(fixed_img.shape[0]/scale), int(fixed_img.shape[1]/scale), fixed_img.shape[2]), anti_aliasing=True)
    elif fixed_img.shape[2] > 3:
        # extract specified channel (DAPI) if multiplexed
        fixed_ch_img = extract_channel(fixed_img, 0)
        fixed_init = resize(fixed_ch_img, (int(fixed_ch_img.shape[0]/scale), int(fixed_ch_img.shape[1]/scale)), anti_aliasing=True)
    fixed_init = fixed_init*255

    # load moving image
    moving_init = load_image_data(moving_path)

    return fixed_init, moving_init



def save_ome_tiff(
    img,
    out_path,
    channel_names=None,
    physical_size_x=None,
    physical_size_y=None,
    source_ome_xml=None):
    """
    Save an image as an OME-TIFF file with specified metadata.

    Parameters:
    - img (ndarray): Image to be saved
    - out_path (str): Output file path
    - channel_names (list of str or None): Names of the channels. If None, will attempt to read from source_ome_xml or auto-generate.
    - physical_size_x (float or None): Physical size of a pixel in X direction (µm). 
    - physical_size_y (float or None): Physical size of a pixel in Y direction (µm)
    - source_ome_xml (str or None): Source OME-XML metadata to extract channel names if channel_names is None
    """

    # prepare data in CYX format
    if img.ndim == 2:
        # grayscale (Y, X)
        Y, X = img.shape
        C = 1
        img = img.reshape(Y, X, 1)
        data = img.transpose(2, 0, 1)

    elif img.ndim == 3:
        if img.shape[2] == 3:
            # RGB (Y, X, 3)
            Y, X, C = img.shape
            data = img.transpose(2, 0, 1)

        elif img.shape[2] > 3:
            # multiplexed (Y, X, C)
            Y, X, C = img.shape
            data = img.transpose(2, 0, 1)

        else:
            raise ValueError(f"Unsupported shape {img.shape}: no alpha allowed and no Z/T.")

    else:
        raise ValueError(f"Unsupported ndim={img.ndim}")

    # determine channel names
    if channel_names is None:
        try:
            if source_ome_xml is not None:
                root = ET.fromstring(source_ome_xml)
                channel_names = [c.get("Name") for c in root.findall(".//{*}Channel")]
            else:
                # if not provided, auto-generate
                channel_names = [f"Channel_{i}" for i in range(C)]
        except:
            channel_names = [f"Channel_{i}" for i in range(C)]

    if len(channel_names) != C:
        raise ValueError(f"Channel name count {len(channel_names)} does not match C={C}")
    
   # save OME-TIFF 
    with TiffWriter(out_path, bigtiff=True) as tif:
         metadata={
             'axes': 'CYX',
             'PhysicalSizeX': physical_size_x,
             'PhysicalSizeXUnit': 'µm',
             'PhysicalSizeY': physical_size_y,
             'PhysicalSizeYUnit': 'µm',
             'Channel': {'Name': channel_names},
         }

         tif.write(
             data,
             resolution=(Y, X),
             metadata=metadata,
         )
