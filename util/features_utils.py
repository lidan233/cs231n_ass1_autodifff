import matplotlib as mpb
import numpy as np
from scipy.ndimage import uniform_filter

def extract_features(imgs,feature_fns,verbose=False):
    """
    lidan
    :param imgs:  N x H X W X C array of pixel data for N images.
    :param feature_fns: List of k feature functions. The ith feature function should
    take as input an H x W x D array and return a (one-dimensional) array of
    length F_i.
    :param verbose: Boolean; if true, print progress.
    :return: 
    An array of shape (N, F_1 + ... + F_k) where each column is the concatenation
    of all features for a single image.
    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # Use the first image to determine feature dimensions

    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feat = feature_fn(imgs[0].squeeze())
        assert len(feat.shape)==1,'feature function must be one dimension'
        feature_dims.append(feat.size)
        first_image_features.append(feat)

    # Now that we know the dimensions of the features, we can allocate a single
    # big array to store all features as columns

    total_feature_nums = sum(feature_dims)
    images_features = np.zeros((num_images,total_feature_nums))
    images_features[0]  = np.hstack(first_image_features)

    # Extract features for the rest of the images.
    # my implements
    # for i in range(1,num_images):
    #     temp = []
    #     for feature_fn in feature_fns:
    #         feat = feature_fn(imgs[i].squeeze())
    #         assert len(feat.shape)==1,'feature function must be one dimension'
    #         temp.append(feat)
    #     images_features[i] = np.hstack(temp)

    # source implements
    for i in range(1,num_images):
        idx = 0
        for feature_fn,feature_dim in zip(feature_fns,feature_dims):
            target_idx = idx+feature_dim
            images_features[i,idx:target_idx] = feature_fn(imgs[i].squeeze())
            idx = target_idx
        if verbose and i%1000==0:
            print('Done extracting features for %d / %d images' % (i, num_images))

    return images_features
def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.299, 0.587, 0.144])


#???
def hog_feature(im):
    # get the image's 9 gradient image 
    if im.ndim==3:
        image = rgb2gray(im)
    else:
        # the inverse manipulation of squeeze
        image = np.atleast_2d(im)
    sx,sy = image.shape
    orientations = 9  # number of gradient bins
    cx, cy = (8, 8)

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # compute gradient on x-direction
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # compute gradient on y-direction
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # gradient magnitude
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # gradient orientation

    n_cellsx = int(np.floor(sx / cx))  # number of cells in x
    n_cellsy = int(np.floor(sy / cy))  # number of cells in y


    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        # create new integral image for this orientation
        # isolate orientations in this range
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                            grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i,
                            temp_ori, 0)
        # select magnitudes for those orientations
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        orientation_histogram[:, :, i] = uniform_filter(temp_mag, size=(cx, cy))[(int)(cx / 2)::cx, (int)(cy / 2)::cy].T
    # uniform is mean filter
    return orientation_histogram.ravel()



def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
  """
  Compute color histogram for an image using hue.

  Inputs:
  - im: H x W x C array of pixel data for an RGB image.
  - nbin: Number of histogram bins. (default: 10)
  - xmin: Minimum pixel value (default: 0)
  - xmax: Maximum pixel value (default: 255)
  - normalized: Whether to normalize the histogram (default: True)

  Returns:
    1D vector of length nbin giving the color histogram over the hue of the
    input image.
  """
  ndim = im.ndim
  bins = np.linspace(xmin, xmax, nbin+1)
  hsv = mpb.colors.rgb_to_hsv(im/xmax) * xmax
  imhist, bin_edges = np.histogram(hsv[:,:,0], bins=bins, density=normalized)
  imhist = imhist * np.diff(bin_edges)

  # return histogram
  return imhist


pass
