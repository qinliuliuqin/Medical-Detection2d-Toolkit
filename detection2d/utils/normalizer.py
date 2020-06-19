import numpy as np


class AdaptiveNormalizer(object):
  """
  Normalize image using z-score normalization.
  """

  def __init__(self, clip_sigma=3, percentile=1):
    """
    :param clip_sigma: clip the intensity within the 'clip_sigma' standard deviation. 68% voxels lies within 1
      standard deviation, 95% within 2 standard deviation, and 99.7% within 3 standard deviation.
    """
    assert clip_sigma > 0
    self.clip_sigma = clip_sigma
    self.percentile = min(100, max(0, percentile))

  def normalize(self, single_image):
    """ Normalize a given image """
    assert isinstance(single_image, np.ndarray), 'image must be an numpy array object'

    image = single_image.astype(dtype=np.float32)

    for color_idx in range(3):
        color_plane = image[:, :, color_idx]

        min_val = np.percentile(color_plane, self.percentile)
        max_val = np.percentile(color_plane, 100 - self.percentile)

        color_plane[color_plane < min_val] = min_val
        color_plane[color_plane > max_val] = max_val

        mean, stddev = np.mean(color_plane), np.std(color_plane)
        if stddev < 1e-6:
            return single_image

        normalized_color_plane = (color_plane - mean) / stddev
        normalized_color_plane[normalized_color_plane > self.clip_sigma] = self.clip_sigma
        normalized_color_plane[normalized_color_plane < -self.clip_sigma] = -self.clip_sigma
        image[:, :, color_idx] = normalized_color_plane

    return image

  def __call__(self, image):
    """ normalize image """
    if isinstance(image, np.ndarray):
      return self.normalize(image)

    elif isinstance(image, list):
      for idx, im in enumerate(image):
        assert isinstance(im, np.ndarray)
        image[idx] = self.normalize(im)
      return image

    else:
      raise ValueError('Unknown type of input. Normalizer only supports Image3d or Image3d list/tuple')