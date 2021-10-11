import albumentations.augmentations.functional as F
import cv2
import numpy as np
from skimage.segmentation import find_boundaries


def make_weight_map(masks, w0=10, sigma=5, longest_max_size=-1):
    """
    Generate the weight maps as specified in the UNet paper
    for a set of binary masks.

    Parameters
    ----------
    masks: array-like
        A 3D array of shape (n_masks, image_height, image_width),
        where each slice of the matrix along the 0th axis represents one binary mask.
    w0: positive float
        maximal value of the weights mask. Actual value might be slightly higher due to
        the interpolation.
    sigma: positive float
        standard deviation of the squared sum of distances to the border
        of the two nearest cells.
    longest_max_size: int
        size of the longest mask's side. Weight map building is memory consuming on high
        resolution mask and rescaling to the lower resolution mask is used to avoid
        long processing and memory consuming.

    Returns
    -------
    array-like
        A 2D array of shape (image_height, image_width)

    """
    nrows, ncols = masks.shape[1:]

    if longest_max_size > 0:
        old_rows, old_cols = nrows, ncols
        max_size = max(nrows, ncols)
        new_rows, new_cols = longest_max_size * nrows // max_size, longest_max_size * ncols // max_size

        resized_masks = []
        for mask in masks:
            resized_masks.append(cv2.resize(mask, (new_cols, new_rows), interpolation=0))
        masks = np.stack(resized_masks)
        nrows, ncols = new_rows, new_cols

    masks = (masks > 0).astype(int)
    distMap = np.zeros((nrows * ncols, masks.shape[0]))
    X1, Y1 = np.meshgrid(np.arange(nrows), np.arange(ncols))
    X1, Y1 = np.c_[X1.ravel(), Y1.ravel()].T
    for i, mask in enumerate(masks):
        # find the boundary of each mask,
        # compute the distance of each pixel from this boundary
        bounds = find_boundaries(mask, mode='inner')
        X2, Y2 = np.nonzero(bounds)
        xSum = (X2.reshape(-1, 1) - X1.reshape(1, -1)) ** 2
        ySum = (Y2.reshape(-1, 1) - Y1.reshape(1, -1)) ** 2
        distMap[:, i] = np.sqrt(xSum + ySum).min(axis=0)
    ix = np.arange(distMap.shape[0])
    if distMap.shape[1] == 1:
        d1 = distMap.ravel()
        border_loss_map = w0 * np.exp((-1 * (d1) ** 2) / (2 * (sigma ** 2)))
    else:
        if distMap.shape[1] == 2:
            d1_ix, d2_ix = np.argpartition(distMap, 1, axis=1)[:, :2].T
        else:
            d1_ix, d2_ix = np.argpartition(distMap, 2, axis=1)[:, :2].T
        d1 = distMap[ix, d1_ix]
        d2 = distMap[ix, d2_ix]
        border_loss_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))
    xBLoss = np.zeros((nrows, ncols))
    xBLoss[X1, Y1] = border_loss_map
    # class weight map
    loss = np.zeros((nrows, ncols))
    w_1 = 1 - masks.sum() / loss.size
    w_0 = 1 - w_1
    loss[masks.sum(0) == 1] = w_1
    loss[masks.sum(0) == 0] = w_0
    ZZ = xBLoss + loss

    if longest_max_size > 0:
        ZZ = cv2.resize(ZZ, (old_cols, old_rows))
    return ZZ


def convert_to_square(bboxes):
    """Convert bounding boxes to a square form.

    Arguments:
        bboxes: a float numpy array of shape [n, 5].

    Returns:
        a float numpy array of shape [n, 5],
            squared bounding boxes.
    """

    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    square_bboxes[:, 4:] = bboxes[:, 4:]
    return np.round_(square_bboxes)


def dist(p1, p2):
    return ((np.array(p2) - np.array(p1)) ** 2).sum() ** 0.5


def rotate_and_crop_rectangle_safe(image, angle, box=None, pad=None, interpolation=1, **kwargs):
    rows, cols = image.shape[:2]
    if box is None:
        box = np.array([[0, 0], [0, rows], [cols, rows], [cols, 0]])
    if pad is None:
        pad = [0, 0, 0, 0]
    h_min_pad, w_min_pad, h_max_pad, w_max_pad = pad

    cx, cy = box.mean(axis=0).astype(int)
    w = int(dist(box[1], box[2]))
    h = int(dist(box[0], box[1]))
    d = int((w ** 2 + h ** 2) ** 0.5) + 1

    h_pad_top = d - cy if cy < d else 0
    h_pad_bottom = d - (rows - cy) if (rows - cy) < d else 0
    w_pad_left = d - cx if cx < d else 0
    w_pad_right = d - (cols - cx) if (cols - cx) < d else 0

    img = cv2.copyMakeBorder(image.copy(), h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, 0)

    cx += w_pad_left
    cy += h_pad_top

    w = int(dist(box[1], box[2]))
    h = int(dist(box[0], box[1]))

    h, w = min(w, h), max(w, h)
    if isinstance(h_max_pad, float):
        h_max_pad = int(h_max_pad * h)
        w_max_pad = int(w_max_pad * w)
        h_min_pad = max(int(h_min_pad * h) - 1, 0)
        w_min_pad = max(int(w_min_pad * w) - 1, 0)

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    dst = cv2.warpAffine(img.astype(float), M, (w // 2 + cx + w_max_pad, h // 2 + cy + h_max_pad), flags=interpolation)

    dst = dst[cy - h // 2 - h_min_pad:, cx - w // 2 - w_min_pad:]
    return dst.astype('uint8')


def rotate_and_crop_keypoints_on_rectangle_safe(keypoints, angle, box, pad=None, **kwargs):
    if pad is None:
        pad = [0, 0, 0, 0]
    h_min_pad, w_min_pad, h_max_pad, w_max_pad = pad
    keypoints = keypoints - box.mean(axis=0).astype(int)

    w = int(dist(box[1], box[2]))
    h = int(dist(box[0], box[1]))

    h, w = min(w, h), max(w, h)
    if isinstance(h_max_pad, float):
        h_min_pad = max(int(h_min_pad * h) - 1, 0)
        w_min_pad = max(int(w_min_pad * w) - 1, 0)

    M = cv2.getRotationMatrix2D((0, 0), -angle, 1)[:, :2]
    keypoints = (keypoints @ M) + (w // 2 + w_min_pad, h // 2 + h_min_pad)
    return keypoints.astype(int)


def keypoints_random_crop(keypoints, crop_height, crop_width, h_start, w_start, rows, cols):
    """Keypoints random crop.

    Args:
        keypoints: (numpy.ndarray): An array of key points `(x, y)`.
        crop_height (int): Crop height.
        crop_width (int): Crop width.
        h_start (int): Crop height start.
        w_start (int): Crop width start.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    crop_coords = F.get_random_crop_coords(rows, cols, crop_height, crop_width, h_start, w_start)
    x1, y1, _, _ = crop_coords
    keypoints[:, :2] = keypoints[:, :2] - (x1, y1)

    return keypoints


def keypoints_scale(keypoints, scale_x, scale_y):
    """Scales a keypoint by scale_x and scale_y.

    Args:
        keypoints (tuple): A keypoint `(x, y, angle, scale)`.
        scale_x (int): Scale coefficient x-axis.
        scale_y (int): Scale coefficient y-axis.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    keypoints[:, :2] = keypoints[:, :2] * (scale_x, scale_y)

    return keypoints


def keypoints_shift_scale_rotate(keypoints, angle, scale, dx, dy, rows, cols, **params):
    target_keypoints = keypoints[:, :2]
    meta_inf = keypoints[:, 2:]

    height, width = rows, cols
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height

    new_keypoints = cv2.transform(target_keypoints[None], matrix).squeeze()

    return np.hstack([new_keypoints, meta_inf])


def keypoints_vflip(keypoints, rows, cols):
    """Flip a keypoint vertically around the x-axis.

    Args:
        keypoints (tuple): A keypoint `(x, y, angle, scale)`.
        rows (int): Image height.
        cols( int): Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    """
    keypoints[:, 1] = (rows - 1) - keypoints[:, 1]

    return keypoints


def keypoints_hflip(keypoints, rows, cols):
    """Flip a keypoints horizontally around the y-axis.

    Args:
        keypoints (tuple): A keypoint `(x, y, angle, scale)`.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    """
    keypoints[:, 0] = (cols - 1) - keypoints[:, 0]

    return keypoints


def keypoints_flip(keypoints, d, rows, cols):
    """Flip a keypoint either vertically, horizontally or both depending on the value of `d`.

    Args:
        keypoints (tuple): A keypoint `(x, y, angle, scale)`.
        d (int): Number of flip. Must be -1, 0 or 1:
            * 0 - vertical flip,
            * 1 - horizontal flip,
            * -1 - vertical and horizontal flip.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    Raises:
        ValueError: if value of `d` is not -1, 0 or 1.

    """
    if d == 0:
        keypoints = keypoints_vflip(keypoints, rows, cols)
    elif d == 1:
        keypoints = keypoints_hflip(keypoints, rows, cols)
    elif d == -1:
        keypoints = keypoints_hflip(keypoints, rows, cols)
        keypoints = keypoints_vflip(keypoints, rows, cols)
    else:
        raise ValueError("Invalid d value {}. Valid values are -1, 0 and 1".format(d))

    return keypoints
