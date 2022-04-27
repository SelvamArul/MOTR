# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------

cimport cython
from libc.math cimport acos, pi
import numpy as np
cimport numpy as np


def bbox_overlaps(
        np.ndarray[float, ndim=2] boxes,
        np.ndarray[float, ndim=2] query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float32 (x1, y1, x2, y2)
    query_boxes: (M, 4) ndarray of float32 (x1, y1, x2, y2)
    Returns
    -------
    overlaps: (N, M) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int M = query_boxes.shape[0]
    cdef np.ndarray[float, ndim=2] overlaps = np.zeros((N, M), dtype=np.float32)
    cdef float iw, ih, box_area
    cdef float ua
    cdef unsigned int m, n
    for m in range(M):
        box_area = (
            (query_boxes[m, 2] - query_boxes[m, 0] + 1) *
            (query_boxes[m, 3] - query_boxes[m, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[m, 2]) -
                max(boxes[n, 0], query_boxes[m, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[m, 3]) -
                    max(boxes[n, 1], query_boxes[m, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, m] = iw * ih / ua
    return overlaps


def rot_errors(
        np.ndarray[float, ndim=3] rots,
        np.ndarray[float, ndim=3] query_rots):
    """
    Parameters
    ----------
    rots: (N, 3, 3) ndarray of float32
    query_rots: (M, 3, 3) ndarray of float32
    Returns
    -------
    errors: (N, M) ndarray of errors between rots and query_rots
    """
    cdef unsigned int N = rots.shape[0]
    cdef unsigned int M = query_rots.shape[0]
    cdef np.ndarray[float, ndim=2] errors = np.zeros((N, M), dtype=np.float32)
    cdef float error_cos
    cdef unsigned int m, n
    for m in range(M):
        for n in range(N):
            error_cos = 0.5 * (np.trace(query_rots[m].dot(rots[n].T)) - 1.0)
            error_cos = min(1.0, max(-1.0, error_cos))  # avoid invalid values due to numerical errors
            errors[n, m] = 180.0 * acos(error_cos) / pi  # [rad] -> [deg]
    return errors
