# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""


Created: 
"""

import logging, time
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cv2

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

def calc_camera_shake_rotation(frames: Union[xr.DataArray, np.ndarray],
                               frame_reference: Union[xr.DataArray, np.ndarray],
                                    method='phase_correlation',
                                    verbose=False):  # pragma: no cover
    """
    https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    Args:
        frames:
        frame_reference:
        method:
        verbose:

    Returns:

    """

    def get_gradient(im):
        # Calculate the x and y gradients using Sobel operator
        grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)

        # Combine the two gradients
        grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
        return grad
    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    (cc, warp_matrix) = cv2.findTransformECC(get_gradient(frames), get_gradient(frame_reference),
                                             warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);


def calc_camera_shake_displacements(frames: Union[xr.DataArray, np.ndarray],
                                    frame_reference: Union[xr.DataArray, np.ndarray],
                                    method='phase_correlation', erroneous_displacement=10,
                                    verbose=False):
    """Return array of pixel (x, y) displacemnts for each frame relative to a reference frame

    Documentation of the opencv phaseCorrelate fuction can be found at:
    https://docs.opencv.org/2.4/modules/imgproc/doc/motion_analysis_and_object_tracking.html

    Args:
        frames          : Frame data with dimensions (t, y_pix, x_pix)
        frame_reference : Reference frame with dims (y_pix, x_pix)
        verbose         : Print displacement information to logger.info

    Returns: Array of pixel displacements with dimensions (t, 2), where displacements (x, y) are those required to
             map the data back to the reference frame

    """
    methods = {'phase_correlation': calc_camera_shake_phase_correlation}
    t0 = time.time()
    frame_reference = np.array(frame_reference)
    if method in methods:
        displacements, camera_shake_stats = methods[method](frames, frame_reference, verbose=verbose)
    else:
        raise NotImplementedError(f'No camera shake detection implemented for method: "{method}". '
                                  f'Options: {list(methods.keys())}')
    t1 = time.time()
    if verbose:
        logger.info(f'Camera shake for {len(frames)} frames calculated in ({t1-t0:0.2f}) s')

    mask_erroneous = np.abs(displacements) > erroneous_displacement
    if (erroneous_displacement is not None) and (np.any(mask_erroneous)):
        frames_erroneous = np.nonzero(np.any(mask_erroneous, axis=1))[0]
        n_erroneous = len(frames_erroneous)
        displacements[mask_erroneous] = np.nan
        camera_shake_stats = calc_camera_shake_stats(displacements)
        logger.warning(f'Erroneous camera shake displacements (>{erroneous_displacement} pixels) set to nan for '
                       f'{n_erroneous}/{len(frames)} frames, i: {frames_erroneous}\n'
                       f'   New camera shake stats: {camera_shake_stats}')

    return displacements, camera_shake_stats

def calc_camera_shake_stats(displacements):
    camera_shake_stats = {}
    displacements_abs = np.abs(displacements)
    camera_shake_stats['disp_abs_av'] = np.nanmean(displacements_abs, axis=0)
    camera_shake_stats['disp_abs_max_overall'] = np.nanmax(displacements_abs, axis=0)
    camera_shake_stats['disp_norm_max'] = np.nanmax(np.linalg.norm(displacements_abs, axis=0))
    return camera_shake_stats

def calc_camera_shake_phase_correlation(frames, reference_frame, verbose=False):
    if reference_frame.ndim == 3:
        reference_frame = cv2.cvtColor(reference_frame, cv2.COLOR_RGB2GRAY)
    # cv2.phaseCorrelate requires 32/64 bit float arrays
    reference_frame = np.float32(reference_frame)
    frames = np.float32(frames)
    displacements = np.full((len(frames), 2), np.nan)
    correlations = np.full(len(frames), np.nan)
    for n in np.arange(len(frames)):
        frame = np.array(frames[n])
        displacements[n], correlations[n] = cv2.phaseCorrelate(frame, reference_frame)

    camera_shake_stats = calc_camera_shake_stats(displacements)
    camera_shake_stats['correl_min'] = correlations.min()
    camera_shake_stats['correl_mean'] = correlations.mean()
    camera_shake_stats['correl_max'] = correlations.max()
    camera_shake_stats['disp_correl_min'] = displacements[correlations.argmin()]
    if verbose:
        logger.debug(f'Camera shake stats: {camera_shake_stats}')
        shape = frames.shape
        disp_max = np.max(camera_shake_stats['disp_abs_max_overall'])
        if disp_max < 1:
            logger.info(f'Camera shake is SUB-PIXEL for all {len(frames)} ({shape[1]}x{shape[1]}) frames')
        elif disp_max > 3:
            logger.warning(f'Camera shake is LARGE (>3 pixels): {camera_shake_stats}')
        elif disp_max > 1:
            logger.info('Camera shake is SIGNIFICANT (1<shake<3 pixels)')
    camera_shake_stats['correlations'] = correlations
    return displacements, camera_shake_stats

def remove_camera_shake(frame_data, pixel_displacements, verbose=False):
    t0 = time.time()
    pixel_displacements_int = np.round(np.array(pixel_displacements))
    pixel_displacements_int[np.isnan(pixel_displacements_int)] = 0  # Apply no roll correction for nans
    pixel_displacements_int = pixel_displacements_int.astype(int)
    n_corrected = 0
    for n in np.arange(len(frame_data)):
        displacement = pixel_displacements_int[n]
        if np.all(displacement == [0, 0]):
            continue
        n_corrected += 1
        frame_data[n] = np.roll(frame_data[n], displacement, axis=(1, 0))

        # Swap wrapped values for nans
        if displacement[1] == 0:
            pass
        elif displacement[1] > 0:
            frame_data[n, :displacement[1], :] = np.nan
        else:
            frame_data[n, displacement[1]:, :] = np.nan

        if displacement[0] == 0:
            pass
        elif displacement[0] > 0:
            frame_data[n, :, :displacement[0]] = np.nan
        else:
            frame_data[n, :, displacement[0]:] = np.nan

        if np.all(np.isnan(frame_data[n])):
            logger.warning(f'Camera shake correction has resulted in a frame {n} being all nans. '
                           f'Correction={displacement}.')
    t1 = time.time()
    if verbose:
        frac = n_corrected/len(frame_data)
        logger.info(f'Camera shake displacement corrections applied to {n_corrected}/{len(frame_data)} '
                    f'{frac:0.3%} frames in {t1-t0:0.2f} s '
                    f'(max correction = {float(np.max(pixel_displacements)):0.3g} pixels). '
                    f'NOTE: This will have introduced nans to corrected frames.')


    return frame_data

def remove_camera_shake_calcam(frames, calcam_calib):
    import calcam
    calibs = []
    for n in np.arange(len(frames)):
        moved_image = np.array(frames[n])
        mov = calcam.movement.detect_movement(calcam_calib, moved_image)
        corrected_image, mask = mov.warp_moved_to_ref(moved_image)
        updated_calib = calcam.movement.update_calibration(calcam_calib, moved_image, mov)
        frames[n] = corrected_image
        calibs.append(updated_calib)
    return frames, calibs

if __name__ == '__main__':
    pass