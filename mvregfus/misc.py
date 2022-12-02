import numpy as np
from tqdm import tqdm
import skimage
from scipy import ndimage


def correct_random_drift(ims, reg_ch=0, zoom_factor=10, particle_reinstantiation_stepsize=30, sigma_t=3):
    """
    ## Stage shift correction (currently 2d, but easily extendable)

    Goal: Correct for random stage shifts in timelapse movies in the absence of reference points that are fixed relative to the stage.

    Method: Assume that in the absence of random shifts particle trajectories are smooth in time. Find raw (uncorrected) trajectories, smooth them over time and determine the random stage shifts as the mean deviation of the actual trajectories from the smoothed ones.

    Steps:
    1) Calculate PIV fields (or optical flow) for the timelapse movie
    2) Track the initial coordinates as virtual particles throughout the timelapse
    3) Smooth trajectories and determine the mean deviations (N_t, d_x, d_y) from the actual trajectories
    4) Use the obtained deviations to correct the timelapse and verify result quality visually

    Comments:
    - assumes input data to be in the format (t, c, y, x)
    """

    import dask.delayed as d
    import dask
    from dask.diagnostics import ProgressBar

    # for some reason float images give a different output
    ims = ims.astype(np.uint16)

    of_channel = reg_ch
    regims = np.array([ndimage.zoom(ims[it, of_channel], 1./zoom_factor, order=1) for it in range(len(ims[:]))])

    fs = [d(skimage.registration.optical_flow_tvl1)(regims[t],
                                                    regims[t+1],
                                                attachment=10000,
                                                )
                for t in tqdm(range(len(ims) - 1)[:])]

    print('Computing optical flow...')
    with ProgressBar():
        fs=np.array(dask.compute(fs)[0])#*zoom_factor


    print('Tracking virtual particles...')

    # coordinates to be tracked
    x,y = np.mgrid[0:regims[0].shape[0], 0:regims[0].shape[1]]

    # define starting point(s) for coordinates to be tracked
    # for long movies, it's good to have several starting tps
    # as the coordinates can move out of the FOV
    
    starting_tps = range(0, len(ims), particle_reinstantiation_stepsize)

    posss = []
    for starting_tp in starting_tps:
        poss = [np.array([x, y])]
        for t in tqdm(range(starting_tp, len(ims) - 1)):
            displacement = np.array([ndimage.map_coordinates(
                                fs[t][dim],
                                np.array([poss[-1][0].flatten(), poss[-1][1].flatten()]),
                                order=1, mode='constant', cval=np.nan)
                            .reshape(x.shape) for dim in range(2)])
            poss.append(displacement + poss[-1])
        posss.append(np.array(poss))


    print('Smoothing trajectories...')
    posss_smooth = [ndimage.gaussian_filter(posss[istp], [sigma_t, 0, 0, 0], mode='nearest')
                        for istp, starting_tp in enumerate(starting_tps)]

    devs = np.array([np.nanmean([posss_smooth[istp][t-starting_tp] - posss[istp][t-starting_tp]
                        for istp, starting_tp in enumerate(starting_tps) if starting_tp <= t], axis=(0, -1, -2))
                    for t in range(len(ims))])

    devs = devs * zoom_factor

    print('Correct drifts...')
    imst = np.array([[ndimage.affine_transform(ims[t,ch],
                                            matrix=[[1,0],[0,1]],
                                            offset=-devs[t], order=1)
                    for ch in range(ims.shape[1])]
                    for t in tqdm(range(len(ims)))])

    return imst, devs