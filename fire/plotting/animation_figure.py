# !/usr/bin/env python
"""
example writing matplotlib movies with matlab "grabframe"-like syntax
Just four lines of code to directly write lossless AVIs from Matplotlib
http://matplotlib.org/api/animation_api.html#matplotlib.animation.FFMpegWriter
"bitrate" does not have an effect on lossless video, only lossy video (higher bitrate=higher quality=bigger files)
codecs:
ffv1: lossless
mpeg4: lossy
"""
import numpy as np
from numpy.random import uniform
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, draw, pause, close  # noqa: E402
import matplotlib.animation as anim  # noqa: E402
from matplotlib import animation

matplotlib.use('agg')  # invisible while plotting, but faster and more robust

BITRATE = 500e3
DPI = 100
X = Y = 128
N = 50
FPS = 15  # keep greater than 3 to avoid VLC playback bug
WRITER = 'ffmpeg_file'

EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'

def testdata(x, y, N):
    # TODO: better example data to show downsides of lossy video
    return uniform(size=(N, y, x))


def setupfig(img, title=''):
    # %% boilerplate for making imshow priming (used in any program)
    assert img.ndim == 2, '2-D image expected'
    fg = figure()
    ax = fg.gca()
    h = ax.imshow(img)  # prime figure with first frame of data
    ax.set_title(title, color='g')
    return fg, h


def loop(fg, h, w, fn, imgs):
    assert imgs.ndim in (3, 4), 'assuming image stack iterating over first dimension Nframe x X x Y [x RGB]'

    with w.saving(fg, fn, DPI):
        print('writing', fn)
        for I in imgs:
            h.set_data(I)
            draw()
            # NOTE: pause is NEEDED for more complicated plots, to avoid random crashes
            pause(0.01)
            w.grab_frame(facecolor='k')

    close(fg)


def config(h, codec):
    Writer = anim.writers[WRITER]
    return Writer(fps=FPS, codec=codec, bitrate=BITRATE)

def save_avi_animation(data, path_fn='movie.mp4'):
    ani = movie_from_data(np.array(data), xlabel='Horizontal coord [pixels]', ylabel='Vertical coord [pixels]',
                          barlabel='Temperature [W/m2]')
    ani.save(path_fn, fps=5, writer='ffmpeg', codec='mpeg4')
    plt.close('all')

def movie_from_data(data, framerate,integration=1,xlabel=(),ylabel=(),barlabel=(),cmap='rainbow',timesteps='auto',
                    extvmin='auto',extvmax='auto',mask=[0],mask_alpha=0.2,time_offset=0,prelude='',vline=None,
                    hline=None,EFIT_path=EFIT_path_default,include_EFIT=False,efit_reconstruction=None,
                    EFIT_output_requested = False,pulse_ID=None,overlay_x_point=False,overlay_mag_axis=False,
                    overlay_structure=False,overlay_strike_points=False,overlay_separatrix=False,structure_alpha=0.5):
    """movie_from_data"""

    form_factor = (np.shape(data[0][0])[1])/(np.shape(data[0][0])[0])
    fig = plt.figure(figsize=(12*form_factor, 12))
    ax = fig.add_subplot(111)

    # I like to position my colorbars this way, but you don't have to
    # div = make_axes_locatable(ax)
    # cax = div.append_axes('right', '5%', '5%')

    # def f(x, y):
    #	 return np.exp(x) + np.sin(y)

    # x = np.linspace(0, 1, 120)
    # y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

    # This is now a list of arrays rather than a list of artists
    frames = [None]*len(data[0])
    frames[0] = data[0, 0]

    for i in range(len(data[0])):
        # x	   += 1
        # curVals  = f(x, y)
        frames[i] = (data[0,i])

    cv0 = frames[0]
    im = ax.imshow(cv0,cmap, origin='lower', interpolation='none')  # Here make an AxesImage rather than contour

    if len(np.shape(mask)) == 2:
        if np.shape(mask) != np.shape(data[0][0]):
            print('The shape of the mask '+str(np.shape(mask))+' does not match with the shape of the record '+str(np.shape(data[0][0])))
        masked = np.ma.masked_where(mask == 0, mask)
        im2 = ax.imshow(masked, 'gray', interpolation='none', alpha=mask_alpha,origin='lower',extent = [0,np.shape(cv0)[1]-1,0,np.shape(cv0)[0]-1])

    cb = fig.colorbar(im).set_label(barlabel)
    cb = ax.set_xlabel(xlabel)
    cb = ax.set_ylabel(ylabel)
    tx = ax.set_title('Frame 0')

    def animate(i):
        arr = frames[i]
        if extvmax=='auto':
            vmax = np.max(arr)
        elif extvmax=='allmax':
            vmax = np.max(data)
        else:
            vmax = extvmax

        if extvmin=='auto':
            vmin = np.min(arr)
        elif extvmin=='allmin':
            vmin = np.min(data)
        else:
            vmin = extvmin
        im.set_data(arr)
        im.set_clim(vmin, vmax)
        if timesteps=='auto':
            time_int = time_offset+i/framerate
            tx.set_text(prelude + 'Frame {0}'.format(i)+', FR %.3gHz, t %.3gs, int %.3gms' %(framerate,time_int,integration))
        else:
            time_int = timesteps[i]
            tx.set_text(prelude + 'Frame {0}'.format(i)+', t %.3gs, int %.3gms' %(time_int,integration))

    ani = animation.FuncAnimation(fig, animate, frames=len(data[0]))

    if EFIT_output_requested == False:
        return ani
    else:
        return ani,efit_reconstruction

def setup_overplot_matnetics(ax):
    if include_EFIT:
        try:
            if efit_reconstruction == None:
                print('reading '+EFIT_path+'/epm0'+str(pulse_ID)+'.nc')
                efit_reconstruction = mclass(EFIT_path+'/epm0'+str(pulse_ID)+'.nc',pulse_ID=pulse_ID)
            else:
                print('EFIT reconstruction externally supplied')
            EFIT_dt = np.median(np.diff(efit_reconstruction.time))
        except Exception as e:
            print('reading '+EFIT_path+'/epm0'+str(pulse_ID)+'.nc failed')
            logging.exception('with error: ' + str(e))
            include_EFIT=False
            overlay_x_point=False
            overlay_mag_axis=False
            overlay_separatrix=False
            overlay_strike_points=False
            overlay_separatrix=False
            efit_reconstruction = None
        if overlay_x_point:
            all_time_x_point_location = return_all_time_x_point_location(efit_reconstruction)
            plot1 = ax.plot(0,0,'-r', alpha=1)[0]
        if overlay_mag_axis:
            all_time_mag_axis_location = return_all_time_mag_axis_location(efit_reconstruction)
            plot2 = ax.plot(0,0,'--r', alpha=1)[0]
        if overlay_separatrix or overlay_strike_points:
            all_time_sep_r,all_time_sep_z,r_fine,z_fine = efit_reconstruction_to_separatrix_on_foil(efit_reconstruction)
        if overlay_strike_points:
            all_time_strike_points_location,all_time_strike_points_location_rot = return_all_time_strike_points_location(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)
            plot3 = ax.plot(0,0,'xr',markersize=20, alpha=1)[0]
            plot4 = []
            for __i in range(len(all_time_strike_points_location[0])):
                plot4.append(ax.plot(0,0,'-r', alpha=1)[0])
        if overlay_separatrix:
            all_time_separatrix = return_all_time_separatrix(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine)
            plot5 = []
            for __i in range(len(all_time_separatrix[0])):
                plot5.append(ax.plot(0,0,'--b', alpha=1)[0])
    if overlay_structure:
        for i in range(len(fueling_point_location_on_foil)):
            ax.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'+k',markersize=40,alpha=structure_alpha)
            ax.plot(np.array(fueling_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(fueling_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'ok',markersize=5,alpha=structure_alpha)
        for i in range(len(structure_point_location_on_foil)):
            ax.plot(np.array(structure_point_location_on_foil[i][:,0])*(np.shape(cv0)[1]-1)/foil_size[0],np.array(structure_point_location_on_foil[i][:,1])*(np.shape(cv0)[0]-1)/foil_size[1],'--k',alpha=structure_alpha)

    # if len(np.shape(mask)) == 2:
    # im = ax.imshow(mask,'gray',interpolation='none',alpha=1)
    if np.sum(vline == None)==0:
        if np.shape(vline)==():
            vline = max(0,min(vline,np.shape(cv0)[1]-1))
            axvline = ax.axvline(x=vline,linestyle='--',color='k')
        else:
            for i in range(len(vline)):
                vline[i] = max(0,min(vline[i],np.shape(cv0)[1]-1))
                axvline = ax.axvline(x=vline[i],linestyle='--',color='k')
    if np.sum(hline == None)==0:
        if np.shape(hline)==():
            hline = max(0,min(hline,np.shape(cv0)[0]-1))
            axhline = ax.axhline(y=hline,linestyle='--',color='k')
        else:
            for i in range(len(hline)):
                hline[i] = max(0,min(hline[i],np.shape(cv0)[0]-1))
                axhline = ax.axhline(y=hline[i],linestyle='--',color='k')

    cb = fig.colorbar(im).set_label(barlabel)
    cb = ax.set_xlabel(xlabel)
    cb = ax.set_ylabel(ylabel)
    tx = ax.set_title('Frame 0')

def update_magnetics_overplot(ax):
    if include_EFIT:
        if np.min(np.abs(
                time_int - efit_reconstruction.time)) > EFIT_dt:  # means that the reconstruction is not available for that time
            if overlay_x_point:
                plot1.set_data(([], []))
            if overlay_mag_axis:
                plot2.set_data(([], []))
            if overlay_strike_points:
                plot3.set_data(([], []))
                for __i in range(len(plot4)):
                    plot4[__i].set_data(([], []))
            if overlay_separatrix:
                for __i in range(len(plot5)):
                    plot5[__i].set_data(([], []))
        else:
            i_time = np.abs(time_int - efit_reconstruction.time).argmin()
            if overlay_x_point:
                if np.sum(np.isnan(all_time_x_point_location[i_time])) >= len(
                        all_time_x_point_location[i_time]):  # means that all the points calculated are outside the foil
                    plot1.set_data(([], []))
                else:
                    plot1.set_data((all_time_x_point_location[i_time][:, 0] * (np.shape(cv0)[1] - 1) / foil_size[0],
                                    all_time_x_point_location[i_time][:, 1] * (np.shape(cv0)[0] - 1) / foil_size[1]))
            if overlay_mag_axis:
                # if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
                # 	plot2.set_data(([],[]))
                # else:
                plot2.set_data((all_time_mag_axis_location[i_time][:, 0] * (np.shape(cv0)[1] - 1) / foil_size[0],
                                all_time_mag_axis_location[i_time][:, 1] * (np.shape(cv0)[0] - 1) / foil_size[1]))
            if overlay_strike_points:
                # if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
                # 	plot3.set_data(([],[]))
                # 	for __i in range(len(plot4)):
                # 		plot4[__i].set_data(([],[]))
                # else:
                plot3.set_data((all_time_strike_points_location[i_time][:, 0] * (np.shape(cv0)[1] - 1) / foil_size[0],
                                all_time_strike_points_location[i_time][:, 1] * (np.shape(cv0)[0] - 1) / foil_size[1]))
                for __i in range(len(plot4)):
                    plot4[__i].set_data((all_time_strike_points_location_rot[i_time][__i][:, 0] * (
                                np.shape(cv0)[1] - 1) / foil_size[0],
                                         all_time_strike_points_location_rot[i_time][__i][:, 1] * (
                                                     np.shape(cv0)[0] - 1) / foil_size[1]))
            if overlay_separatrix:
                # if np.sum(np.isnan(all_time_mag_axis_location[i_time]))>=len(all_time_mag_axis_location[i_time]):	# means that all the points calculated are outside the foil
                for __i in range(len(plot5)):
                    plot5[__i].set_data((all_time_separatrix[i_time][__i][:, 0] * (np.shape(cv0)[1] - 1) / foil_size[0],
                                         all_time_separatrix[i_time][__i][:, 1] * (np.shape(cv0)[0] - 1) / foil_size[
                                             1]))
        # In this version you don't have to do anything to the colorbar,
        # it updates itself when the mappable it watches (im) changes


if __name__ == '__main__':
    imgs = testdata(X, Y, N)
    # %% lossless
    fg, h = setupfig(imgs[0], 'lossless')
    w = config(h, 'ffv1')
    loop(fg, h, w, 'lossless.avi', imgs)
    # %% lossy
    fg, h = setupfig(imgs[0], 'lossy')
    w = config(h, 'mpeg4')
    loop(fg, h, w, 'lossy.avi', imgs)