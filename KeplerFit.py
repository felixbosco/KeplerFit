# KeplerFit
#
# Author: F. Bosco (Max Planck Institute for Astronomy, Heidelberg)
# Last major update: 09/01/2019
# Last edited: 30/04/2020
# Description: A small piece of code to fit a Keplerian velocity distribution model to position-velocity data
#

from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
import warnings

from astropy import constants as const
from astropy.io import fits, ascii
from astropy import units as u
from astropy.utils.exceptions import AstropyUserWarning
from astropy.table import Table, QTable
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter


# Definition of the principal class
class PVdata(object):

    def __init__(self, filename, noise=None, position_reference=None, debug=False):

        """Create a PVdata instance.

        Args:
            filename (str):
                Name of the FITS file to initialize from.
            noise (float/ astropy.units.Quantity, optional):
                Noise of the data, required for the threshold of the extreme velocities. By default the standard
                deviation of the data is used.
            position_reference (int, optional):
                Position of the central massive object. By default, the central position is used.
        """

        # Read FITS file
        data, hdr = fits.getdata(filename=filename, header=True)

        # Remove empty dimensions and mask invalid entries
        if debug:
            print('Input data shape:', data.shape)
        data = np.squeeze(data)
        if np.isnan(data).any():
            print('Masking invalid data...')
            data = np.ma.masked_invalid(data)
        if debug:
            print('Data shape:', data.shape)

        self.data = data
        self.data_unit = u.Unit(hdr['BUNIT'])
        if noise is None:
            self.noise = np.std(self.data)
        else:
            if isinstance(noise, u.Quantity):
                self.noise = float(noise / self.data_unit)
            else:
                self.noise = noise

        # Read coordinate system parameters from FITS header
        self.position_resolution = u.Quantity(f"{hdr['CDELT1']} {hdr['CUNIT1']}")
        if self.position_resolution.value < 0.0:
            print('Shifting the position increment to positive value...')
            self.position_resolution = np.abs(self.position_resolution.value) * self.position_resolution.unit
        if position_reference is None:
            self.position_reference = int(hdr['NAXIS1'] / 2)
        else:
            self.position_reference = position_reference

        self.vLSR = u.Quantity(f"{hdr['CRVAL2']} {hdr['CUNIT2']}")
        self.velocity_resolution = u.Quantity(f"{hdr['CDELT2']} {hdr['CUNIT2']}")
        self.vLSR_channel = int(hdr['CRPIX2'] - 1)  # convert from 1-based to 0-based counting

        if debug:
            self.vLSR_channel = 16

    # Seifried et al. (2016) algorithm
    def start_low(self, indices=None, weak_quadrants=False):
        """Compare the flux in the four quadrants.

        Args:
            indices (dict, optional):
                Index dictionary with entries for min, max and central pixel. If none, than fall back values are
                applied.
            weak_quadrants (bool, optional):
                Invert return value.

        Returns:
            start_low(bool):
                True, if the flux in the pair of quadrants that contains low velocities at low positions is larger than
                the flux of the other two quadrants.
        """

        # Apply fall back values
        if indices is None:
            indices = {'min': 0, 'max': -1, 'central': self.vLSR_channel}

        # Estimate total flux of quadrants
        q1 = np.sum(self.data[self.position_reference:, indices['central']:indices['max']])  # +x, +v
        q2 = np.sum(self.data[:self.position_reference, indices['central']:indices['max']])  # -x, +v
        q3 = np.sum(self.data[:self.position_reference, indices['min']:indices['central']])  # -x, -v
        q4 = np.sum(self.data[self.position_reference:, indices['min']:indices['central']])  # +x, -v

        if not weak_quadrants:
            return q1 + q3 > q2 + q4
        else:
            return q1 + q3 < q2 + q4

    def estimate_extreme_channels(self, threshold, plot=False, weak_quadrants=False, channel_interval=None):
        # initialize
        indices = {'min': 0, 'max': -1, 'central': self.vLSR_channel}
        if channel_interval is not None:
            if isinstance(channel_interval, tuple):
                indices = {'min': channel_interval[0], 'max': channel_interval[1], 'central': int((channel_interval[1] + channel_interval[0]) / 2)}
            else:
                raise TypeError('The function estimate_extreme_channels() can only handle a single channel interval at a time but got {}!'.format(channel_interval))
                print('>> Restriction for channels set to {}.'.format(indices))
        self.channels = np.ma.masked_array(np.zeros(self.data.shape[1]), mask=np.zeros(self.data.shape[1], dtype=bool))
        print('Indices are {}.'.format(indices))

        # iteration over position coordinate
        for i, pos in enumerate(self.data.transpose()):
            # initialize channel iteration
            if self.start_low(indices=indices, weak_quadrants=weak_quadrants):
                if i < self.position_reference:
                    j = indices['min']
                    dj = 1
                else:
                    j = indices['max']
                    dj = -1
            else:
                if i < self.position_reference:
                    j = indices['max']
                    dj = -1
                else:
                    j = indices['min']
                    dj = 1
            # iteration over channels for the current position i
            while pos[j] < threshold * self.noise and j % self.data.shape[0] != indices['central']:
                j += dj
            if j == self.vLSR_channel:
                self.channels.mask[i] = True
            else:
                self.channels[i] = j % self.data.shape[0]

        # for safety repeat flagging if channels equal to v_LSR
        for position, channel in enumerate(self.channels):
            if channel == indices['min'] or channel == indices['max'] or channel == indices['central']:
                self.channels.mask[position] = True

        # plot
        if plot:
            plt.plot(self.channels, 'o')
            plt.xlim(-1, self.data.shape[1]+1)
            plt.ylim(-1, self.data.shape[0]+1)
            plt.grid()
            plt.show()
            plt.close()

        # return
        return self.channels

    def _angle_to_length(self, angle, distance):
        return (angle.to(u.arcsec)).value * (distance.to(u.pc)).value * u.AU

    def _velocity_to_channel(self, velocity_tuple):
        channel_tuple = []
        for velocity in velocity_tuple:
            channel = int((velocity - self.vLSR) / self.velocity_resolution) + self.vLSR_channel
            channel_tuple.append(channel)
        channel_tuple = (np.min(channel_tuple), np.max(channel_tuple))
        print('>> Transfering the velocity interval {} into the channel interval {}, using the PV attributes:'.format(velocity_tuple, channel_tuple))
        print({'vLSR': self.vLSR, 'v resolution': self.velocity_resolution, 'vLSR channel': self.vLSR_channel})
        return channel_tuple

    def estimate_extreme_velocities(self, threshold, source_distance, plot=False, weak_quadrants=False, **kwargs):
        # initialize the data table
        self.table = QTable(names=('Position', 'Channel', 'Angular distance', 'Distance', 'Velocity'),
                           dtype=(int, int, u.Quantity, u.Quantity, u.Quantity))

        # estimate the extreme channels
        if 'velocity_interval' in kwargs:
            #print('The handling of velocity intervals is not supported yet. Try channel_intervals...')
            velocity_interval = kwargs['velocity_interval']
            if isinstance(velocity_interval, tuple):
                channel_interval = self._velocity_to_channel(velocity_interval)
            else:
                raise TypeError('The function estimate_extreme_velocities() can only handle a single velocity interval at a time but got {}!'.format(velocity_interval))
            self.estimate_extreme_channels(threshold, plot=False, weak_quadrants=weak_quadrants, channel_interval=channel_interval)
        elif 'channel_interval' in kwargs:
            self.estimate_extreme_channels(threshold, plot=False, weak_quadrants=weak_quadrants, channel_interval=kwargs['channel_interval'])
        else:
            self.estimate_extreme_channels(threshold, plot=False, weak_quadrants=weak_quadrants)

        # transfer the channels into physical units
        for position, channel in enumerate(self.channels):
            angular_distance = (position - self.position_reference) * self.position_resolution
            distance = self._angle_to_length(angular_distance, source_distance)
            velocity = (channel - self.vLSR_channel) * self.velocity_resolution + self.vLSR
            try:
                self.table.add_row([position, channel, angular_distance.value, distance.value, velocity.value])
            except AttributeError:
                #print([position, channel, angular_distance, distance, velocity])
                pass
        self.table['Angular distance'] = self.table['Angular distance']  * self.position_resolution.unit
        self.table['Distance'] = self.table['Distance'] * u.AU
        self.table['Velocity'] = self.table['Velocity'] * self.velocity_resolution.unit

        # plot
        if plot:
            plt.plot(self.table['Distance'], self.table['Velocity'], 'o', label='data')
            plt.xlabel('Position offest ({})'.format(self.table['Distance'].unit))
            plt.xlabel('Velocity ({})'.format(self.table['Velocity'].unit))
            plt.axhline(self.vLSR.value, c='k', ls='--', label='$v_\mathrm{LSR}$')
            plt.grid()
            plt.legend()
            plt.show()
            plt.close()
        return self.table

    def write_table(self, filename):
        """Write data to an ascii file.

        Args:
            filename (str):
                Name of the file to write to.
        """
        self.table.write(filename, format='ascii.fixed_width', overwrite=True)


# Models
@custom_model
def Keplerian1D(x, mass=1., v0=0., r0=0.):
    v = np.sign(x - r0) * np.sqrt(const.G * mass * const.M_sun / np.abs(x - r0) / u.AU).to(u.km/u.s).value + v0
    return v


@custom_model
def Keplerian1D_neg(x, mass=1., v0=0., r0=0.):
    v = -1*np.sign(x - r0) * np.sqrt(const.G * mass * const.M_sun / np.abs(x - r0) / u.AU).to(u.km/u.s).value + v0
    return v


# Main function
def model_Keplerian(self, threshold, source_distance,
                    return_stddevs=True, print_results=False, plot=False,
                    flag_singularity=True, weak_quadrants=False, fit_method=LevMarLSQFitter(),
                    velocity_interval=None, channel_interval=None,
                    flag_radius=None, flag_intervals=None,
                    debug=False,
                    **kwargs):

    """Model a keplerian profile to PVdata.

    Args:
        threshold (int/ float):
            Set as multiples of PVdata.noise (for instance 3sigma)
        source_distance (astropy.units.quantity):
            .
        return_stddevs (boolean, optional):
            The fit method LevMarLSQFitter is able to return the standard deviation of the fit parameters. Default is
            True.
        print_results (boolean, optional):
            If True, the fit parameters will be displayed to the terminal.
        plot (boolean, optional):
        If True, the fit will be displayed as a matplotlib pyplot.

    #Optional keyword arguments:
        flag_radius (astropy.units.Quantity, optional):
            If given, then all data points within this given radius from the position_reference are flagged.
        flag_intervals (list of tupels of astropy.units.Quantity, optional):
            Similar to flag_radius, but arbitrary intervals may be flagged. Each interval is
            given as a tuple of two radial distances from the position_reference.

    Returns:
        best_fit (astropy.modelling.models.custom_model):
        stddevs (numpy.array):
            Only if return_stddevs is True. The array entries correspond to the best_fit instance parameters in the
            same order.
        chi2 (float):
            chi-squared residual of the fit to the unflagged data.
    """

    # compute the velocity table
    # if 'velocity_interval' in kwargs:
    #     velocity_interval = kwargs['velocity_interval']
    if velocity_interval is not None:
        channel_interval = self._velocity_to_channel(velocity_interval)
        indices = {'min': channel_interval[0], 'max': channel_interval[1], 'central': int((channel_interval[1] + channel_interval[0]) / 2)}
        self.estimate_extreme_velocities(threshold=threshold, source_distance=source_distance,
                                         plot=False, weak_quadrants=weak_quadrants,
                                         velocity_interval=velocity_interval)
    # elif 'channel_interval' in kwargs:
    #     channel_interval = kwargs['channel_interval']
    elif channel_interval is not None:
        indices = {'min': channel_interval[0], 'max': channel_interval[1], 'central': int((channel_interval[1] + channel_interval[0]) / 2)}
        self.estimate_extreme_velocities(threshold=threshold, source_distance=source_distance,
                                         plot=False, weak_quadrants=weak_quadrants,
                                         channel_interval=channel_interval)
    else:
        indices = None
        self.estimate_extreme_velocities(threshold=threshold, source_distance=source_distance,
                                         plot=False, weak_quadrants=weak_quadrants)

    # flag
    xdata = np.ma.masked_array(self.table['Distance'].value, np.zeros(self.table['Distance'].shape, dtype=bool))
    ydata = np.ma.masked_array(self.table['Velocity'].value, np.zeros(self.table['Velocity'].shape, dtype=bool))
    if flag_singularity:
        print('Flagging the singularity:')
        i = np.where(np.abs(self.table['Distance'].value) < 1e-6)[0]
        xdata.mask[i] = True
        ydata.mask[i] = True
        print('>> Flagged the elements {}.'.format(i))
    # if 'flag_radius' in kwargs:
    #     flag_radius = kwargs['flag_radius']
    if flag_radius is not None:
        print('Flagging towards a radial distance of {}:'.format(flag_radius))
        flag_radius = flag_radius.to(u.AU).value
        i = np.where(np.abs(self.table['Distance'].value) < flag_radius)[0]
        xdata.mask[i] = True
        ydata.mask[i] = True
        print('>> Flagged the elements {}.'.format(i))
    # if 'flag_intervals' in kwargs:
    #     flag_intervals = kwargs['flag_intervals']
    if flag_intervals is not None:
        print('Flagging intervals:')
        flagged = np.empty(shape=(0,), dtype=int)
        for interval in flag_intervals:
            i1 = np.where(interval[0].value < self.table['Distance'].value)[0]
            i2 = np.where(self.table['Distance'].value < interval[1].value)[0]
            i = np.intersect1d(i1, i2)
            xdata.mask[i] = True
            ydata.mask[i] = True
            flagged = np.append(flagged, i)
        if len(np.unique(flagged)) < 10:
            print('>> Flagged the elements {}.'.format(np.unique(flagged)))
        else:
            print('>> Flagged {} elements.'.format(len(np.unique(flagged))))

    # choose and initialize the fit model
    if self.start_low(indices=indices, weak_quadrants=weak_quadrants):
        init = Keplerian1D(mass=10., v0=self.vLSR.value, r0=0, bounds={'mass': (0.0, None)})
    else:
        init = Keplerian1D_neg(mass=10., v0=self.vLSR.value, r0=0, bounds={'mass': (0.0, None)})

    # model
    # with warnings.catch_warnings():
    #     # Catching RuntimeWarnings turning them to errors
    #     warnings.simplefilter('error')
    #     try:
    #         best_fit = fit_method(init, xdata.compressed(), ydata.compressed())
    #     except AstropyUserWarning as e:
    #         print(e)
    #         print("fit_info['message']:")
    #         print(fit_method.fit_info['message'])
    best_fit = fit_method(init, xdata.compressed(), ydata.compressed())

    # estimate chi2
    fdata = best_fit(xdata)
    residuals = fdata - ydata
    chi2 = np.sum(np.square(residuals))

    # plot
    if plot:
        plt.plot(self.table['Distance'], self.table['Velocity'], 'o', label='data')
        plt.xlabel('Position offest ({})'.format(self.table['Distance'].unit))
        plt.ylabel('Velocity ({})'.format(self.table['Velocity'].unit))
        plt.axhline(self.vLSR.value, c='k', ls='--', label='$v_\mathrm{LSR}$')
        plt.plot(xdata, best_fit(xdata), label='model')
        plt.fill_between(xdata, best_fit(xdata), best_fit.v0, facecolor='tab:orange', alpha=.5)
        if debug:
            plt.plot(xdata, init(xdata), label='init')
        plt.grid()
        plt.legend()
        plt.show()
        plt.close()

    # return
    if not isinstance(fit_method, LevMarLSQFitter):
        return_stddevs = False
    if return_stddevs:
        covariance = fit_method.fit_info['param_cov']
        try:
            stddevs = np.sqrt(np.diag(covariance))
            if print_results:
                pass
            return best_fit, stddevs, chi2
        except ValueError as e:
            if covariance is None:
                print('Catched the following error, which is due to an unsucessful fit (covariance matrix is {}):'.format(covariance))
                print('ValueError: {}'.format(e))
            return best_fit, chi2
    if debug:
        print(fit_method.fit_info['message'])
    if print_results:
        pass
    return best_fit, chi2
