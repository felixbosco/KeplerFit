### KeplerFit
# Author: F. Bosco (Max Planck Institute for Astronomy, Heidelberg)
# Last edited: 12/12/2018
# Description: A small piece of code to fit a Keplerian velocity distribution model to position-velocity data


# import
import numpy as np
import matplotlib.pyplot as plt
import warnings
from astropy import units as u
from astropy import constants as const
from astropy.io import fits, ascii
from astropy.table import Table, QTable
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter


# central class definition
class PVdata(object):

    def __init__(self, filename, noise, position_reference):
        with fits.open(filename) as hdulist:
            hdr = hdulist[0].header
            data = hdulist[0].data[0, 0]
            if np.isnan(data).any():
                print('Masking invalid data...')
                data = np.ma.masked_invalid(data)
        self.data = data
        self.data_unit = u.Unit(hdr['BUNIT'])
        self.noise = float(noise / self.data_unit)
        self.position_resolution = (hdr['CDELT1'] * u.deg).to(u.arcsec)
        if self.position_resolution.value < 0.0:
            print('Shifting the position increment to positive value...')
            self.position_resolution = np.abs(self.position_resolution.value) * self.position_resolution.unit
        self.position_reference = position_reference
        self.vLSR = (hdr['CRVAL2'] * u.m).to(u.km) / u.s
        self.velocity_resolution = (hdr['CDELT2'] * u.m).to(u.km) / u.s
        self.vLSR_channel = int(hdr['CRPIX2'] - 1) # convert from 1-based to 0-based counting

    # Seifried et al. (2016) algorithm
    def start_low(self, weak_quadrants=False):
        low = np.sum(self.data[:self.vLSR_channel, :self.position_reference]) + \
                np.sum(self.data[self.vLSR_channel:, self.position_reference:])
        high = np.sum(self.data[:self.vLSR_channel, self.position_reference:]) + \
                np.sum(self.data[self.vLSR_channel:, :self.position_reference])
        if not weak_quadrants:
            return low > high
        else:
            return low < high

    def estimate_extreme_channels(self, threshold, plot=False, weak_quadrants=False):
        self.channels = np.ma.masked_array(np.zeros(self.data.shape[1]), mask=np.zeros(self.data.shape[1], dtype=bool))
        for i, pos in enumerate(self.data.transpose()):
            if self.start_low(weak_quadrants=weak_quadrants):
                if i < self.position_reference:
                    j = 0
                    dj = 1
                else:
                    j = -1
                    dj = -1
            else:
                if i < self.position_reference:
                    j = -1
                    dj = -1
                else:
                    j = 0
                    dj = 1
            while pos[j] < threshold * self.noise and j % self.data.shape[0] != self.vLSR_channel:
                j += dj
            if j == self.vLSR_channel:
                self.channels.mask[i] = True
            else:
                self.channels[i] = j % self.data.shape[0]

        for position, channel in enumerate(self.channels):
            if channel == self.vLSR_channel:
                self.channels.mask[position] = True

        if plot:
            plt.plot(self.channels, 'o')
            plt.xlim(-1, self.data.shape[1]+1)
            plt.ylim(-1, self.data.shape[0]+1)
            plt.grid()
            plt.show()
            plt.close()
        return self.channels

    def __angle_to_length(self, angle, distance):
        return (angle.to(u.arcsec)).value * (distance.to(u.pc)).value * u.AU

    def estimate_extreme_velocities(self, threshold, source_distance, plot=False, weak_quadrants=False):
        self.table = QTable(names=('Position', 'Channel', 'Angular distance', 'Distance', 'Velocity'),
                           dtype=(int, int, u.Quantity, u.Quantity, u.Quantity))
        self.estimate_extreme_channels(threshold, plot=False, weak_quadrants=weak_quadrants)
        for position, channel in enumerate(self.channels):
            angular_distance = (position - self.position_reference) * self.position_resolution
            distance = self.__angle_to_length(angular_distance, source_distance)
            velocity = (channel - self.vLSR_channel) * self.velocity_resolution + self.vLSR
            try:
                self.table.add_row([position, channel, angular_distance.value, distance.value, velocity.value])
            except AttributeError:
                #print([position, channel, angular_distance, distance, velocity])
                pass
        self.table['Angular distance'] = self.table['Angular distance']  * self.position_resolution.unit
        self.table['Distance'] = self.table['Distance'] * u.AU
        self.table['Velocity'] = self.table['Velocity'] * self.velocity_resolution.unit
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

    def write_table(self, filename, x_offset=None, x_unit=u.arcsec):
        x = self.table['Angular distance']
        x = x.to(x_unit)
        if x_offset is not None:
            x += x_offset
        y = self.table['Velocity']
        t = Table([x.value, y.value])
        ascii.write(t, output=filename, overwrite=True)




# other functions
# Modelling
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
                    **kwargs):

    """
    # Arguments
    threshold (int/ float): Set as multiples of PVdata.noise (for instance
        3sigma)
    source_distance (astropy.units.quantity):

    # Keyword arguments
    return_stddevs (boolean): The fit method LevMarLSQFitter is able to return
        the standard deviation of the fit parameters. Default is True.
    print_results (boolean): If True, the fit parameters will be displayed to
        the terminal.
    plot (boolean): If True, the fit will be displayed as a matplotlib pyplot.

    #Optional keyword arguments:
    flag_radius (astropy.units.Quantity): If given, then all data points within
        this given radius from the position_reference are flagged.
    flag_intervals (list of tupels of astropy.units.Quantity): Similar to
        flag_radius, but arbitrary intervals may be flagged. Each interval is
        given as a tuple of two radial distances from the position_reference.

    # Returns:
    best_fit (astropy.modelling.models.custom_model):
    stddevs (numpy.array): Only if return_stddevs is True. The array entries
        correspond to the best_fit instance parameters in the same order.
    chi2 (float): chi-squared residual of the fit to the unflagged data.
    """

    if 'model_kwargs' in kwargs:
        model_kwargs = kwargs['model_kwargs']
    else:
        model_kwargs = {}

    if self.start_low(weak_quadrants=weak_quadrants):
        init = Keplerian1D(mass=10., v0=self.vLSR.value, r0=0, bounds={'mass': (0.0, None)}, **model_kwargs)
    else:
        init = Keplerian1D_neg(mass=10., v0=self.vLSR.value, r0=0, bounds={'mass': (0.0, None)}, **model_kwargs)
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
        print('>> flagged the elements {}.'.format(i))
    if 'flag_radius' in kwargs:
        flag_radius = kwargs['flag_radius']
        print('Flagging towards a radial distance of {}:'.format(flag_radius))
        flag_radius = flag_radius.to(u.AU).value
        i = np.where(np.abs(self.table['Distance'].value) < flag_radius)[0]
        xdata.mask[i] = True
        ydata.mask[i] = True
        print('>> flagged the elements {}.'.format(i))
    if 'flag_intervals' in kwargs:
        flag_intervals = kwargs['flag_intervals']
        print('Flagging intervals:')
        flagged = []
        for interval in flag_intervals:
            i1 = np.where(interval[0].value < self.table['Distance'].value)[0]
            i2 = np.where(self.table['Distance'].value < interval[1].value)[0]
            i = np.intersect1d(i1, i2)
            xdata.mask[i] = True
            ydata.mask[i] = True
            flagged.append(i)
        if len(np.unique(flagged)) < 10:
            print('>> flagged the elements {}.'.format(np.unique(flagged)))
        else:
            print('>> flagged {} elements.'.format(len(np.unique(flagged))))

    # model
    with warnings.catch_warnings():
        # Catching RuntimeWarnings turning them to errors
        warnings.simplefilter('error')
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
    if print_results:
        pass
    return best_fit, chi2
