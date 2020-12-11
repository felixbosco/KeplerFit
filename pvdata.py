import numpy as np

from astropy.io import fits
from astropy.units import Unit, Quantity


class PVData(object):

    # Default header keys when loading a PVData instance `from_file`. Change these to the header cards in your FITS file
    default_header_keys = {'position_reference': 'CRPIX1',
                           'd_pos': 'CDELT1',
                           'pos_unit': 'CUNIT1',
                           'v_lsr_channel': 'CRPIX2',
                           'd_vel': 'CDELT2',
                           'v_lsr': 'CRVAL2',
                           'vel_unit': 'CUNIT2',
                           'data_unit': 'BUNIT'}

    def __init__(self, data, noise=None, position_reference=None, v_lsr_channel=None, d_pos=None, d_vel=None,
                 pos_unit=None, vel_unit=None, data_unit=None, v_lsr=None, transpose=False):
        """Initialize a position-velocity data (PVData) object:

        Args:
            data (np.ndarray):
                Position-velocity data image.
            noise (float, optional):
                Uncertainty of `data` for parsing independent estimates. The code is using the standard deviation of
                `data` if not provided.
            position_reference (int, optional):
                Spatial index of the center of rotation in `data`.
            v_lsr_channel (int, optional):
                Spectral index of the local standard of rest in `data`.
            d_pos (float, optional):
                Increment in position in `pos_unit` per pixel.
            d_vel (float, optional):
                Increment in velocity in `vel_unit` per pixel.
            pos_unit (str, optional):
                Unit of spatial indexes per pixel.
            vel_unit (str, optional):
                Unit of spectral indexes per pixel.
            v_lsr (float, optional):
                Local standard of rest velocity in units of km/ s.
            transpose (bool, optional):
                Transpose the data array, if requested. This can be used if the first axis is not (spatial, velocity).
        """

        # Store input parameters
        self.data = np.squeeze(data)  # Get rid of additional axes
        self.noise = noise
        self.position_reference = int(position_reference)
        self.v_lsr_channel = int(v_lsr_channel)
        self.d_pos = d_pos
        self.d_vel = d_vel
        self.pos_unit = pos_unit
        self.vel_unit = vel_unit
        self.v_lsr = v_lsr

        # Check that data is 2-dimensional image
        if self.data.ndim != 2:
            raise ValueError(f"data have shape {self.data.shape!r} but is expected to be 2-dimensional!")

        # Transpose data if requested
        if transpose:
            self.data = self.data.transpose()

        # Check data validity
        if np.isnan(self.data).any():
            print('Masking invalid data...')
            self.data = np.ma.masked_invalid(self.data)

        # Set noise if not provided
        if self.noise is None:
            self.noise = np.std(self.data)

        # Initialize future attributes
        self.min_channels = None
        self.max_channels = None
        self.extreme_channels = None

    @classmethod
    def from_file(cls, file_name, extension=None, transpose=False):
        """Initialize an object from a FITS file.

        Args:
            file_name (str):
                Path to the file.
            extension (int or str, optional):
                Index or name of the desired FITS extension.
            transpose (bool, optional):
                Transpose the data array, if requested. This can be used if the first axis is not (spatial, velocity).

        Returns:
            pv_data (PVData object):
                Instance of the PVData class, based on the data and parameters from the FITS file.
        """

        # Read data from header
        data, hdr = fits.getdata(file_name, extension, header=True)

        # Extract other parameters from FITS header
        pars = {}
        for par, card in cls.default_header_keys.items():
            pars[par] = hdr.get(card)

        return cls(data=data, **pars, transpose=transpose)

    @property
    def shape(self):
        return self.data.shape

    def estimate_extreme_velocity_channels(self, sigma=3):

        number_channels = self.data.shape[0]

        # Mask all pixels with signal < sigma * noise
        valid_data = np.ma.masked_greater(self.data, sigma * self.noise).mask

        # Estimate channels as first entries with a True value
        min_channels = np.argmax(valid_data, axis=0)
        max_channels = number_channels - 1 - np.argmax(valid_data[::-1], axis=0)

        # Mask min values larger than v_LSR and max values smaller than v_LSR
        min_channels = np.ma.masked_values(min_channels, 0)
        max_channels = np.ma.masked_values(max_channels, number_channels - 1)

        # Store results
        self.min_channels = min_channels
        self.max_channels = max_channels

        return min_channels, max_channels

    def start_low(self, weak_quadrants=False):
        """Compare the flux in the four quadrants.

        Args:
            weak_quadrants (bool, optional):
                Invert return value.

        Returns:
            start_low(bool):
                True, if the flux in the pair of quadrants that contains low velocities at low positions is larger than
                the flux of the other two quadrants.
        """

        # Estimate total flux of quadrants
        q1 = np.sum(self.data[self.v_lsr_channel:, self.position_reference:])  # +x, +v
        q2 = np.sum(self.data[self.v_lsr_channel:, :self.position_reference])  # -x, +v
        q3 = np.sum(self.data[:self.v_lsr_channel, :self.position_reference])  # -x, -v
        q4 = np.sum(self.data[:self.v_lsr_channel, self.position_reference:])  # +x, -v

        if not weak_quadrants:
            return q1 + q3 > q2 + q4
        else:
            return q1 + q3 < q2 + q4

    def combine_extreme_channels(self, weak_quadrants=False):

        # Initialize extreme channels as an empty copy of the other channels attributes
        self.extreme_channels = np.ma.empty_like(self.min_channels)

        # Fill array with values from the quadrants
        if self.start_low(weak_quadrants=weak_quadrants):
            self.extreme_channels[:self.position_reference] = self.min_channels[:self.position_reference]
            self.extreme_channels[self.position_reference:] = self.max_channels[self.position_reference:]
        else:
            self.extreme_channels[:self.position_reference] = self.max_channels[:self.position_reference]
            self.extreme_channels[self.position_reference:] = self.min_channels[self.position_reference:]

    def estimate_extreme_velocities(self, sigma=3, distance=None, weak_quadrants=False, return_quantities=True):
        """

        Args:
            sigma (float, optional):
                Multiple of the data noise, used as a threshold for finding extreme channels.
            distance (float, optional):
                Source distance in units of parsecs. Can also be provided as an astropy.units.Quantity object.
            weak_quadrants (bool, optional):
                Use the combination of the weak quadrants.
            return_quantities (bool, optional):
                Return positions and velocities as astropy.units.Quantities, if `True`, otherwise np.ndarrays.

        Returns:
            positions (np.ndarray or astrop.units.Quantity):
                Positions of the measurements. Returned as a float or astropy.units.Quantity, depending on
                `return_quantities`. The unit of the float or Quantity is arcsec if distance is not provided, otherwise
                astronomical units (AU).
            velocities (np.ndarray or astrop.units.Quantity):
                Extreme velocities at the positions. Returned as a float (in units of km/ s) or astropy.units.Quantity,
                depending on `return_quantities`.
        """

        # Estimate extreme channels if not done yet
        if self.min_channels is None or self.max_channels is None:
            self.estimate_extreme_velocity_channels(sigma=sigma)

        # Combine extreme channels if not done yet
        if self.extreme_channels is None:
            self.combine_extreme_channels(weak_quadrants=weak_quadrants)

        # Remove masked values
        use = self.extreme_channels.mask == 0

        # Transform velocity channel indexes into velocities relative to the v_LSR channel
        velocities = (self.extreme_channels[use] - self.v_lsr_channel).astype(float)
        velocities *= self.d_vel
        velocities += self.v_lsr  # Offset by the v_LSR
        if isinstance(self.vel_unit, str):
            velocities = velocities * Unit(self.vel_unit)
            velocities = velocities.to('km/ s')
        else:
            velocities = velocities * Unit('km/ s')

        # Transform position indexes into angles relative to the position reference index
        positions = (np.arange(self.shape[1])[use] - self.position_reference).astype(float)
        positions *= self.d_pos
        if isinstance(self.pos_unit, str):
            positions = positions * Unit(self.pos_unit)
            positions = positions.to('arcsec')
        else:
            positions = positions * Unit('arcsec')

        # Transform angular positions into linear scale if the distance is provided
        if distance is not None:
            # Assert that distance is provided in units of parsecs
            if isinstance(distance, Quantity):
                distance = distance.to('pc')
            else:
                distance = distance * Unit('pc')

            positions = positions.value * distance.value * Unit('AU')

        if return_quantities:
            return positions, velocities
        else:
            return positions.value, velocities.value
