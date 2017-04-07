"""Testing module of infinite-diff."""
import unittest

import numpy as np
import pytest
import xarray as xr


class InfiniteDiffTestCase(unittest.TestCase):
    def setUp(self):
        self.array_len = 10
        self.dim = 'testdim'
        self.dummy_len = 3
        self.dummy_dim = 'dummydim'

        self.arange = xr.DataArray(
            np.arange(self.array_len*self.dummy_len).reshape((self.dummy_len,
                                                              self.array_len)),
            dims=[self.dummy_dim, self.dim],
            coords={self.dummy_dim: np.arange(self.dummy_len),
                    self.dim: np.arange(self.array_len)}
        )

        self.zeros = xr.zeros_like(self.arange)
        self.ones = xr.ones_like(self.arange)

        randstate = np.random.RandomState(12345)
        self.random = xr.DataArray(randstate.rand(*self.ones.shape),
                                   dims=self.ones.dims,
                                   coords=self.ones.coords)
        randstate = np.random.RandomState(54321)
        self.random2 = xr.DataArray(randstate.rand(*self.ones.shape),
                                    dims=self.ones.dims,
                                    coords=self.ones.coords)

        self.zeros_trunc = [self.zeros.isel(**{self.dim: slice(None, -(n+1))})
                            for n in range(self.array_len)]
        self.ones_trunc = [self.ones.isel(**{self.dim: slice(None, -(n+1))})
                           for n in range(self.array_len)]
        self.arange_trunc = [self.arange.isel(**{self.dim:
                                                 slice(None, -(n+1))})
                             for n in range(self.array_len)]
        self.random_trunc = [self.random.isel(**{self.dim:
                                                 slice(None, -(n+1))})
                             for n in range(self.array_len)]

        lon = np.arange(5., 356, 10)
        lat = np.arange(-85., 86, 10)
        phalf = np.array(
            [1., 34.747942, 75.055556, 127.872428, 191.113683,
             262.117174, 339.069366, 419.987958, 498.927204, 570.232663,
             634.339923, 691.673132, 742.649553, 787.681052, 827.17644,
             861.541426, 891.180034, 916.494699, 937.886685, 955.756486,
             970.503752, 982.527596, 1000.]
        )
        pk = np.array(
            [100., 3474.79418945, 7505.55566406, 12787.24316406,
             19111.36914062, 21854.92773438, 22884.1875, 22776.30664062,
             21716.16015625, 20073.296875, 18110.51171875, 16004.78320312,
             13877.625, 11812.54492188, 9865.88378906, 8073.97265625,
             6458.08349609, 5027.98974609, 3784.60839844, 2722.00854492,
             1828.97521973, 1090.23962402, 0.]
        )
        bk = np.array(
            [0., 0., 0., 0., 0., 0.0435679, 0.1102275, 0.1922249,
             0.28176561, 0.36949971, 0.45323479, 0.53162527, 0.60387331,
             0.6695556, 0.72851759, 0.78080171, 0.82659918, 0.86621481,
             0.90004063, 0.92853642, 0.952214, 0.97162521, 1.]
        )
        pfull = np.array(
            [19.08839744, 52.3401932, 99.12992393, 157.38101859,
             224.74922656, 298.94438175, 378.08657048, 458.32513787,
             533.78639558, 601.71723231, 662.59316591, 716.85928688,
             764.94440205, 807.26772719, 844.24236698, 876.2771919,
             903.77827901, 927.14956119, 946.79347942, 963.11130143,
             976.5033364, 987.36930553]
        )
        pressure = np.array(
            [1000., 925., 850., 700., 600., 500., 400., 300., 250., 200., 150.,
             100., 70., 50., 30., 20., 10.]
        )
        sigma = np.arange(0.1, 1.01, 0.1)
        self.lon = xr.DataArray(lon, dims=['lon'], coords={'lon': lon})
        self.lat = xr.DataArray(lat, dims=['lat'], coords={'lat': lat})
        self.phalf = xr.DataArray(phalf, dims=['phalf'],
                                  coords={'phalf': phalf})
        self.bk = xr.DataArray(bk, dims=self.phalf.dims,
                               coords=self.phalf.coords)
        self.pk = xr.DataArray(pk, dims=self.phalf.dims,
                               coords=self.phalf.coords)
        self.pfull = xr.DataArray(pfull, dims=['pfull'],
                                  coords={'pfull': pfull})
        self.pressure = xr.DataArray(pressure, dims=['pressure'],
                                     coords={'pressure': pressure})
        self.sigma = xr.DataArray(sigma, dims=['sigma'],
                                  coords={'sigma': sigma})

    def assertArrayEqual(self, arr1, arr2):
        np.testing.assert_array_equal(arr1, arr2)

    def assertDatasetIdentical(self, d1, d2):
        assert d1.identical(d2), (d1, d2, d1/d2)

    def assertCoordsIdentical(self, d1, d2):
        for dim in d1.dims:
            assert d1[dim].identical(d2[dim]), (dim, d1[dim], d2[dim])
        assert d1.shape == d2.shape, ('shape mismatch', d1.shape, d2.shape)

    def assertNotImplemented(self, func, *args, **kwargs):
        with pytest.raises(NotImplementedError):
            func(*args, **kwargs)

    def assertAllZeros(self, arr):
        assert not np.any(arr), arr
