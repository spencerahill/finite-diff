import itertools
import sys
import unittest

import numpy as np
import xarray as xr

from indiff._constants import LAT_STR, LON_STR, PFULL_STR
from indiff.advec import (PhysUpwind, LonUpwind, LatUpwind, SphereUpwind,
                          EtaUpwind, SphereEtaUpwind, LonUpwindConstP,
                          LatUpwindConstP)
from indiff.deriv import (PhysDeriv, LonFwdDeriv, LonBwdDeriv, LatFwdDeriv,
                          LatBwdDeriv, EtaFwdDeriv, EtaBwdDeriv,
                          SphereEtaFwdDeriv, SphereEtaBwdDeriv)
from . import InfiniteDiffTestCase


class PhysAdvecSharedTests(object):
    def test_init(self):
        assert isinstance(self.advec_obj._deriv_bwd_obj, self._DERIV_BWD_CLS)
        assert isinstance(self.advec_obj._deriv_fwd_obj, self._DERIV_FWD_CLS)

    def test_advec(self):
        self.assertNotImplemented(self.advec_obj.advec)


class PhysUpwindTestCase(InfiniteDiffTestCase):
    _ADVEC_CLS = PhysUpwind
    _DERIV_FWD_CLS = PhysDeriv
    _DERIV_BWD_CLS = PhysDeriv
    _DIM = LAT_STR

    def setUp(self):
        super(PhysUpwindTestCase, self).setUp()
        self.flow = xr.DataArray(
            np.random.random((len(self.pfull), len(self.lat), len(self.lon))),
            dims=[PFULL_STR, LAT_STR, LON_STR],
            coords={PFULL_STR: self.pfull, LAT_STR: self.lat,
                    LON_STR: self.lon}
        )*10. - 5.
        self.ps = xr.DataArray(
            np.random.random((len(self.lat), len(self.lon))),
            dims=[LAT_STR, LON_STR],
            coords={LAT_STR: self.lat, LON_STR: self.lon}
        )*1e3 + 1e5
        self.arr = xr.DataArray(
            np.random.random((len(self.pfull), len(self.lat), len(self.lon))),
            dims=[PFULL_STR, LAT_STR, LON_STR],
            coords={PFULL_STR: self.pfull, LAT_STR: self.lat,
                    LON_STR: self.lon}
        )
        self.advec_obj = self._ADVEC_CLS(self.flow, self.arr, self._DIM)


class TestPhysUpwind(PhysAdvecSharedTests, PhysUpwindTestCase):
    pass


class LonUpwindTestCase(PhysUpwindTestCase):
    _ADVEC_CLS = LonUpwind
    _DERIV_FWD_CLS = LonFwdDeriv
    _DERIV_BWD_CLS = LonBwdDeriv
    _DIM = LON_STR


class TestLonUpwind(PhysAdvecSharedTests, LonUpwindTestCase):
    def test_advec(self):
        self.advec_obj.advec(self.lat)


class LatUpwindTestCase(LonUpwindTestCase):
    _ADVEC_CLS = LatUpwind
    _DERIV_FWD_CLS = LatFwdDeriv
    _DERIV_BWD_CLS = LatBwdDeriv
    _DIM = LAT_STR


class TestLatUpwind(PhysAdvecSharedTests, LatUpwindTestCase):
    def test_advec(self):
        self.advec_obj.advec()


class SphereUpwindTestCase(InfiniteDiffTestCase):
    _ADVEC_CLS = SphereUpwind

    def setUp(self):
        super(SphereUpwindTestCase, self).setUp()
        self.flow = xr.DataArray(
            np.random.random((len(self.pfull), len(self.lat), len(self.lon))),
            dims=[PFULL_STR, LAT_STR, LON_STR],
            coords={PFULL_STR: self.pfull, LAT_STR: self.lat,
                    LON_STR: self.lon}
        )
        self.arr = xr.DataArray(np.random.random(self.flow.shape),
                                dims=self.flow.dims, coords=self.flow.coords)
        self.advec_obj = self._ADVEC_CLS(self.arr)


class TestSphereUpwind(SphereUpwindTestCase):
    def test_advec(self):
        self.advec_obj.advec_x(self.flow)
        self.advec_obj.advec_y(self.flow)
        self.advec_obj.advec(self.flow, self.flow)

    def test_advec_output_coords(self):
        desired = self.arr
        orders = [1, 2]
        cyclics = [True, False]
        methods = ['advec_x', 'advec_y']
        for o, cyclic, method in itertools.product(orders, cyclics, methods):
            advec_obj = self._ADVEC_CLS(self.arr, order=o, cyclic_lon=cyclic)
            actual = getattr(advec_obj, method)(self.flow)
            self.assertCoordsIdentical(actual, desired)

    def test_advec_zero_flow(self):
        zeros = self.flow.copy()
        zeros.values = np.zeros(zeros.shape)
        self.assertAllZeros(self.advec_obj.advec_x(zeros))
        self.assertAllZeros(self.advec_obj.advec_y(zeros))
        self.assertAllZeros(self.advec_obj.advec(zeros, zeros))


class EtaUpwindTestCase(InfiniteDiffTestCase):
    _ADVEC_CLS = EtaUpwind
    _DERIV_FWD_CLS = EtaFwdDeriv
    _DERIV_BWD_CLS = EtaBwdDeriv

    def setUp(self):
        super(EtaUpwindTestCase, self).setUp()
        self.flow = xr.DataArray(
            np.random.random((len(self.pfull), len(self.lat), len(self.lon))),
            dims=[PFULL_STR, LAT_STR, LON_STR],
            coords={PFULL_STR: self.pfull, LAT_STR: self.lat,
                    LON_STR: self.lon}
        )
        self.ps = xr.DataArray(
            np.random.random((len(self.lat), len(self.lon))),
            dims=[LAT_STR, LON_STR],
            coords={LAT_STR: self.lat, LON_STR: self.lon}
        )*1e3 + 1e5
        self.arr = xr.DataArray(
            np.random.random((len(self.pfull), len(self.lat), len(self.lon))),
            dims=[PFULL_STR, LAT_STR, LON_STR],
            coords={PFULL_STR: self.pfull, LAT_STR: self.lat,
                    LON_STR: self.lon}
        )
        self.advec_obj = self._ADVEC_CLS(self.flow, self.arr, self.pk, self.bk,
                                         self.ps)


class TestEtaUpwind(PhysAdvecSharedTests, EtaUpwindTestCase):
    def test_advec(self):
        self.advec_obj.advec()


class LonUpwindConstPTestCase(EtaUpwindTestCase):
    _ADVEC_CLS = LonUpwindConstP
    _DERIV_FWD_CLS = SphereEtaFwdDeriv
    _DERIV_BWD_CLS = SphereEtaBwdDeriv
    _DERIV_METHOD = 'd_dx_const_p'
    _DIM = LON_STR

    def setUp(self):
        super(LonUpwindConstPTestCase, self).setUp()


class TestLonUpwindConstP(PhysAdvecSharedTests, LonUpwindConstPTestCase):
    def test_init(self):
        assert self.advec_obj._DERIV_METHOD == self._DERIV_METHOD
        assert self.advec_obj._DERIV_BWD_CLS == self._DERIV_BWD_CLS
        assert self.advec_obj._DERIV_FWD_CLS == self._DERIV_FWD_CLS
        assert self.advec_obj._DIM == self._DIM

    def test_advec(self):
        self.advec_obj.advec()

    def test_advec_output_coords(self):
        desired = self.arr
        for o in [1, 2]:
            # Cyclic.
            actual = self._ADVEC_CLS(self.flow, self.arr, self.pk, self.bk,
                                     self.ps, order=o, cyclic=True,
                                     fill_edge=False).advec()
            self.assertCoordsIdentical(actual, desired)
            # Not cyclic, but fill edges.
            actual = self._ADVEC_CLS(self.flow, self.arr, self.pk, self.bk,
                                     self.ps, order=o, cyclic=False,
                                     fill_edge=True).advec()
            self.assertCoordsIdentical(actual, desired)

    def test_advec_zero_flow(self):
        zeros = self.flow.copy()
        zeros.values = np.zeros(zeros.shape)
        self.assertAllZeros(self._ADVEC_CLS(zeros, self.arr, self.pk, self.bk,
                                            self.ps).advec())

    def test_advec_unity_flow(self):
        actual = self._ADVEC_CLS(xr.ones_like(self.flow), self.arr, self.pk,
                                 self.bk, self.ps, order=1, cyclic=True,
                                 fill_edge=False).advec()
        desired = self._DERIV_BWD_CLS(self.arr, self.pk, self.bk, self.ps,
                                      order=1, cyclic_lon=True,
                                      fill_edge_lon=False).d_dx_const_p()
        xr.testing.assert_identical(actual, desired)

    def test_advec_unity_flow_not_cyclic(self):
        actual = self._ADVEC_CLS(xr.ones_like(self.flow), self.arr, self.pk,
                                 self.bk, self.ps, order=1, cyclic=False,
                                 fill_edge=True).advec()
        desired = self._DERIV_BWD_CLS(self.arr, self.pk, self.bk, self.ps,
                                      order=1, cyclic_lon=False,
                                      fill_edge_lon=True).d_dx_const_p()
        xr.testing.assert_identical(actual, desired)


class LatUpwindConstPTestCase(EtaUpwindTestCase):
    _ADVEC_CLS = LatUpwindConstP
    _DERIV_FWD_CLS = SphereEtaFwdDeriv
    _DERIV_BWD_CLS = SphereEtaBwdDeriv
    _DERIV_METHOD = 'd_dy_const_p'
    _DIM = LAT_STR

    def setUp(self):
        super(LatUpwindConstPTestCase, self).setUp()


class TestLatUpwindConstP(TestEtaUpwind, LatUpwindConstPTestCase):
    def test_advec_unity_flow(self):
        actual = self._ADVEC_CLS(xr.ones_like(self.flow), self.arr, self.pk,
                                 self.bk, self.ps, order=1,
                                 fill_edge=True).advec()
        desired = self._DERIV_BWD_CLS(self.arr, self.pk, self.bk, self.ps,
                                      order=1).d_dy_const_p()
        xr.testing.assert_identical(actual, desired)


class SphereEtaUpwindTestCase(InfiniteDiffTestCase):
    _ADVEC_CLS = SphereEtaUpwind

    def setUp(self):
        super(SphereEtaUpwindTestCase, self).setUp()
        self.flow = xr.DataArray(
            np.random.random((len(self.pfull), len(self.lat), len(self.lon))),
            dims=[PFULL_STR, LAT_STR, LON_STR],
            coords={PFULL_STR: self.pfull, LAT_STR: self.lat,
                    LON_STR: self.lon}
        )
        self.ps = xr.DataArray(
            np.random.random((len(self.lat), len(self.lon))),
            dims=[LAT_STR, LON_STR],
            coords={LAT_STR: self.lat, LON_STR: self.lon}
        )*1e3 + 1e5
        self.arr = xr.DataArray(
            np.random.random((len(self.pfull), len(self.lat), len(self.lon))),
            dims=[PFULL_STR, LAT_STR, LON_STR],
            coords={PFULL_STR: self.pfull, LAT_STR: self.lat,
                    LON_STR: self.lon}
        )
        self.advec_obj = self._ADVEC_CLS(self.arr, self.pk, self.bk, self.ps)


class TestSphereEtaUpwind(SphereEtaUpwindTestCase):
    def test_advec(self):
        self.advec_obj.advec_x_const_p(self.flow)
        self.advec_obj.advec_y_const_p(self.flow)
        self.advec_obj.advec_p(self.flow)

    def test_advec_output_coords(self):
        desired = self.arr
        orders = [1, 2]
        cyclics = [True, False]
        methods = ['advec_x_const_p', 'advec_y_const_p', 'advec_p']
        for o, cyclic, method in itertools.product(orders, cyclics, methods):
            advec_obj = self._ADVEC_CLS(self.arr, self.pk, self.bk,
                                        self.ps, order=o, cyclic_lon=cyclic)
            actual = getattr(advec_obj, method)(self.flow)
            self.assertCoordsIdentical(actual, desired)

    def test_advec_zero_flow(self):
        zeros = self.flow.copy()
        zeros.values = np.zeros(zeros.shape)
        self.assertAllZeros(self._ADVEC_CLS(self.arr, self.pk, self.bk,
                                            self.ps).advec_x_const_p(zeros))


if __name__ == '__main__':
    sys.exit(unittest.main())
