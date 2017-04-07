import sys
import unittest

import pytest
import xarray as xr

from indiff import (FiniteDiff, OneSidedDiff, BwdDiff, FwdDiff, CenDiff,
                    FiniteDeriv, OneSidedDeriv, BwdDeriv, FwdDeriv, CenDeriv)

from . import InfiniteDiffTestCase


class DerivSharedTests(object):
    def test_arr_coord(self):
        xr.testing.assert_identical(self.deriv_obj._arr_coord(None),
                                    self.arr[self.dim])
        xr.testing.assert_identical(self.deriv_obj._arr_coord(self.random),
                                    self.random)

    def test_init(self):
        xr.testing.assert_identical(self.deriv_obj.arr, self.arr)
        assert self.dim == self.deriv_obj.dim
        assert isinstance(self.deriv_obj._arr_diff_obj, self._DIFF_CLS)
        assert isinstance(self.deriv_obj._coord_diff_obj, self._DIFF_CLS)

    def test_slice_edge(self):
        with pytest.raises(NotImplementedError):
            self.deriv_obj._slice_edge(0)

    def test_concat(self):
        with pytest.raises(NotImplementedError):
            self.deriv_obj._concat()

    def test_private_deriv(self):
        with pytest.raises(NotImplementedError):
            self.deriv_obj._deriv()

    def test_public_deriv(self):
        with pytest.raises(NotImplementedError):
            self.deriv_obj.deriv()


class FiniteDerivTestCase(InfiniteDiffTestCase):
    _DIFF_CLS = FiniteDiff
    _DERIV_CLS = FiniteDeriv

    def setUp(self):
        super(FiniteDerivTestCase, self).setUp()
        self.arr = self.arange
        self.deriv_obj = self._DERIV_CLS(self.arr, self.dim)
        self.method = self.deriv_obj.deriv


class TestFiniteDeriv(DerivSharedTests, FiniteDerivTestCase):
    pass


class OneSidedDerivTestCase(FiniteDerivTestCase):
    _DIFF_CLS = OneSidedDiff
    _DERIV_CLS = OneSidedDeriv

    def setUp(self):
        super(OneSidedDerivTestCase, self).setUp()


class TestOneSidedDeriv(TestFiniteDeriv, OneSidedDerivTestCase):
    def test_edge_deriv_rev(self):
        with pytest.raises(NotImplementedError):
            self.deriv_obj._edge_deriv_rev()


class FwdDerivTestCase(OneSidedDerivTestCase):
    _DIFF_CLS = FwdDiff
    _DERIV_CLS = FwdDeriv

    def setUp(self):
        super(FwdDerivTestCase, self).setUp()
        self.is_bwd = False


class TestFwdDeriv(TestOneSidedDeriv, FwdDerivTestCase):
    def test_slice_edge(self):
        spacing, order = 1, 1
        deriv_obj = self._DERIV_CLS(self.arr, self.dim, spacing=spacing,
                                    order=order, fill_edge=False)
        actual = deriv_obj._slice_edge(self.arr)
        desired = self.arr[{self.dim: slice(-(spacing*order), None)}]
        xr.testing.assert_identical(actual, desired)

    def test_diff_arr(self):
        desired = self._DIFF_CLS(self.arr, self.dim).diff()
        actual = self.deriv_obj._arr_diff_obj.diff()
        xr.testing.assert_identical(actual, desired)

    def test_diff_coord(self):
        desired = self._DIFF_CLS(self.arr[self.dim], self.dim).diff()
        actual = self.deriv_obj._coord_diff_obj.diff()
        xr.testing.assert_identical(actual, desired)

    def test_concat(self):
        actual = self.deriv_obj._concat(self.ones, self.arr)
        desired = xr.concat([self.ones, self.arr], dim=self.dim)
        xr.testing.assert_identical(actual, desired)

    def test_edge_deriv_rev(self):
        self.deriv_obj._edge_deriv_rev()

    def test_private_deriv(self):
        self.deriv_obj._deriv()

    def test_public_deriv(self):
        self.deriv_obj.deriv()

    def test_deriv_constant_slope_order1_no_fill(self):
        actual = self._DERIV_CLS(self.arange, self.dim,
                                 fill_edge=False).deriv()
        desired = self.ones_trunc[0]
        xr.testing.assert_identical(actual, desired)

    def test_deriv_constant_slope_order1_fill(self):
        actual = self._DERIV_CLS(self.arange, self.dim, fill_edge=True).deriv()
        desired = self.ones
        xr.testing.assert_identical(actual, desired)

    def test_deriv_constant_slope_order2_no_fill(self):
        actual = self._DERIV_CLS(self.arr, self.dim, order=2,
                                 fill_edge=False).deriv()
        desired = self.ones_trunc[1]
        xr.testing.assert_identical(actual, desired)

    def test_deriv_constant_slope_order2_fill(self):
        actual = self._DERIV_CLS(self.arr, self.dim, order=2,
                                 fill_edge=True).deriv()
        desired = self.ones
        xr.testing.assert_identical(actual, desired)

    def test_deriv_zero_slope_order1_no_fill(self):
        actual = self._DERIV_CLS(self.ones, self.dim, order=1,
                                 fill_edge=False).deriv()
        desired = self.zeros_trunc[0]
        xr.testing.assert_identical(actual, desired)

    def test_deriv_zero_slope_order1_fill(self):
        actual = self._DERIV_CLS(self.ones, self.dim, order=1,
                                 fill_edge=True).deriv()
        desired = self.zeros
        xr.testing.assert_identical(actual, desired)

    def test_deriv_zero_slope_order2_no_fill(self):
        actual = self._DERIV_CLS(self.ones, self.dim, order=2,
                                 fill_edge=False).deriv()
        desired = self.zeros_trunc[1]
        xr.testing.assert_identical(actual, desired)

    def test_deriv_zero_slope_order2_fill(self):
        actual = self._DERIV_CLS(self.ones, self.dim, order=2,
                                 fill_edge=True).deriv()
        desired = self.zeros
        xr.testing.assert_identical(actual, desired)

    def test_deriv_order1_spacing1_no_fill(self):
        label = 'upper' if self.is_bwd else 'lower'
        desired = (self.random.diff(self.dim, label=label) /
                   self.random[self.dim].diff(self.dim, label=label))
        actual = self._DERIV_CLS(self.random, self.dim,
                                 fill_edge=False).deriv()
        xr.testing.assert_identical(actual, desired)

    def test_deriv_order1_spacing1_fill(self):
        label = 'lower'
        edge_label = 'upper'
        trunc = slice(-2, None)

        interior = (self.random.diff(self.dim, label=label) /
                    self.random[self.dim].diff(self.dim, label=label))

        arr_edge = self.random[{self.dim: trunc}]
        edge = (arr_edge.diff(self.dim, label=edge_label) /
                arr_edge[self.dim].diff(self.dim, label=edge_label))

        desired = xr.concat([interior, edge], dim=self.dim)
        actual = self._DERIV_CLS(self.random, self.dim, fill_edge=True).deriv()
        xr.testing.assert_identical(actual, desired)


class BwdDerivTestCase(FwdDerivTestCase):
    _DIFF_CLS = BwdDiff
    _DERIV_CLS = BwdDeriv

    def setUp(self):
        super(BwdDerivTestCase, self).setUp()
        self.is_bwd = True

        self.zeros_trunc = [self.zeros.isel(**{self.dim: slice(n+1, None)})
                            for n in range(self.array_len)]
        self.ones_trunc = [self.ones.isel(**{self.dim: slice(n+1, None)})
                           for n in range(self.array_len)]
        self.arange_trunc = [self.arange.isel(**{self.dim: slice(n+1, None)})
                             for n in range(self.array_len)]
        self.random_trunc = [self.random.isel(**{self.dim: slice(n+1, None)})
                             for n in range(self.array_len)]


class TestBwdDeriv(TestFwdDeriv, BwdDerivTestCase):
    def test_slice_edge(self):
        spacing, order = 1, 1
        actual = self.deriv_obj._slice_edge(self.arr)
        desired = self.arr[{self.dim: slice(None, (spacing*order))}]
        xr.testing.assert_identical(actual, desired)

    def test_concat(self):
        actual = self.deriv_obj._concat(self.ones, self.arr)
        desired = xr.concat([self.arr, self.ones], dim=self.dim)
        xr.testing.assert_identical(actual, desired)

    def test_deriv_order1_spacing1_fill(self):
        trunc = slice(0, 2)

        interior = (self._DIFF_CLS(self.random, self.dim).diff() /
                    self._DIFF_CLS(self.random[self.dim], self.dim).diff())

        arr_edge = self.random[{self.dim: trunc}]
        edge = (FwdDiff(arr_edge, self.dim).diff() /
                FwdDiff(arr_edge[self.dim], self.dim).diff())

        desired = xr.concat([edge, interior], dim=self.dim)
        actual = self._DERIV_CLS(self.random, self.dim, fill_edge=True).deriv()
        xr.testing.assert_identical(actual, desired)


class CenDerivTestCase(FiniteDerivTestCase):
    _DIFF_CLS = CenDiff
    _DERIV_CLS = CenDeriv

    def setUp(self):
        super(CenDerivTestCase, self).setUp()
        self.arr = self.arange
        self.zeros_trunc = [self.zeros.isel(**{self.dim: slice(n+1, -(n+1))})
                            for n in range(self.array_len // 2 - 1)]
        self.ones_trunc = [self.ones.isel(**{self.dim: slice(n+1, -(n+1))})
                           for n in range(self.array_len // 2 - 1)]
        self.arange_trunc = [self.arange.isel(**{self.dim: slice(n+1, -(n+1))})
                             for n in range(self.array_len // 2 - 1)]
        self.random_trunc = [self.random.isel(**{self.dim: slice(n+1, -(n+1))})
                             for n in range(self.array_len // 2 - 1)]


class TestCenDeriv(TestFiniteDeriv, CenDerivTestCase):
    def test_init(self):
        xr.testing.assert_identical(self.deriv_obj.arr, self.arr)
        assert self.dim == self.deriv_obj.dim
        assert isinstance(self.deriv_obj._deriv_fwd_obj, FwdDeriv)
        assert isinstance(self.deriv_obj._deriv_bwd_obj, BwdDeriv)

    def test_concat(self):
        actual = self.deriv_obj._concat(self.random, self.ones, self.arr)
        desired = xr.concat([self.random, self.ones, self.arr], dim=self.dim)
        xr.testing.assert_identical(actual, desired)

    def test_diff_arr(self):
        desired = self._DIFF_CLS(self.arr, self.dim).diff()
        actual = self.deriv_obj._arr_diff_obj.diff()
        xr.testing.assert_identical(actual, desired)

    def test_diff_coord(self):
        desired = self._DIFF_CLS(self.arr[self.dim], self.dim).diff()
        actual = self.deriv_obj._coord_diff_obj.diff()
        xr.testing.assert_identical(actual, desired)

    def test_private_deriv(self):
        desired = (self._DIFF_CLS(self.zeros, self.dim).diff() /
                   self._DIFF_CLS(self.zeros[self.dim], self.dim).diff())
        actual = self._DERIV_CLS(self.zeros, self.dim,
                                 fill_edge=False)._deriv()
        xr.testing.assert_identical(actual, desired)

    def test_public_deriv(self):
        self.deriv_obj.deriv()

    def test_deriv_constant_slope_order2_no_fill(self):
        actual = self._DERIV_CLS(self.arange, self.dim, order=2,
                                 fill_edge=False).deriv()
        desired = self.ones_trunc[0]
        xr.testing.assert_identical(actual, desired)

    def test_deriv_constant_slope_order2_fill(self):
        actual = self._DERIV_CLS(self.arange, self.dim, order=2,
                                 fill_edge=True).deriv()
        desired = self.ones
        xr.testing.assert_identical(actual, desired)

    def test_deriv_constant_slope_order4_no_fill(self):
        actual = self._DERIV_CLS(self.arange, self.dim, order=4,
                                 fill_edge=False).deriv()
        desired = self.ones_trunc[1]
        xr.testing.assert_identical(actual, desired)

    def test_deriv_constant_slope_order4_fill(self):
        actual = self._DERIV_CLS(self.arange, self.dim, order=4,
                                 fill_edge=True).deriv()
        desired = self.ones
        xr.testing.assert_identical(actual, desired)

    def test_deriv_zero_slope_order2_no_fill(self):
        actual = self._DERIV_CLS(self.ones, self.dim, order=2,
                                 fill_edge=False).deriv()
        desired = self.zeros_trunc[0]
        xr.testing.assert_identical(actual, desired)

    def test_deriv_zero_slope_order2_fill(self):
        actual = self._DERIV_CLS(self.ones, self.dim, order=2,
                                 fill_edge=True).deriv()
        desired = self.zeros
        xr.testing.assert_identical(actual, desired)

    def test_deriv_zero_slope_order4_no_fill(self):
        actual = self._DERIV_CLS(self.ones, self.dim, order=4,
                                 fill_edge=False).deriv()
        desired = self.zeros_trunc[1]
        xr.testing.assert_identical(actual, desired)

    def test_deriv_zero_slope_order4_fill(self):
        actual = self._DERIV_CLS(self.ones, self.dim, order=4,
                                 fill_edge=True).deriv()
        desired = self.zeros
        xr.testing.assert_identical(actual, desired)

    def test_deriv_order2_spacing1_fill(self):
        interior = (self._DIFF_CLS(self.random, self.dim).diff() /
                    self._DIFF_CLS(self.random[self.dim], self.dim).diff())

        trunc_left = slice(0, 2)
        arr_left = self.random[{self.dim: trunc_left}]
        left = FwdDeriv(arr_left, self.dim, order=1, fill_edge=False).deriv()

        trunc_right = slice(-2, None)
        arr_right = self.random[{self.dim: trunc_right}]
        right = BwdDeriv(arr_right, self.dim, order=1, fill_edge=False).deriv()

        desired = xr.concat([left, interior, right], dim=self.dim)
        actual = self._DERIV_CLS(self.random, self.dim, fill_edge=True).deriv()
        xr.testing.assert_identical(actual, desired)


if __name__ == '__main__':
    sys.exit(unittest.main())

# TODO: non-unity spacing
# TODO: comparison to analytical solutions, e.g. sin/cos, e^x
