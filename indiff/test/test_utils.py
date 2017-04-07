import itertools
import sys
import unittest

import numpy as np
import pytest
import xarray as xr

from indiff._constants import LON_STR
from indiff.utils import wraparound

from . import InfiniteDiffTestCase


class WraparoundTestCase(InfiniteDiffTestCase):
    def setUp(self):
        super(WraparoundTestCase, self).setUp()
        self.arr = xr.DataArray(np.random.random(self.lon.shape),
                                dims=self.lon.dims, coords=self.lon.coords)


class TestWraparound(WraparoundTestCase):
    def test_no_wrap(self):
        desired = self.random
        dim = [self.dim, self.dummy_dim]
        circumf = [360, False, 'abc']
        spacing = range(5)
        for d, c, s in itertools.product(dim, circumf, spacing):
            actual = wraparound(self.random, d, left_to_right=0,
                                right_to_left=0, circumf=c, spacing=s)
            xr.testing.assert_identical(actual, desired)

    def test_1d_left_to_right_no_circumf(self):
        dim = LON_STR
        for i in range(1, 5):
            trunc = slice(0, i)
            edge = self.arr[{dim: trunc}].copy()
            desired = xr.concat([self.arr, edge], dim=dim)
            actual = wraparound(self.arr, dim, left_to_right=i,
                                right_to_left=0, circumf=0, spacing=1)
            xr.testing.assert_identical(actual, desired)

    def test_1d_right_to_left_no_circumf(self):
        dim = LON_STR
        for i in range(1, 5):
            trunc = slice(-i, None)
            edge = self.arr[{dim: trunc}].copy()
            desired = xr.concat([edge, self.arr], dim=dim)
            actual = wraparound(self.arr, dim, left_to_right=0,
                                right_to_left=i, circumf=0, spacing=1)
            xr.testing.assert_identical(actual, desired)

    @pytest.mark.xfail(reason='known bug w/ two-way wraparound')
    def test_1d_both_dir_no_circumf(self):
        dim = LON_STR
        ileft = range(1, 5)
        iright = range(1, 5)
        for l, r in itertools.product(ileft, iright):
            trunc_left = slice(0, l)
            trunc_right = slice(-r, None)
            edge_left = self.arr[{dim: trunc_left}].copy()
            edge_right = self.arr[{dim: trunc_right}].copy()
            desired = xr.concat([edge_right, self.arr, edge_left], dim=dim)
            actual = wraparound(self.arr, dim, left_to_right=l,
                                right_to_left=r, circumf=0, spacing=1)
            xr.testing.assert_identical(actual, desired)

    def test_1d_left_to_right_circumf(self):
        dim = LON_STR
        circumf = 360.
        for i in range(1, 5):
            trunc = slice(0, i)
            edge = self.arr.copy()[{dim: trunc}]
            edge_coord_values = edge[dim].values + circumf
            edge[dim].values = edge_coord_values
            desired = xr.concat([self.arr, edge], dim=dim)
            actual = wraparound(self.arr, dim, left_to_right=i,
                                right_to_left=0, circumf=circumf, spacing=1)
            xr.testing.assert_identical(actual, desired)

    def test_1d_right_to_left_circumf(self):
        dim = LON_STR
        circumf = 360.
        for i in range(1, 5):
            trunc = slice(-i, None)
            edge = self.arr.copy()[{dim: trunc}]
            edge_coord_values = edge[dim].values - circumf
            edge[dim].values = edge_coord_values
            desired = xr.concat([edge, self.arr], dim=dim)
            actual = wraparound(self.arr, dim, left_to_right=0,
                                right_to_left=i, circumf=circumf, spacing=1)
            xr.testing.assert_identical(actual, desired)

    @pytest.mark.xfail(reason='known bug w/ two-way wraparound')
    def test_1d_both_dir_circumf(self):
        dim = LON_STR
        circumf = 360.
        ileft = range(1, 5)
        iright = range(1, 5)
        for l, r in itertools.product(ileft, iright):
            trunc_left = slice(0, l)
            edge_left = self.arr.copy()[{dim: trunc_left}]
            edge_left_coord_values = edge_left[dim].values + circumf
            edge_left[dim].values = edge_left_coord_values

            trunc_right = slice(-r, None)
            edge_right = self.arr.copy()[{dim: trunc_right}]
            edge_right_coord_values = edge_right[dim].values - circumf
            edge_right[dim].values = edge_right_coord_values

            desired = xr.concat([edge_right, self.arr, edge_left], dim=dim)
            actual = wraparound(self.arr, dim, left_to_right=l,
                                right_to_left=r, circumf=circumf, spacing=1)
            xr.testing.assert_identical(actual, desired)


if __name__ == '__main__':
    sys.exit(unittest.main())
