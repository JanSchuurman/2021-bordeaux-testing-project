from population_growth import logistic_map
from population_growth import iterate_f
import pytest
import math
from numpy.testing import assert_array_almost_equal
import numpy as np
import random

@pytest.mark.parametrize("x, r, expected", [(0.1, 2.2, 0.198), (0.2, 3.4, 0.544), (0.75, 1.7, 0.31875)])
def test_logistic_map(x,r,expected):
    output = logistic_map(x, r)
    assert math.isclose(output, expected)
    
@pytest.mark.parametrize("x, r, my_iter, expected", [(0.1, 2.2, 1, [0.198]), (0.2, 3.4, 4, [0.544, 0.843418, 0.449019, 0.841163]), (0.75, 1.7, 2, [0.31875, 0.369152])])
def test_iterate_f(my_iter,x,r,expected):
    output, t = iterate_f(my_iter, x, r)
    assert_array_almost_equal(output, np.asarray(expected), decimal=3)


def test_fuzzing(random_state):
    r = 1.5
    my_iter = 100
    n = 100
    # random_state = np.random.RandomState(10)
    x_list = random_state.rand(n)
    expected = np.ones(n) * 1/3
    output = []
    for x in x_list:
       output.append(iterate_f(my_iter, x, r)[-1])
    
    assert_array_almost_equal(output, expected)
    
    
