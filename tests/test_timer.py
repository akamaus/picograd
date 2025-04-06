""" Tests for global and local timers """
import torch

from picograd.utils.timer import Timer
import picograd.utils.timer as T

timer = Timer()


@T.wrap('multiply')
def multiply(m, k):
    for _ in range(k):
        x = m @ m

    return k

def test_timer():

    with timer.measure('nrand'):
        m = torch.randn([1000,1000])

    with timer.measure('mul'):
        m = m @ m

    r1 = multiply(m, 10)
    r2 = multiply(m, 20)

    assert r1 == 10
    assert r2 == 20

    timer.print_report()
    T.print_report()
