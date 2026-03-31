import math
import numpy as np
from bytedmd import measureDMD


def add(a, b, c, d):
    return b + c


def test_add_int8():
    a, b, c, d = np.int8(1), np.int8(2), np.int8(3), np.int8(4)
    cost = measureDMD(add, a, b, c, d)
    assert cost == math.sqrt(3) + math.sqrt(2)


def test_add_int16():
    a, b, c, d = np.int16(1), np.int16(2), np.int16(3), np.int16(4)
    cost = measureDMD(add, a, b, c, d)
    assert cost == math.sqrt(6) + math.sqrt(5) + math.sqrt(4) + math.sqrt(3)


def test_add_hybrid():
    a, b, c, d = np.int8(1), np.int16(2), np.int16(3), np.int8(4)
    cost = measureDMD(add, a, b, c, d)
    assert cost == math.sqrt(5) + math.sqrt(4) + math.sqrt(3) + math.sqrt(2)


def hybridInt8Int16(a, b, c, d):
    e = b + c
    f = a + d
    return e > f


def test_hybridInt8Int16():
    a, b, c, d = np.int8(1), np.int16(2), np.int16(3), np.int8(4)
    cost = measureDMD(hybridInt8Int16, a, b, c, d)
    # e=b+c: read b at distances 5,4; read c at distances 3,2
    # f=a+d: read a at distance 8; read d at distance 7
    # e>f:   read e at distances 5,4; read f at distance 1
    expected = (math.sqrt(5) + math.sqrt(4) + math.sqrt(3) + math.sqrt(2)
                + math.sqrt(8) + math.sqrt(7)
                + math.sqrt(5) + math.sqrt(4) + math.sqrt(1))
    assert cost == expected