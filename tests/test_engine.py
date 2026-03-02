from micrograd.engine import Value
import math

def test_add():
    a = Value(3)
    b = Value(2)
    c = a + b
    assert c.data == 5

def test_mul():
    a = Value(3)
    b = Value(2)
    c = a * b
    assert c.data == 6

def test_radd():
    a = 3
    b = Value(2)
    c = a + b
    assert c.data == 5

def test_rmul():
    a = 3
    b = Value(2)
    c = a * b
    assert c.data == 6

def test_sub():
    a = Value(5)
    b = Value(2)
    c = a - b
    assert c.data == 3

def test_rsub():
    a = 10
    b = Value(3)
    c = a - b
    assert c.data == 7

def test_neg():
    a = Value(5)
    b = -a
    assert b.data == -5

def test_pow():
    a = Value(2)
    b = a ** 3
    assert b.data == 8

def test_pow_float():
    a = Value(4)
    b = a ** 0.5
    assert b.data == 2.0

def test_truediv():
    a = Value(10)
    b = Value(2)
    c = a / b
    assert c.data == 5

def test_truediv_float():
    a = Value(7)
    b = Value(2)
    c = a / b
    assert abs(c.data - 3.5) < 1e-6

def test_exp():
    a = Value(1)
    b = a.exp()
    assert abs(b.data - math.e) < 1e-6

def test_exp_zero():
    a = Value(0)
    b = a.exp()
    assert b.data == 1.0

def test_tanh():
    a = Value(0)
    b = a.tanh()
    assert abs(b.data - 0.0) < 1e-6

def test_tanh_positive():
    a = Value(1)
    b = a.tanh()
    expected = math.tanh(1)
    assert abs(b.data - expected) < 1e-6

def test_complex_expression():
    a = Value(2)
    b = Value(3)
    c = Value(4)
    d = (a * b + c) / 2
    assert d.data == 5.0