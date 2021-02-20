import unittest
import pytest

from picograd.configs.base import BaseConfig


class TestConfigs(unittest.TestCase):
    def test_simple(self):
        class X(BaseConfig):
            def __init__(self):
                self.a = 2
                self.b = None
                super().__init__()

        with pytest.raises(ValueError):
            x = X()
            x.override_attrs(a=3)

        x = X()
        x.override_attrs(a=3, b=1)
        assert x.a == 3
        assert x.b == 1

    def test_bools(self):
        class X(BaseConfig):
            def __init__(self):
                self.a = 2
                self.b = False
                super().__init__()

        with pytest.raises(ValueError):
            x = X()
            x.override_attrs(b='TT')

        x = X()
        x.override_attrs(b='T')
        assert x.a == 2
        assert x.b == True

        x.override_attrs(b='0')
        assert x.b == False

        x.override_attrs(b='true')
        assert x.b == True

        x.override_attrs(b='F')
        assert x.b == False

    def test_int(self):
        class X(BaseConfig):
            def __init__(self):
                self.a = 2
                super().__init__()

        with pytest.raises(ValueError):
            x = X()
            x.override_attrs(a='wow')

        x = X()
        x.override_attrs(a='10')
        assert x.a == 10

        x.override_attrs(a='-5')
        assert x.a == -5

        with pytest.raises(ValueError, match='invalid literal for int()'):
            x.override_attrs(a='3.14')
