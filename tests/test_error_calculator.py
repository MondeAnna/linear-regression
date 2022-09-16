from  utilities.error_calculator import ErrorCalculator as Calc


def test_instantiation():
    assert isinstance(Calc(), Calc)
