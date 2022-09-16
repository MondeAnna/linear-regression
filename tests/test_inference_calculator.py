from utilities.inference_calculator import InferenceCalculator as Calc


def test_instantiation():
    assert isinstance(Calc(), Calc)
