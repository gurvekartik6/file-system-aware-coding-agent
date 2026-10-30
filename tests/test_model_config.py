import pytest

from model_config import sampling_kwargs


def test_sampling_is_omitted_by_default(monkeypatch):
    monkeypatch.delenv("GEMINI_TEMPERATURE", raising=False)

    assert sampling_kwargs() == {}


def test_sampling_temperature_is_configurable(monkeypatch):
    monkeypatch.setenv("GEMINI_TEMPERATURE", "0.25")

    assert sampling_kwargs() == {"temperature": 0.25}


@pytest.mark.parametrize("value", ["not-a-number", "-0.1", "2.1"])
def test_invalid_sampling_temperature_is_rejected(monkeypatch, value):
    monkeypatch.setenv("GEMINI_TEMPERATURE", value)

    with pytest.raises(RuntimeError, match="GEMINI_TEMPERATURE"):
        sampling_kwargs()
