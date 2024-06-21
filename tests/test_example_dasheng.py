import importlib
import unittest

import torch

try:
    importlib.import_module("dasheng")
except ImportError:
    raise ImportError("dasheng is not installed. Have you ran `pip install -e .[example]`?")

from example.dasheng.dasheng_encoder import DashengEncoder


class TestDasheng(unittest.TestCase):
    def test_dasheng_base_rand_input(self):
        audio = torch.randn(2, 48_000)
        encoder = DashengEncoder()

        assert encoder.check_input_audio(audio, encoder.sampling_rate)
        encoded_audio = encoder(audio, encoder.sampling_rate)
        assert encoder.check_encoded_audio(encoded_audio)

        assert encoded_audio.shape == (2, 75, 768)


if __name__ == "__main__":
    # python -m unittest tests/test_example_dasheng.py
    unittest.main()
