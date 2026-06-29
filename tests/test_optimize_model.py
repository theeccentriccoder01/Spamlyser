"""Tests for the model optimisation and export script."""

from pathlib import Path
from models.optimize_model import export_onnx, export_torchscript


class FakeModel:
    """Minimal stand-in for a torch.nn.Module."""

    def eval(self):
        pass

    def __call__(self, x):
        return x


def test_export_onnx_creates_file(tmp_path):
    output = tmp_path / "test.onnx"
    export_onnx(FakeModel(), output, seq_len=4, opset=14)
    assert output.exists()


def test_export_torchscript_creates_file(tmp_path):
    output = tmp_path / "test.pt"
    export_torchscript(FakeModel(), output, seq_len=4)
    assert output.exists()
