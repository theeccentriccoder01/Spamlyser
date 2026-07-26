import torch
from models.quantizer import quantize_model_dynamic

def test_quantize_model_dynamic():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 2)
        def forward(self, x):
            return self.fc(x)
            
    model = SimpleModel()
    q_model = quantize_model_dynamic(model)
    assert q_model is not None
