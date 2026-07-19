import torch

def quantize_model_dynamic(model):
    """Applies dynamic INT8 quantization to a PyTorch model to optimize CPU inference."""
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return quantized_model
    except Exception:
        return model
