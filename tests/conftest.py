import models.model_init
import sys
from unittest.mock import MagicMock

# 1. Always mock heavy / optional modules so imports succeed quickly.
#    These modules take too long to import for real in test fixtures.
_ALWAYS_MOCKED = [
    "torch",
    "torch.cuda",
    "transformers",
    "streamlit",
    "lime",
    "lime.lime_text",
    "plotly",
    "plotly.express",
    "plotly.graph_objects",
    "sklearn",
    "sentencepiece",
    "datasets",
]
for _mod in _ALWAYS_MOCKED:
    sys.modules.setdefault(_mod, MagicMock())

sys.modules["torch"].cuda.is_available.return_value = False

# 2. Conditionally mock fpdf only when it is not installed (so PDF tests
#    that require it can skip via pytest.importorskip).
if "fpdf" not in sys.modules:
    try:
        __import__("fpdf")
    except ImportError:
        sys.modules["fpdf"] = MagicMock()

# 3. Setup a default mock pipeline returning structured scores for tests
mock_pipeline = MagicMock()
mock_pipeline.return_value = [{"label": "SPAM", "score": 0.95}]
sys.modules["transformers"].pipeline.return_value = mock_pipeline

# 4. Prevent models/model_init.py verification from calling Hugging Face on import

models.model_init.MODEL_STATUS = True
models.model_init.MODEL_ERROR_MESSAGE = ""
models.model_init.MODEL_WARNINGS = []
models.model_init.verify_model_availability = MagicMock(
    return_value=(True, "", [])
)

sys.modules["torch"].cuda.is_available.return_value = False

# Setup a default mock pipeline returning structured scores for tests
mock_pipeline = MagicMock()
mock_pipeline.return_value = [{"label": "SPAM", "score": 0.95}]
sys.modules["transformers"].pipeline.return_value = mock_pipeline

# 2. Prevent models/model_init.py verification from calling Hugging Face on import

models.model_init.MODEL_STATUS = True
models.model_init.MODEL_ERROR_MESSAGE = ""
models.model_init.MODEL_WARNINGS = []
models.model_init.verify_model_availability = MagicMock(
    return_value=(True, "", [])
)
