# 🚀 Deployment Guide

Deploy your trained SLT model to production using Docker, ONNX, or FastAPI.

## Option 1: Docker Container

### Build Image

```bash
docker build -t slt-hybrid-ssl:latest .
```

### Run Container

```bash
docker run --gpus all -it \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    slt-hybrid-ssl:latest
```

### Using Docker Compose

```bash
docker-compose up -d
docker-compose exec slt bash
```

## Option 2: FastAPI Web Service

### Create `app/main.py`:

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, '..')
from src.inference import SignLanguageTranslator

app = FastAPI(title="SLT API")

translator = SignLanguageTranslator(
    checkpoint_path="../models/best_model.pt"
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict sign gloss from uploaded video."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = translator.predict_from_video(tmp_path)
        return JSONResponse(result)
    finally:
        Path(tmp_path).unlink()

@app.get("/health")
async def health():
    return {"status": "ok"}
```

### Run

```bash
pip install fastapi uvicorn python-multipart
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Test

```bash
curl -X POST "http://localhost:8000/predict" \
    -F "file=@path/to/video.mp4"
```

## Option 3: ONNX Export

### Convert to ONNX

```python
import torch
from src.models.slt_model import SLTModel
from src.utils.config import load_config

cfg = load_config("configs/train_config.yaml")
model = SLTModel(cfg)
model.load_state_dict(torch.load("models/best_model.pt"))
model.eval()

# Dummy inputs
frames = torch.randn(1, 16, 3, 224, 224)
keypoints = torch.randn(1, 16, 258)

# Export
torch.onnx.export(
    model,
    (frames, keypoints),
    "models/slt_model.onnx",
    input_names=['frames', 'keypoints'],
    output_names=['logits'],
    opset_version=14,
)
print("✓ Exported to models/slt_model.onnx")
```

### Use ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("models/slt_model.onnx")

frames = np.random.randn(1, 16, 3, 224, 224).astype(np.float32)
keypoints = np.random.randn(1, 16, 258).astype(np.float32)

outputs = sess.run(None, {
    'frames': frames,
    'keypoints': keypoints
})
logits = outputs[0]
print(f"Prediction: {logits.argmax()}")
```

## Option 4: TorchServe (Production)

### Create model archive

```bash
# Save model
torch.save(model.state_dict(), "models/slt_model.pt")

# Create handler
cat > models/handler.py << 'EOF'
from ts.torch_handler.base_handler import BaseHandler
import torch

class SLTHandler(BaseHandler):
    def handle(self, data):
        frames = torch.from_numpy(data[0]).float()
        keypoints = torch.from_numpy(data[1]).float()
        with torch.no_grad():
            output = self.model(frames, keypoints)
        return output
EOF

# Archive
torch-model-archiver \
    --model-name slt_model \
    --version 1.0 \
    --model-file src/models/slt_model.py \
    --serialized-file models/slt_model.pt \
    --handler models/handler.py \
    --export-path model_store
```

### Serve

```bash
torchserve --start --model-store model_store --ncs
```

## Option 5: AWS SageMaker

### Create SageMaker endpoint script

```python
import boto3
import json

sagemaker = boto3.client('sagemaker-runtime')

response = sagemaker.invoke_endpoint(
    EndpointName='slt-endpoint',
    ContentType='application/json',
    Body=json.dumps({
        'frames': frames.tolist(),
        'keypoints': keypoints.tolist()
    })
)

result = json.loads(response['Body'].read())
print(f"Prediction: {result['gloss']}")
```

## Performance Optimization

### 1. Quantization (INT8)

```python
import torch.quantization as quantization

model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
quantization.prepare_qat(model, inplace=True)
# ... train ...
quantization.convert(model, inplace=True)
torch.save(model.state_dict(), 'models/slt_quantized.pt')
```

### 2. Pruning

```python
import torch.nn.utils.prune as prune

for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.2)
```

### 3. Knowledge Distillation

Train a smaller "student" model to mimic the larger teacher:

```python
student = SLTModel(small_cfg)
teacher = SLTModel(cfg)
teacher.load_state_dict(torch.load('models/best_model.pt'))

for batch in train_loader:
    student_logits = student(...)
    teacher_logits = teacher(...)
    
    kl_loss = nn.KLDivLoss()(
        torch.log_softmax(student_logits, dim=1),
        torch.softmax(teacher_logits, dim=1)
    )
    kl_loss.backward()
```

## Monitoring

### Health Check Endpoint

```python
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "slt_hybrid_ssl",
        "version": "1.0",
        "gpu": "available" if torch.cuda.is_available() else "unavailable"
    }
```

### Metrics

```python
from prometheus_client import Counter, Histogram
import time

predictions = Counter('predictions_total', 'Total predictions')
latency = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.post("/predict")
async def predict(file: UploadFile):
    start = time.time()
    result = translator.predict_from_video(...)
    latency.observe(time.time() - start)
    predictions.inc()
    return result
```

## Logging & Monitoring

### Structured Logging

```python
import logging
import json

logger = logging.getLogger("inference")
handler = logging.FileHandler("logs/predictions.jsonl")
handler.setFormatter(logging.Formatter(json.dumps({
    'timestamp': '%(asctime)s',
    'level': '%(levelname)s',
    'message': '%(message)s',
})))
logger.addHandler(handler)

logger.info(f"Prediction: {result['gloss']}, confidence: {result['confidence']}")
```

## Scaling

### Horizontal Scaling (Multiple Workers)

```bash
gunicorn -w 4 -b 0.0.0.0:8000 app.main:app
```

### Load Balancing

```bash
# With nginx
docker run -d -p 80:80 \
    -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf \
    nginx:latest
```

## Cost Optimization

| Deployment | Cost | Latency | Complexity |
|-----------|------|---------|-----------|
| Docker | $ | 50-100ms | Low |
| FastAPI | $$ | 100-200ms | Medium |
| ONNX | $ | 30-50ms | Low |
| TorchServe | $$$ | 50-100ms | High |
| SageMaker | $$$$ | 100-500ms | High |

---

**Next Steps:**
1. Choose deployment option based on requirements
2. Set up monitoring and logging
3. Implement health checks and metrics
4. Load test the endpoint
5. Set up auto-scaling rules
