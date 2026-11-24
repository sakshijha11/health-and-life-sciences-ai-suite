# Models

This folder is filled by the setup step and is not committed:

```powershell
python -m screening_tools setup
```

| File | Content |
|---|---|
| `hit_mlp.xml`, `hit_mlp.bin` | Hit model (OpenVINO IR, FP32 weights) |
| `hit_scaler.json` | Feature standardisation for the hit model |
| `anomaly_iforest.joblib` | Anomaly model (scikit-learn Isolation Forest) |
| `anomaly_scaler.json` | Feature standardisation for the anomaly model |
| `manifest.json` | Training settings, test metrics, versions and a SHA-256 hash of every file |

The app checks every hash in `manifest.json` before loading a model. If a file
was changed or the feature definitions changed, it stops and asks you to run
setup again. Only load model files that you built yourself: the anomaly model
is a Python pickle (joblib), which can run code when it is loaded.
