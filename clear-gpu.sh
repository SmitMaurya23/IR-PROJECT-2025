alias clear-gpu='python - <<PY
import torch, gc
gc.collect(); torch.cuda.empty_cache(); print("GPU cache cleared ✅")
PY'
