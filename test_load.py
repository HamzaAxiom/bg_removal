import torch
from transformers import pipeline
import sys

try:
    from transformers import AutoModelForImageSegmentation, AutoProcessor
    model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
    print("Success loading model and processor manually")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
