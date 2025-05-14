import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from transformers import (
    AutoModelForCausalLM
)
import torch
model_name="HuggingFaceH4/zephyr-7b-beta"
model = AutoModelForCausalLM.from_pretrained(
      model_name, 
      device_map={"": 0}, 
      torch_dtype=torch.bfloat16)