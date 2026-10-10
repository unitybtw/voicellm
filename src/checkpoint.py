"""Publish complete native checkpoints atomically; keep the previous file on failure."""
import os
from pathlib import Path
import tempfile
import torch

def save_atomic(payload,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix=path.name+'.',suffix='.writing',dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload,temporary)
        os.replace(temporary,path)
    finally:
        Path(temporary).unlink(missing_ok=True)
