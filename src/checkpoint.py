"""Publish complete native checkpoints atomically; keep the previous file on failure."""
import os
import json
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


def write_json_atomic(payload,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix=path.name+'.',suffix='.writing',dir=path.parent)
    try:
        with os.fdopen(descriptor,'w') as stream:
            json.dump(payload,stream,indent=2,allow_nan=False)
            stream.write('\n')
        os.replace(temporary,path)
    finally:
        Path(temporary).unlink(missing_ok=True)
