"""Verified optimizer recovery and resumable public-data ordering."""
import hashlib
import json
import random
from pathlib import Path

import torch

from src.checkpoint import save_atomic, write_json_atomic
from src.data.video_audio import file_sha256


class TrainingOrder:
    def __init__(self, record_ids, seed=103, state=None):
        if not record_ids or len(set(record_ids)) != len(record_ids):
            raise ValueError('Unique nonempty training record IDs required')
        self.count = len(record_ids)
        self.signature = hashlib.sha256(json.dumps(record_ids).encode()).hexdigest()
        self.restored = bool(state and state.get('record_signature') == self.signature)
        if self.restored:
            values = [state.get(key) for key in ('seed', 'epoch', 'cursor')]
            if any(type(value) != int or value < 0 for value in values) or values[2] >= self.count:
                raise ValueError('Invalid saved training order')
            self.seed, self.epoch, self.cursor = values
        else:
            if type(seed) != int or seed < 0:
                raise ValueError('Nonnegative integer training seed required')
            self.seed, self.epoch, self.cursor = seed, 0, 0
        self._shuffle()

    def _shuffle(self):
        self.indices = list(range(self.count))
        random.Random(self.seed + self.epoch).shuffle(self.indices)

    def next_index(self):
        index = self.indices[self.cursor]
        self.cursor += 1
        if self.cursor == self.count:
            self.epoch += 1
            self.cursor = 0
            self._shuffle()
        return index

    def state_dict(self):
        return {'record_signature': self.signature, 'seed': self.seed,
                'epoch': self.epoch, 'cursor': self.cursor}


def optimizer_path_for(checkpoint, metadata, active):
    checkpoint = Path(checkpoint)
    name = metadata.get('optimizer_file')
    if name is not None:
        if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Optimizer metadata must name a sibling file')
        return checkpoint.parent / name
    # Compatibility with existing verified local checkpoints.
    if checkpoint.resolve() == Path(active).resolve():
        return checkpoint.parent / 'aligned_optimizer.pt'
    if checkpoint.name == 'voice_aligned_candidate.pt':
        return checkpoint.parent / 'aligned_candidate_optimizer.pt'
    raise ValueError('Checkpoint has no optimizer metadata; it cannot be resumed safely')


def restore_optimizer(optimizer, checkpoint, path, step, device, lr):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('checkpoint_sha256') != file_sha256(checkpoint) or saved.get('step') != step:
        raise ValueError('Aligned optimizer mismatch')
    optimizer.load_state_dict(saved['optimizer'])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and key != 'step':
                state[key] = value.to(device)
    for group in optimizer.param_groups:
        group['lr'] = lr
    return saved.get('training_order')


def save_training_state(model, optimizer, checkpoint, optimizer_path, transfer, order, kind='candidate'):
    checkpoint, optimizer_path = Path(checkpoint), Path(optimizer_path)
    if checkpoint.parent.resolve() != optimizer_path.parent.resolve():
        raise ValueError('Checkpoint and optimizer must share a directory')
    model.save(checkpoint, fp16=True)
    checksum = file_sha256(checkpoint)
    step = int(model.speech_steps)
    save_atomic({'optimizer': optimizer.state_dict(), 'step': step,
                 'checkpoint_sha256': checksum, 'training_order': order.state_dict()}, optimizer_path)
    # Metadata is published last; a partial save fails checksum verification on resume.
    write_json_atomic({'model_kind': model.model_kind, 'checkpoint_sha256': checksum,
                       'speech_steps': step, 'external_weights': False, 'user_audio_used': False,
                       'development_candidate': kind == 'candidate', 'development_attempt': kind == 'attempt',
                       'speech_ready': False, 'intelligibility_verified': False,
                       'optimizer_file': optimizer_path.name, 'transfer': transfer},
                      Path(str(checkpoint) + '.training.json'))
    return checksum


def step_optimizer(model, optimizer, loss):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite aligned objective')
    loss.backward()
    # A finite scalar loss can still produce overflowing gradients. Do not
    # update weights, moments, or the successful-step counter in that case.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    model.speech_steps.add_(1)
