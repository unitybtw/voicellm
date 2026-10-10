import json
from pathlib import Path

import pytest
import torch

from src.checkpoint import write_json_atomic
from src.models.aligned_voice import AlignedConfig, AlignedVoiceModel
from src.models.voice_unified import UnifiedVoiceModel
from src.training.aligned_state import TrainingOrder, optimizer_path_for, restore_optimizer, save_training_state, step_optimizer


def test_shuffled_training_order_resumes_across_epoch_boundary():
    ids = ['clip-' + str(i) for i in range(7)]
    uninterrupted = TrainingOrder(ids, seed=83)
    first_epoch = [uninterrupted.next_index() for _ in ids]
    assert sorted(first_epoch) == list(range(len(ids)))
    uninterrupted.next_index()
    state = json.loads(json.dumps(uninterrupted.state_dict()))
    resumed = TrainingOrder(ids, seed=999, state=state)
    assert resumed.restored
    assert [resumed.next_index() for _ in range(20)] == [uninterrupted.next_index() for _ in range(20)]
    changed = TrainingOrder(ids + ['new-clip'], state=state)
    assert not changed.restored and changed.cursor == 0
    with pytest.raises(ValueError):
        TrainingOrder(ids, state={**state, 'cursor': len(ids)})
    with pytest.raises(ValueError):
        TrainingOrder(['duplicate', 'duplicate'])


def test_last_attempt_preserves_moments_and_training_order(tmp_path):
    config = AlignedConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                           intermediate_size=48, max_seq_len=128)
    model = AlignedVoiceModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    model.spectrum_out.weight.square().sum().backward()
    optimizer.step()
    model.speech_steps.fill_(7)
    order = TrainingOrder(['one', 'two', 'three'])
    order.next_index()
    checkpoint = tmp_path / 'last_attempt.pt'
    optimizer_path = tmp_path / 'last_attempt.optimizer.pt'
    save_training_state(model, optimizer, checkpoint, optimizer_path, {'kind': 'native_resume'}, order, kind='attempt')
    metadata = json.loads(Path(str(checkpoint) + '.training.json').read_text())
    assert metadata['development_attempt'] and not metadata['speech_ready']
    assert metadata['external_weights'] is False and metadata['user_audio_used'] is False
    loaded = UnifiedVoiceModel.load(checkpoint).float()
    restored = torch.optim.AdamW(loaded.parameters(), lr=.1)
    actual_path = optimizer_path_for(checkpoint, metadata, tmp_path / 'active.pt')
    restored_order = restore_optimizer(restored, checkpoint, actual_path, 7, 'cpu', .0005)
    assert restored_order == order.state_dict()
    assert restored.param_groups[0]['lr'] == .0005
    original_moment = optimizer.state[model.spectrum_out.weight]['exp_avg']
    assert torch.equal(restored.state[loaded.spectrum_out.weight]['exp_avg'], original_moment)
    with pytest.raises(ValueError, match='mismatch'):
        restore_optimizer(restored, checkpoint, actual_path, 8, 'cpu', .0005)
    checkpoint.write_bytes(checkpoint.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='mismatch'):
        restore_optimizer(restored, checkpoint, actual_path, 7, 'cpu', .0005)


def test_optimizer_lookup_preserves_legacy_paths_and_rejects_traversal(tmp_path):
    active = tmp_path / 'voice_unified.pt'
    assert optimizer_path_for(active.resolve(), {}, active) == tmp_path / 'aligned_optimizer.pt'
    candidate = tmp_path / 'voice_aligned_candidate.pt'
    assert optimizer_path_for(candidate, {}, active) == tmp_path / 'aligned_candidate_optimizer.pt'
    for name in ('../secret.pt', '/tmp/state.pt', ''):
        with pytest.raises(ValueError):
            optimizer_path_for(candidate, {'optimizer_file': name}, active)
    with pytest.raises(ValueError, match='metadata'):
        optimizer_path_for(tmp_path / 'unknown.pt', {}, active)


def test_failed_json_publication_preserves_existing_metadata(tmp_path):
    target = tmp_path / 'metadata.json'
    write_json_atomic({'step': 7}, target)
    previous = target.read_bytes()
    with pytest.raises(ValueError):
        write_json_atomic({'invalid': float('nan')}, target)
    assert target.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [target]


def test_finite_loss_with_overflowing_gradients_cannot_corrupt_model_or_counter():
    model = torch.nn.Linear(2, 1)
    model.register_buffer('speech_steps', torch.tensor(7))
    optimizer = torch.optim.AdamW(model.parameters())
    previous = model.weight.detach().clone()
    hook = model.weight.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
    loss = model(torch.ones(1, 2)).square().mean()
    assert torch.isfinite(loss)
    with pytest.raises(RuntimeError, match='non-finite'):
        step_optimizer(model, optimizer, loss)
    assert torch.equal(model.weight, previous) and int(model.speech_steps) == 7
    assert not optimizer.state
    hook.remove()


def test_interrupted_training_keeps_periodic_checkpoint_resumable(tmp_path, monkeypatch):
    from argparse import Namespace
    from scripts import train_aligned

    monkeypatch.chdir(tmp_path)
    # Controlled synthetic fixtures exercise orchestration, not speech quality.
    public = tmp_path / 'data/public/LJSpeech-1.1/wavs'
    public.mkdir(parents=True)
    rows = [{'id': 'fixture-' + str(i), 'text': 'Hello.',
             'audio': str(public / (str(i) + '.wav')), 'dataset': 'LJSpeech-1.1',
             'source': 'human_recording'} for i in range(2)]
    monkeypatch.setattr(train_aligned, 'select', lambda *args: rows)
    monkeypatch.setattr(train_aligned, 'read_audio', lambda *args: torch.zeros(1, 4096))
    metrics = {'free_mse': 1., 'mean_baseline_mse': 1., 'delta_ratio': 1.,
               'duration_error': .1, 'voiced_log_f0_mae': .1, 'voicing_bce': .1, 'log_rms_mae': .1}
    monkeypatch.setattr(train_aligned, 'evaluate', lambda *args: dict(metrics))
    config = AlignedConfig(hidden_size=32, num_layers=2, num_heads=4,
                           num_kv_heads=2, intermediate_size=48, max_seq_len=128)
    model = AlignedVoiceModel(config)
    model.speech_steps.fill_(1)
    optimizer = torch.optim.AdamW(model.parameters())
    source = tmp_path / 'models/checkpoints/voice_aligned_candidate.pt'
    save_training_state(model, optimizer, source, source.parent / 'aligned_candidate_optimizer.pt',
                        {}, TrainingOrder([r['id'] for r in rows]))

    def objective(self, *args, **kwargs):
        if int(self.speech_steps) == 101:
            raise RuntimeError('simulated interruption')
        return {'loss': self.spectrum_out.weight.square().mean()}

    monkeypatch.setattr(AlignedVoiceModel, 'speech_loss', objective)
    args = Namespace(device='cpu', checkpoint=source, soft_clock=False,
                     confident_periodic_prior=False, clips=2, min_seconds=2., max_seconds=4.5,
                     fixed_alignments=False, lr=.0001, steps=102, sanity_only=False)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        train_aligned.run(args)
    last = source.parent / 'voice_aligned_last_attempt.pt'
    metadata = json.loads(Path(str(last) + '.training.json').read_text())
    restored = UnifiedVoiceModel.load(last).float()
    recovered = torch.optim.AdamW(restored.parameters())
    state = restore_optimizer(recovered, last, optimizer_path_for(last, metadata, source),
                              int(restored.speech_steps), 'cpu', .0001)
    assert int(restored.speech_steps) == 101
    assert state['epoch'] == 50 and state['cursor'] == 0
    assert metadata['development_attempt'] and not metadata['speech_ready']
