from pathlib import Path

import torch

from nanomoe.train import Checkpointer, read_tracker


def _model_and_optimizer() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer


def test_overwrite_latest_reuses_checkpoint_path(tmp_path: Path) -> None:
    model, optimizer = _model_and_optimizer()
    checkpointer = Checkpointer(str(tmp_path), keep_last=0, async_io=False, overwrite_latest=True)

    path1 = Path(checkpointer.save(step=1, model=model, optimizer=optimizer, tokens_seen=10))

    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)

    path2 = Path(checkpointer.save(step=2, model=model, optimizer=optimizer, tokens_seen=20))

    assert path1 == tmp_path.resolve() / "latest" / "checkpoint.pt"
    assert path2 == path1
    assert list(tmp_path.glob("step_*")) == []
    assert read_tracker(tmp_path) == 2
    assert checkpointer.find_latest() == (2, str(path1))

    restored_model, restored_optimizer = _model_and_optimizer()
    restored_checkpointer = Checkpointer(str(tmp_path), keep_last=0, async_io=False, overwrite_latest=True)

    step, tokens_seen = restored_checkpointer.load(restored_model, restored_optimizer)

    assert step == 2
    assert tokens_seen == 20
