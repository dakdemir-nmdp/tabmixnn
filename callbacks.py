import copy
import os
from typing import Any, Dict, Iterable, List, Optional


class Callback:
    """Base callback with Keras-like hook names."""

    def set_trainer(self, trainer: Any):
        self.trainer = trainer

    # Hooks are intentionally lightweight no-ops
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        pass

    def on_validation_end(self, logs: Optional[Dict[str, Any]] = None):
        pass


class CallbackList:
    """Container that forwards hook calls to each callback."""

    def __init__(self, callbacks: Optional[Iterable[Callback]] = None):
        self.callbacks: List[Callback] = list(callbacks or [])
        self.trainer = None

    def set_trainer(self, trainer: Any):
        self.trainer = trainer
        for cb in self.callbacks:
            cb.set_trainer(trainer)

    def append(self, callback: Callback):
        self.callbacks.append(callback)
        if self.trainer is not None:
            callback.set_trainer(self.trainer)

    # Hook dispatchers
    def on_train_begin(self, logs=None):
        for cb in self.callbacks:
            cb.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for cb in self.callbacks:
            cb.on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        for cb in self.callbacks:
            cb.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        for cb in self.callbacks:
            cb.on_batch_end(batch, logs)

    def on_validation_end(self, logs=None):
        for cb in self.callbacks:
            cb.on_validation_end(logs)


class History(Callback):
    """Collect epoch-level logs."""

    def on_train_begin(self, logs=None):
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class EarlyStopping(Callback):
    """Stop training when monitored metric has stopped improving."""

    def __init__(self, monitor="val_loss", min_delta=0.0, patience=5, mode="min", restore_best_weights=True):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

    def set_trainer(self, trainer: Any):
        super().set_trainer(trainer)
        if not hasattr(trainer, "stop_training"):
            trainer.stop_training = False

    def _is_improvement(self, current):
        if self.best is None:
            return True
        if self.mode == "max":
            return current > self.best + self.min_delta
        return current < self.best - self.min_delta

    def on_train_begin(self, logs=None):
        if self.mode == "auto":
            # Heuristic: accuracy/auc -> max, else min
            if "acc" in self.monitor or "auc" in self.monitor:
                self.mode = "max"
            else:
                self.mode = "min"
        self.best = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if self._is_improvement(current):
            self.best = current
            self.wait = 0
            if self.restore_best_weights and hasattr(self, "trainer"):
                self.best_weights = copy.deepcopy(self.trainer.state_dict())
        else:
            self.wait += 1
            if self.wait > self.patience and hasattr(self, "trainer"):
                self.stopped_epoch = epoch
                self.trainer.stop_training = True

    def on_train_end(self, logs=None):
        if self.restore_best_weights and self.best_weights is not None and hasattr(self, "trainer"):
            self.trainer.load_state_dict(self.best_weights)


class ModelCheckpoint(Callback):
    """Save model weights during training."""

    def __init__(self, filepath: str, monitor="val_loss", mode="min", save_best_only: bool = True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best = None

    def set_trainer(self, trainer: Any):
        super().set_trainer(trainer)
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)

    def _is_improvement(self, current):
        if self.best is None:
            return True
        if self.mode == "max":
            return current > self.best
        return current < self.best

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None and self.save_best_only:
            return

        should_save = True
        if self.save_best_only:
            should_save = self._is_improvement(current)

        if should_save and hasattr(self, "trainer"):
            self.best = current if current is not None else self.best
            state = self.trainer.state_dict()
            # Include epoch in filename if placeholder exists
            path = self.filepath.format(epoch=epoch)
            import torch
            torch.save(state, path)


class LambdaCallback(Callback):
    """Lightweight callback that wraps simple callables."""

    def __init__(
        self,
        on_train_begin=None,
        on_train_end=None,
        on_epoch_begin=None,
        on_epoch_end=None,
        on_batch_begin=None,
        on_batch_end=None,
        on_validation_end=None,
    ):
        self._on_train_begin = on_train_begin
        self._on_train_end = on_train_end
        self._on_epoch_begin = on_epoch_begin
        self._on_epoch_end = on_epoch_end
        self._on_batch_begin = on_batch_begin
        self._on_batch_end = on_batch_end
        self._on_validation_end = on_validation_end

    def on_train_begin(self, logs=None):
        if self._on_train_begin:
            self._on_train_begin(logs)

    def on_train_end(self, logs=None):
        if self._on_train_end:
            self._on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        if self._on_epoch_begin:
            self._on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        if self._on_epoch_end:
            self._on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        if self._on_batch_begin:
            self._on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        if self._on_batch_end:
            self._on_batch_end(batch, logs)

    def on_validation_end(self, logs=None):
        if self._on_validation_end:
            self._on_validation_end(logs)
