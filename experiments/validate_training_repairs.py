"""Six synthetic training sanity checks, independent of benchmark outcomes."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from tabmixnn import TabMixNN
from tabmixnn.utils.callbacks import Callback


class SanityTrace(Callback):
    def on_train_begin(self, logs=None):
        self.initial = {name: p.detach().clone() for name, p in self.trainer.named_parameters()
                        if p.requires_grad}
        self.finite_gradients = True
        self.gradient_norms = []

    def on_batch_end(self, batch, logs=None):
        parameters = [p for p in self.trainer.parameters() if p.requires_grad]
        self.finite_gradients &= all(p.grad is not None and torch.isfinite(p.grad).all().item()
                                    for p in parameters)
        self.gradient_norms.append(float(sum(p.grad.square().sum() for p in parameters).sqrt()))

    def on_train_end(self, logs=None):
        self.moved = {name: not torch.equal(p.detach(), self.initial[name])
                      for name, p in self.trainer.named_parameters() if name in self.initial}


def fit(frame, epochs, callback=None):
    return TabMixNN('y ~ x', family='gaussian', hidden_dims=[16], embedding_dim=4,
                    weight_decay=0, backbone_dropout=0, lr=.01, batch_size=len(frame),
                    epochs=epochs, random_state=19).fit(frame, callbacks=[callback] if callback else [])


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(19)
    x = rng.uniform(-2, 2, 384)
    frame = pd.DataFrame({'x': x, 'y': 10 + 2*x + .3*np.sin(3*x)})
    train, test = frame.iloc[:256].copy(), frame.iloc[256:].copy()
    tiny = train.iloc[:16].copy()
    trace = SanityTrace()
    one = fit(tiny, 500, trace)
    prediction = one.predict(tiny)
    tiny_rmse = float(np.sqrt(np.mean((tiny.y.to_numpy()-prediction)**2)))
    # The old zero-centered initialization is ~20 nats away on this data.
    mean_only_nll = float(.5*np.log(2*np.pi) + np.log(tiny.y.std(ddof=0)) + .5)
    clean = fit(train, 300)
    shuffled = train.copy()
    shuffled['y'] = rng.permutation(shuffled.y.to_numpy())
    noise = fit(shuffled, 300)

    def r2(m):
        return float(1 - np.sum((test.y.to_numpy()-m.predict(test))**2)
                     / np.sum((test.y-test.y.mean())**2))

    clean_r2, shuffled_r2 = r2(clean), r2(noise)
    gates = dict(shapes=prediction.shape == (len(tiny),),
                 initialization=abs(one.history[0]-mean_only_nll) < .2,
                 overfit_one_batch=tiny_rmse < .1,
                 finite_gradients=bool(trace.finite_gradients),
                 parameters_move=all(trace.moved.values()),
                 shuffled_labels=clean_r2 > .95 and shuffled_r2 < .1)
    result = dict(seed=19, gates=gates, initial_nll=one.history[0],
                  mean_only_nll=mean_only_nll, tiny_rmse=tiny_rmse,
                  clean_test_r2=clean_r2, shuffled_test_r2=shuffled_r2,
                  parameters_moved=trace.moved,
                  gradient_norms=trace.gradient_norms,
                  tiny_loss=one.history, clean_loss=clean.history, shuffled_loss=noise.history,
                  config=dict(hidden_dims=[16], embedding_dim=4, lr=.01, weight_decay=0,
                              tiny_epochs=500, control_epochs=300, full_batch=True))
    output = Path(__file__).parent/'results'/'training_sanity.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if not k.endswith('_loss') and k != 'gradient_norms'}, indent=2))
    if not all(gates.values()):
        raise SystemExit('A sanity gate failed; inspect the saved trace.')


if __name__ == '__main__':
    main()
