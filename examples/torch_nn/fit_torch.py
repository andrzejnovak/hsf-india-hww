#!/usr/bin/env python3
"""The same fit, the network evaluated with torch at run time instead of numpy: the route for a model that is
not a plain multilayer perceptron (a transformer over the event's objects, say), whose forward pass you would
rather not rewrite by hand.

    python3 fit_torch.py <dataset_dir> <output_dir>      (point run.sh here instead of fit.py)

model_state.json is the trained network's state_dict, every tensor as nested lists (train.py writes it beside
model.json): the same nn.Sequential is built again, the state_dict loaded back, the events scored under
torch.no_grad on one thread, the same answer on every machine and fast enough for 17k events. Everything else,
templates, fit, errors, significance, is fit.py's, imported.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402
from train import FEATURES, features  # noqa: E402

torch.set_num_threads(1)


def load_model(path):
    """The network of train.py with its trained weights, from model_state.json beside model.json."""
    m = json.loads(Path(path).with_name("model_state.json").read_text())
    net = torch.nn.Sequential(torch.nn.Linear(len(FEATURES), 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 16), torch.nn.ReLU(),
                              torch.nn.Linear(16, 1))
    net.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in m["state_dict"].items()})
    net.eval()
    return {"mean": np.array(m["mean"]), "std": np.array(m["std"]), "net": net}


def score(model, ev):
    x = torch.from_numpy(((features(ev) - model["mean"]) / model["std"]).astype(np.float32))
    with torch.no_grad():
        return torch.sigmoid(model["net"](x)[:, 0]).numpy().astype(float)


F.load_model, F.score = load_model, score   # fit.py's main() looks both up when it runs

if __name__ == "__main__":
    F.main(Path(sys.argv[1]), Path(sys.argv[2]))
