# LOGIT-STREAM

Give every answer token a birth certificate: which layer pushed it to the top.

LOGIT-STREAM takes a logit-lens style stream: the logits read out after every layer prefix. It decomposes the final logit of any token into per-layer increments, finds the birth layer where the final answer first becomes the top token, and reports how the positive increments are shared across layers.

## What it does

- Per-layer logit increments for a target token, with an exact sum check.
- Birth layer: the first layer prefix whose argmax matches the final answer.
- Positive-increment shares per layer.

## Install

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from logit_stream.stream import birth_certificate

# logits read out after each layer prefix: list of 1-D vectors
prefix_logits = [np.load(f) for f in ["lens0.npy", "lens1.npy", "lens2.npy", "lens3.npy"]]
cert = birth_certificate(prefix_logits)
print(cert["answer"], cert["birth_layer"])
print(cert["layer_increments"])
```

## How it works

Read out logits after every layer, define the layer -1 logits as zero, and difference the stream. The increments telescope exactly back to the final logit, so the sum check must always pass.

## Honest scope

- The stream must come from one shared unembedding/readout; raw logit magnitudes are not comparable across different readouts.
- Increments measure readout pressure, not ground-truth knowledge.
- Tests use synthetic logit streams, so no model download is needed.

## License

MIT
