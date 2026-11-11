---
license: mit
library_name: pytorch
tags:
  - neuroscience
  - flyvis
  - direction-estimation
  - visual-tracking
  - research
---

# fw-est342 — Flyper Wolf direction estimator

fw-est342 decodes a target's camera-local direction from the activity of a frozen FlyVis visual circuit. This repository releases the **estimator only**: it does not contain FlyVis weights, a full tracking runner, or learned controller weights.

## Files and download

```text
README.md
fw-est342/
  best.pt
  manifest.yaml
  output-order.npz
```

Download the estimator artifacts with the Hugging Face CLI:

```bash
hf download mospira/fw-est \
  --revision 0563ee0400b30a34a729cbf742978a7beb1a6f87 \
  --include 'fw-est342/*' --local-dir models/fw-est
```

The command pins the [verified artifact release](https://huggingface.co/mospira/fw-est/tree/0563ee0400b30a34a729cbf742978a7beb1a6f87/fw-est342) to an immutable commit. Record that revision alongside your run.

`manifest.yaml` records the architecture, identity, and SHA256 checksums. `best.pt` contains the estimator weights, fitted input mean and scale, and the fallback direction used for near-zero predictions. `output-order.npz` specifies the exact **24,514 ordered input features** through node indices, cell types, and axial coordinates. Matching only the feature count is insufficient: the circuit outputs must match this order exactly.

| Artifact | SHA256 |
| --- | --- |
| `best.pt` | `858eaff1cbc17bf021fd66dec0509491da007f322a7471930dd83926c5d771d5` |
| `output-order.npz` | `75d0bf0199b362498b9ac440db5986fa408caad7a6a63ac3bf639c93714d1b2c` |

## Model and compatible inference

The model is a PyTorch multilayer perceptron: **24,514 → 512 → 512 → 3**, with ReLU hidden activations and **12,815,875 parameters**. Training dropout is 0.1664114783; evaluation mode disables dropout. All parameters remain frozen during inference.

Standardize circuit activity with the checkpoint's stored mean and scale, preserving float64 standardization before the float32 network input. The network predicts `(right, forward, up)`. Normalize that output to unit length before deriving direction angles; vectors with norm below `1e-10` use the checkpoint's stored unit fallback. The output vector's magnitude is not a confidence score.

The compatible upstream visual circuit is **FlyVis 1.2.0, pretrained `flow/0000/000`**, which must be obtained separately with its applicable source terms and exact artifact identity. Its parameters stay frozen while its activity evolves. The reference pipeline uses 384 × 384 images, grayscale averaging over 11 × 11 neighborhoods at 721 receptors, 100 Hz observations, two 5 ms circuit steps per observation, and one second of neutral-gray adaptation at reset. Current activity carries circuit history; fw-est342 adds no explicit lagged feature stack.

The full episode runner and loading code are in the curated **`flyper-wolf` research repository**. A published GitHub URL has not yet been established. This download alone cannot generate episodes. The reference **fw-cont24** controller is nonlearned code/configuration retained in that research repository: yaw/pitch gains 24/s, turn-rate limits ±60°/s, and zero camera action for the first 20 observations.

The estimator receives ordered circuit activity, and the controller receives its predicted direction. Neither receives simulator target truth, camera pose, trajectory seeds, future observations, rewards, or raw images directly.

## Training and evidence

fw-est342 continued from estimator 315 using supervised camera-local direction labels from simulator geometry. Frozen 315 + fw-cont24 generated 1,000 ten-second training episodes and 64 separate validation episodes, recorded at 100 Hz. Complete episodes were split to avoid mixing nearby, strongly related ticks between training and validation. The continuation used normalized-cosine loss, AdamW, batch size 128, and one training seed. Selected additional epoch 27 corresponds to lineage epoch 342.

| Measurement | Result | Scope |
| --- | ---: | --- |
| Mean episode direction error | **0.249331°** | 64 saved validation episodes; used for checkpoint selection |
| Ten-second tracking example | **87.13%** time on target | One closed-loop example with fw-cont24 |
| Continuous sixty-second example | **96.811667%** time on target | One different closed-loop example with fw-cont24 |

Time on target measures the percentage of the full episode during which the camera's forward ray intersects the sphere, including startup. The two tracking examples are demonstrations, not benchmarks or estimates of average performance. Saved-view accuracy does not establish closed-loop superiority because steering changes future visual inputs. No human-performance comparison is claimed, and these results are not a final held-out test evaluation.

Intended use is research on decoding frozen visual-circuit activity and synthetic tracking. Natural clutter, multiple targets, real cameras, and general-purpose vision are unvalidated. Low average error can hide brief severe misses. The original training activity arrays were deleted after verified export; retained statistics do not permit fresh recomputation of every prediction.

## Credits and license

This estimator release uses the **MIT** license. FlyVis artifacts are separate dependencies; preserve their applicable notices and verify their source terms independently.

[FlyVis](https://github.com/TuragaLab/flyvis) provides the connectome-constrained visual-system model, pretrained visual parameters, and circuit simulation. See Janne K. Lappalainen et al., *Connectome-constrained networks predict neural activity across the fly visual system*, Nature (2024), [paper](https://doi.org/10.1038/s41586-024-07939-3) and [software documentation](https://turagalab.github.io/flyvis/).

Flyper Wolf contributes the synthetic tracking task, image adapter, supervised direction estimator, prediction-based steering, scoring, and episode packaging. Its tracking results are separate from the upstream paper's findings.
