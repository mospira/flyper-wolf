# Evidence notes

These are summaries of recorded research and compatibility checks, not new runs.
The [original machine receipts](https://github.com/mospira/flyper-wolf/tree/6aee8115931f0c7416064ac053f27683d0d5ca56/docs/evidence)
remain available in that fixed repository snapshot. Current artifact identities
belong in the [release manifest](../../releases/fw-est342.json).

## Saved-view validation

fw-est342 continued from estimator 315 using 1,000 ten-second training episodes
and 64 separate validation episodes, collected by frozen 315 + fw-cont24 at 100 Hz.
The saved validation views contained 62,720 eligible observations.

| Estimator | Mean episode bearing error |
| --- | ---: |
| Initialization 315 | 1.195592° |
| Selected fw-est342 | 0.249331° |
| Final epoch 343 | 0.257250° |

For fw-est342, episode means had median 0.197808° and range 0.128643–0.812589°.
The report also recorded 63/64 wins against online 315 receipts; this is not a
fresh matched closed-loop comparison. Validation was used to select the checkpoint,
and only one training seed was used. These are transcribed report statistics:
the original training/validation activity arrays were deleted after export, preventing a fresh
prediction recomputation. This is selection-used validation, not final-test evidence.

## Earlier matched comparison

Estimator 277 and estimator 315 each ran the same 16 ten-second scenarios with
fw-cont24: 32 closed-loop episodes and 32,000 circuit ticks in total.

| Metric | Estimator 277 | Estimator 315 |
| --- | ---: | ---: |
| Mean time on target | 58.679375% | 82.758125% |
| Mean aiming error | 6.117510° | 3.028146° |
| Fraction of time over 45° error | 0.0064125 | 0.0079625 |

Time on target improved in 15/16 pairs. The mean paired gain was 24.07875
percentage points; its pointwise 95% percentile-bootstrap interval was
[17.81246875, 30.14196875], resampling matched episodes 20,000 times with seed 74.
Both arms acquired the target in every episode, but only 7/16 cases acquired
sooner with 315. This preliminary replacement check did not evaluate fw-est342,
use the final test, change the controller, or establish a deployment promotion.
The [unchanged per-case CSV](277-vs315-per-case.csv) retains the detailed metrics.

## Standalone runner

A fresh independent environment, with no development `fly_aim` package installed,
ran the locally exported fw-est342 + fw-cont24 bundle on the ten-second watch case
(target/camera seeds 302000001/303000001). It reproduced **87.13% time on target**,
identical metrics, and all 1,000 numeric trace rows with maximum absolute difference
0. The video contained 500 frames at 50 fps. Model parameters remained frozen.

This was Windows 11, Python 3.12.13, PyTorch 2.14.0+cu130, FlyVis 1.2.0,
and an RTX 4070, using deterministic algorithms with TF32 disabled. Cold
initialization plus the run took 318.09 seconds. Exact agreement is scoped to
that setup and known case; it does not establish cross-hardware equality or
closed-loop parity for a freshly downloaded bundle. See the executable
[reference configuration](../../configs/reference-fw-est342-fw-cont24.json).

## Neural recording

Enabling recording on that reference case left metrics and trace byte-identical
to the recording-disabled run, with frozen parameters unchanged. It captured
1,000 ticks at 10 ms intervals across 45,669 units, their neutral-gray-adapted
baseline, cell types/hexagonal positions, and 721 receptor inputs per tick.
The saved graph contained 12,000 edges sampled uniformly without replacement
with seed 74, sorted by original edge index; it was a subset of the real graph.
The source recording manifest SHA-256 was
`cb79f333a3e5c552b5fc5a852a003add6dc9c08330474226242c76310d2f511e`.

## Estimator download

A fresh Hugging Face CLI download from [mospira/fw-est at its pinned artifact revision](https://huggingface.co/mospira/fw-est/tree/0563ee0400b30a34a729cbf742978a7beb1a6f87)
matched all three estimator files by SHA-256 and size. The CPU loader validated
the manifest and exactly matched local-checkpoint predictions on three seeded
activity rows (seed 74). An assembled bundle using that download and the existing
local exact frozen FlyVis circuit passed `--dry-run` configuration validation.
No fresh closed-loop episode used the downloaded bundle. FlyVis is obtained
separately; these checks establish estimator-only download/prediction parity.

## Extraction checks

Recorded component checks passed exact equality for nine CPU predictor outputs
(three synthetic 24,514-feature activity rows, seed 7), five proportional-controller
bearings, 721 preprocessed receptor values (one seeded RGB image, seed 9), and
simulation positions/midpoint scores over four action intervals. Four focused
inference-contract tests checked checkpoint identity/freeze/output order, the
circuit-only estimator interface, startup and causal scheduling/cleanup, and
geometric scoring/preprocessing/fallback. Loop spies were confined to tests;
the runner has no mock or oracle mode. These component checks ran no actual
FlyVis episode and are separate from the standalone run above.

## Provenance

The extraction source was [mospira/fly-aim at 28e97cd8f1e7c4be0f8fa2c472eb88db79dd90b5](https://github.com/mospira/fly-aim/tree/28e97cd8f1e7c4be0f8fa2c472eb88db79dd90b5).
Historical report names identify evidence, not runtime dependencies:

- `experiments/estimator-mlp315-plain24-1000-v1-results.md` — continuation validation.
- `experiments/estimator-mlp315-plain24-check-v1-results.md` — 277-versus-315 comparison.
- `experiments/proportional-mlp342-plain24-watch-v1.md` — ten-second reference case.
- `experiments/proportional-mlp342-plain24-watch-60s-v1.md` — separate minute example.
- `configs/controllers/proportional-mlp315-plain24-v1.yaml` — source controller settings.
