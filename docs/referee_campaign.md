# Referee campaign

The campaign repeats the paper's 8,388,608-example training setup, then evaluates
the pretrained object U-Net, the selected joint checkpoint, three short
fine-tuning controls, Richardson–Lucy and Wiener. Validation has 524,288 examples
for training/checkpoint selection; a fixed first 1,024 examples are used for the
paired comparison and RL stopping selection. All 1,210 GalSim cases are evaluated.

Prepare a new run from a clean checkout of the desired commit:

```bash
python3 workflow/referee_campaign.py prepare \
  --out "$WORK/V1H_campaigns/referee_YYYYMMDD_HHMMSS" \
  --dataset-dir "$SCRATCH/V1H_datasets/referee_YYYYMMDD_HHMMSS" \
  --store "$STORE" \
  --naco-data /path/to/naco_observation.npy \
  --reference-psf /path/to/CubeRef1068Mtot.fits
python3 workflow/referee_campaign.py submit \
  --out "$WORK/V1H_campaigns/referee_YYYYMMDD_HHMMSS" --test-only
python3 workflow/referee_campaign.py submit \
  --out "$WORK/V1H_campaigns/referee_YYYYMMDD_HHMMSS"
```

To reuse a dataset still being generated, add `--reuse-dataset /path/to/dataset`
and `--dataset-job JOB_ID` to preparation. Training waits for that producer and
a shard-completeness check. The run has its own stage-2 cache and GalSim dataset;
neither modifies the reused dataset. Without these options, the campaign creates
the full dataset using ten CPU array tasks. No array concurrency cap is applied.

The preparation command saves the exact configs, source commit/archive/hashes
and supplied scientific inputs. Submission journals each accepted job immediately
and can resume after interruption without duplicate submissions. Dry runs and
Slurm test-only previews have separate script directories. Config/source/input
changes after preparation are rejected.

## Scientific comparisons

- The three controls use PINN weights 0, 0.1 and 1 for one epoch each, with the
  same pretrained checkpoints, learning rate, seed, serial record order and
  frozen PSF/noise modules. Weight zero controls for additional supervised
  training. Each arm has a fresh optimizer and its own output directory.
- Every numerical image metric uses the same single 16-pixel border crop.
  Forward reconstruction and variance propagation use full 128×128 outputs
  before this crop. Summary tables use per-image values, not training's
  medians of batch metrics. The legacy `r2_im` field is **vnMSE**, not the
  conventional coefficient of determination.
- PSNR and SSIM use each ground-truth image's max-minus-min range, shared by all
  methods. Constant truths use `max(abs(truth), 1)` as the range. Predictions
  retain negative intensities for metrics; their pixel and absolute-flux
  fractions are reported. Nonpositive intensities are masked only in log plots.
- Coverage is calculated per image and component at 1σ and 2σ, with noise-level
  breakdowns. These are marginal, simulator-conditioned predictive intervals.
  Correlated pixels are not independent calibration trials. The predicted
  variances do not account for epistemic uncertainty or all real-data mismatch.
- RL is run for 10, 30 and 100 iterations. One global setting is selected using
  validation median object vnMSE, without selecting against GalSim truth. RL
  clips negative observation pixels and small negative numerical PSF samples to
  zero, normalizes its PSF to unit sum, and disables output clipping. This is an
  explicitly approximate treatment of signed Gaussian observations.
- The existing Wiener comparison is an oracle: both PSF and the noise variance
  come from ground truth. Its signed numerical PSF samples are retained.
- Timing loads saved models, including the actual 59,179,876-parameter joint
  model and the object head alone. BS=1 is latency; BS=64 per-image times are
  amortized throughput. Network-only timings exclude preprocessing/transfers;
  a separate series includes input conversion and input/output transfers, still
  excluding scientific preprocessing. Larger input sizes are not supported by
  simply resizing the fixed dense PSF outputs.

For the code's noise correction `n`, reconstruction is `y = x*h - n`. This
corresponds to the manuscript's additive residual **η = −n**. Figure labels state
that the correction is subtracted; numerical targets are unchanged.

Observation variance uses one shared routine:

```text
Var(y) ≈ Var(x) * h² + x² * Var(h) + Var(x) * Var(h) + Var(n)
```

Here `*` denotes convolution. This assumes independent pixels/components and
neglects covariance introduced by PSF normalization. It is not a full posterior
covariance calculation. The independent-product variance term is included in
both evaluation and plotting.

## NaCo and outputs

The NaCo comparison averages the supplied observation frames once, preserves
their intensity scale, and applies both same-campaign models to that mean input.
Figures share field of view and stretch and include pixel-axis profiles. Sky
orientation is not inferred without WCS. An optional MISTRAL array must already
be registered, have the same shape and use the same intensity units. When it is
unavailable, the comparison records that fact; no published image is treated as
a calibrated scientific array.

A supplied reference-star cube is averaged over its recorded frame axis. The
PSF check peak-centers profiles and normalizes a common angular aperture. Its
optional outer-annulus median background subtraction is recorded together with
the original cube, derived images, pixel scale and source hashes. Calibration
mismatch remains possible; the star is an external check, not target PSF truth.

`referee_evaluation/` contains per-image metrics, selected full-field examples
and manifests. Streaming avoids collecting the full validation cube in RAM.
`referee_report/` contains median/IQR tables, paired differences, coverage,
validation-selected RL results and comparison figures. `naco_comparison/`
contains raw predictions, FITS, profiles and provenance.

The final job uses `archive`, `nab@cpu`, `qos_cpu-t3`, waits for every writer with
`afterany`, and writes `$STORE/<run-name>.tar` plus checksum/receipt. The entire
top-level `dataset/` is omitted and symlinks are not followed; bulk training
caches live under that dataset tree. Models, metrics, figures, logs, scientific
observation inputs and source provenance are retained. Failed dependent jobs
are cancelled; the report still runs to record missing products before archive.

Literature, manuscript notation, observational claims and the deferred MISTRAL
comparison still require author revision; no numerical result is inserted into
the paper before the campaign finishes.

## Validation

Run `python3 -m pytest -q tests`. The numerical and checkpoint roundtrip tests
require TensorFlow and scikit-image. On Jean Zay CPU nodes, load
`tensorflow-gpu/py3/2.16.1` and `git/2.39.1` first: the campaign-controller tests
exercise source provenance and need Git, which is absent from the default
compute-node PATH. Production preparation/submission run on a login node.
