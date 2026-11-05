# Cloud renderer comparison: eval

| Item | Value |
|---|---|
| GPU | Apple M1 |
| OS | Version 27.2 (Build 26B123) |
| Renderer version | nogit |
| Experiment | `experiments/eval.json` sha256 `d74a6b497c10da76f1b272bf8ad68156c5a7842c44019b019fa2502207a8cb65` |
| Timing | 30 warm-up + 300 frames x 3 runs; GPU time from MTL4CommitFeedback; headline = median of run medians; p5/p95 nearest-rank |
| Validation layer | off |
| Reference gates | est. relMSE <= 0.0001 and <= 0.1 x min method relMSE; spp per half 1024..65536 |

Metrics are against the path-traced reference (relMSE on linear radiance; PSNR and SSIM on display-mapped images). Methods run at their recorded configurations, which differ in quality settings (column *Quality*). *Equal-quality pt spp* is the first power-of-two spp at which a progressive path tracer (seed 3) reaches the method's relMSE; its GPU time is the cumulative GPU time of that run.

## eval_thin / side

Reference: 65536 spp per half, est. relMSE 2.583e-07, accepted, bounce-cap hits 0, mean bounces 0.008042. ![reference](images/eval_thin_side_reference.png)

| Method | Quality | GPU ms median [p5, p95] | CPU prep ms | relMSE | PSNR dB | SSIM | Renderer MB | Equal-quality pt spp (GPU ms) |
|---|---|---|---|---|---|---|---|---|
| analytic | 128/16 steps, 4 oct, ambient | 8.844 [6.719, 11.9] | 0.06658 | 0.001865 | 37.27 | 0.9905 | 7.97 | 32 (25.56) |
| neural | 128/16 steps | 83.88 [47.78, 121.7] | 0.09746 | 3.635e-06 | 62.57 | 0.9999 | 7.976 | > 4096 |
| pt_bench | 64 spp | 0.68 [0.6701, 0.716] | 0.03617 | 0.0005109 | 39.2 | 0.9615 | 7.97 | - |

- analytic: [image](images/eval_thin_side_analytic.png), [abs error](images/eval_thin_side_analytic_abs.png), [rel error](images/eval_thin_side_analytic_rel.png)
- neural: [image](images/eval_thin_side_neural.png), [abs error](images/eval_thin_side_neural_abs.png), [rel error](images/eval_thin_side_neural_rel.png)
- pt_bench: [image](images/eval_thin_side_pt_bench.png), [abs error](images/eval_thin_side_pt_bench_abs.png), [rel error](images/eval_thin_side_pt_bench_rel.png)

## eval_thin / backlit

Reference: 65536 spp per half, est. relMSE 7.887e-07, accepted, bounce-cap hits 0, mean bounces 0.007906. ![reference](images/eval_thin_backlit_reference.png)

| Method | Quality | GPU ms median [p5, p95] | CPU prep ms | relMSE | PSNR dB | SSIM | Renderer MB | Equal-quality pt spp (GPU ms) |
|---|---|---|---|---|---|---|---|---|
| analytic | 128/16 steps, 4 oct, ambient | 7.131 [5.469, 9.517] | 0.06488 | 0.005208 | 36.17 | 0.995 | 7.97 | 32 (27.55) |
| neural | 128/16 steps | 80.34 [57.04, 101] | 0.09708 | 1.409e-05 | 60.02 | 0.9999 | 7.976 | > 4096 |
| pt_bench | 64 spp | 0.6713 [0.6631, 0.7176] | 0.03242 | 0.001723 | 37.73 | 0.9591 | 7.97 | - |

- analytic: [image](images/eval_thin_backlit_analytic.png), [abs error](images/eval_thin_backlit_analytic_abs.png), [rel error](images/eval_thin_backlit_analytic_rel.png)
- neural: [image](images/eval_thin_backlit_neural.png), [abs error](images/eval_thin_backlit_neural_abs.png), [rel error](images/eval_thin_backlit_neural_rel.png)
- pt_bench: [image](images/eval_thin_backlit_pt_bench.png), [abs error](images/eval_thin_backlit_pt_bench_abs.png), [rel error](images/eval_thin_backlit_pt_bench_rel.png)

## eval_medium / side

Reference: 16384 spp per half, est. relMSE 3.25e-06, accepted, bounce-cap hits 0, mean bounces 0.02984. ![reference](images/eval_medium_side_reference.png)

| Method | Quality | GPU ms median [p5, p95] | CPU prep ms | relMSE | PSNR dB | SSIM | Renderer MB | Equal-quality pt spp (GPU ms) |
|---|---|---|---|---|---|---|---|---|
| analytic | 128/16 steps, 4 oct, ambient | 10.26 [6.742, 15.73] | 0.07117 | 0.005706 | 35.46 | 0.9936 | 7.97 | 32 (53.92) |
| neural | 128/16 steps | 90.89 [47.88, 135.4] | 0.09688 | 5.033e-05 | 54.61 | 0.9997 | 7.976 | 4096 (5895) |
| pt_bench | 64 spp | 1.441 [1.421, 1.469] | 0.04283 | 0.001648 | 38.66 | 0.9619 | 7.97 | - |

- analytic: [image](images/eval_medium_side_analytic.png), [abs error](images/eval_medium_side_analytic_abs.png), [rel error](images/eval_medium_side_analytic_rel.png)
- neural: [image](images/eval_medium_side_neural.png), [abs error](images/eval_medium_side_neural_abs.png), [rel error](images/eval_medium_side_neural_rel.png)
- pt_bench: [image](images/eval_medium_side_pt_bench.png), [abs error](images/eval_medium_side_pt_bench_abs.png), [rel error](images/eval_medium_side_pt_bench_rel.png)

## eval_medium / backlit

Reference: 8192 spp per half, est. relMSE 1.331e-05, accepted, bounce-cap hits 0, mean bounces 0.0281. ![reference](images/eval_medium_backlit_reference.png)

| Method | Quality | GPU ms median [p5, p95] | CPU prep ms | relMSE | PSNR dB | SSIM | Renderer MB | Equal-quality pt spp (GPU ms) |
|---|---|---|---|---|---|---|---|---|
| analytic | 128/16 steps, 4 oct, ambient | 8.209 [5.521, 12.2] | 0.06125 | 0.00746 | 35.83 | 0.9959 | 7.97 | 32 (62.04) |
| neural | 128/16 steps | 81.82 [44.36, 126.7] | 0.09863 | 0.0001781 | 50.92 | 0.9994 | 7.976 | 2048 (3240) |
| pt_bench | 64 spp | 1.579 [1.557, 1.607] | 0.04838 | 0.003426 | 37.21 | 0.9582 | 7.97 | - |

- analytic: [image](images/eval_medium_backlit_analytic.png), [abs error](images/eval_medium_backlit_analytic_abs.png), [rel error](images/eval_medium_backlit_analytic_rel.png)
- neural: [image](images/eval_medium_backlit_neural.png), [abs error](images/eval_medium_backlit_neural_abs.png), [rel error](images/eval_medium_backlit_neural_rel.png)
- pt_bench: [image](images/eval_medium_backlit_pt_bench.png), [abs error](images/eval_medium_backlit_pt_bench_abs.png), [rel error](images/eval_medium_backlit_pt_bench_rel.png)

## eval_thick / side

Reference: 4096 spp per half, est. relMSE 1.772e-05, accepted, bounce-cap hits 0, mean bounces 0.05997. ![reference](images/eval_thick_side_reference.png)

| Method | Quality | GPU ms median [p5, p95] | CPU prep ms | relMSE | PSNR dB | SSIM | Renderer MB | Equal-quality pt spp (GPU ms) |
|---|---|---|---|---|---|---|---|---|
| analytic | 128/16 steps, 4 oct, ambient | 8.48 [6.066, 12.48] | 0.06304 | 0.003782 | 37.94 | 0.9947 | 7.97 | 64 (195.6) |
| neural | 128/16 steps | 71.89 [42.49, 105.5] | 0.1013 | 0.0002179 | 50.44 | 0.999 | 7.976 | 1024 (2980) |
| pt_bench | 64 spp | 2.903 [2.842, 2.991] | 0.05879 | 0.00222 | 38.94 | 0.9649 | 7.97 | - |

- analytic: [image](images/eval_thick_side_analytic.png), [abs error](images/eval_thick_side_analytic_abs.png), [rel error](images/eval_thick_side_analytic_rel.png)
- neural: [image](images/eval_thick_side_neural.png), [abs error](images/eval_thick_side_neural_abs.png), [rel error](images/eval_thick_side_neural_rel.png)
- pt_bench: [image](images/eval_thick_side_pt_bench.png), [abs error](images/eval_thick_side_pt_bench_abs.png), [rel error](images/eval_thick_side_pt_bench_rel.png)

## eval_thick / backlit

Reference: 8192 spp per half, est. relMSE 1.751e-05, accepted, bounce-cap hits 0, mean bounces 0.05397. ![reference](images/eval_thick_backlit_reference.png)

| Method | Quality | GPU ms median [p5, p95] | CPU prep ms | relMSE | PSNR dB | SSIM | Renderer MB | Equal-quality pt spp (GPU ms) |
|---|---|---|---|---|---|---|---|---|
| analytic | 128/16 steps, 4 oct, ambient | 7.335 [5.664, 10.36] | 0.06354 | 0.004709 | 37.67 | 0.996 | 7.97 | 64 (205.6) |
| neural | 128/16 steps | 71.55 [50.69, 94.8] | 0.1001 | 0.0003118 | 48.73 | 0.9991 | 7.976 | 1024 (3116) |
| pt_bench | 64 spp | 3.026 [2.974, 3.092] | 0.06375 | 0.004397 | 36.85 | 0.9612 | 7.97 | - |

- analytic: [image](images/eval_thick_backlit_analytic.png), [abs error](images/eval_thick_backlit_analytic_abs.png), [rel error](images/eval_thick_backlit_analytic_rel.png)
- neural: [image](images/eval_thick_backlit_neural.png), [abs error](images/eval_thick_backlit_neural_abs.png), [rel error](images/eval_thick_backlit_neural_rel.png)
- pt_bench: [image](images/eval_thick_backlit_pt_bench.png), [abs error](images/eval_thick_backlit_pt_bench_abs.png), [rel error](images/eval_thick_backlit_pt_bench_rel.png)

## Excluded costs (not part of steady-state timing)

| Scene | Camera | Method | Compile ms | Bake ms | Model load ms | Reference GPU ms | Memory density MB | Device delta MB |
|---|---|---|---|---|---|---|---|---|
| eval_thin | side | analytic | 0.3577 | 64.31 | n/a | 8.79e+04 | 24 | 7.969 |
| eval_thin | side | neural | 0.8285 | 64.31 | 0.6165 | 8.79e+04 | 24 | 7.969 |
| eval_thin | side | pt_bench | 1.016 | 64.31 | n/a | 8.79e+04 | 24 | 7.969 |
| eval_thin | backlit | analytic | 0.2436 | 64.31 | n/a | 8.91e+04 | 24 | 7.969 |
| eval_thin | backlit | neural | 1.276 | 64.31 | 0.852 | 8.91e+04 | 24 | 7.969 |
| eval_thin | backlit | pt_bench | 0.8182 | 64.31 | n/a | 8.91e+04 | 24 | 7.969 |
| eval_medium | side | analytic | 0.3186 | 65.68 | n/a | 4.724e+04 | 24 | 7.969 |
| eval_medium | side | neural | 1.165 | 65.68 | 0.8461 | 4.724e+04 | 24 | 8.219 |
| eval_medium | side | pt_bench | 0.774 | 65.68 | n/a | 4.724e+04 | 24 | 7.969 |
| eval_medium | backlit | analytic | 0.2402 | 65.68 | n/a | 2.588e+04 | 24 | 7.969 |
| eval_medium | backlit | neural | 1.221 | 65.68 | 0.9306 | 2.588e+04 | 24 | 7.969 |
| eval_medium | backlit | pt_bench | 0.8287 | 65.68 | n/a | 2.588e+04 | 24 | 7.969 |
| eval_thick | side | analytic | 0.3053 | 65.16 | n/a | 2.379e+04 | 24 | 7.969 |
| eval_thick | side | neural | 1.551 | 65.16 | 1.341 | 2.379e+04 | 24 | 7.969 |
| eval_thick | side | pt_bench | 1.031 | 65.16 | n/a | 2.379e+04 | 24 | 7.969 |
| eval_thick | backlit | analytic | 0.2376 | 65.16 | n/a | 4.975e+04 | 24 | 7.969 |
| eval_thick | backlit | neural | 1.222 | 65.16 | 1.192 | 4.975e+04 | 24 | 7.969 |
| eval_thick | backlit | pt_bench | 0.6911 | 65.16 | n/a | 4.975e+04 | 24 | 7.969 |

Compile times are wall time of MTL4Compiler calls and include Metal's shader cache hits.

## Bias sources

- pt: bounce cap 4096 (hits listed per reference); low-spp pt methods carry Monte Carlo noise, not bias.
- analytic: ray-march discretization, two-direction ambient term, octave multiple-scattering approximation (not energy-conserving).
- neural: ray-march discretization, Monte Carlo noise in training targets (log-space loss bias), fp16 weights.

## Neural model

- `models/cloud_mlp.cnrm` sha256 `e3f8036c1732a9c3ae051f91efa221c4ab9c99b4f8ad1d3eea7ad4ae1c2c8c6d`, format 1
- training density seeds 1000-1999 (evaluation scenes use seeds 1-999)
- best validation loss (MSE on log1p f) 0.0002521, validation seeds 2000-2999
