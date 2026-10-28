# Comparación en test: variantes comprimidas y réplicas

Protocolo `a54fbea8ce81e3842fb0044b588ede8c9d8b6ae506027398bc1880e22b78f0b9`. 4730 pares, crop central 256. Réplicas: v1, v2.

## Calidad por tarea, PSNR en dB

| Modelo | Época | deblur | dehaze | denoise | derain | enhance | Macro | Ponderado | SSIM macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | — | 24.600 | 29.407 | 37.478 | 26.782 | 22.370 | 28.1274 | 27.2480 | 0.8507 |
| S0_sel | — | 24.644 | 29.720 | 37.534 | 26.782 | 22.004 | 28.1366 | 27.3091 | 0.8496 |
| S1_d32_v1 | 3 | 24.664 | 29.594 | 37.394 | 26.676 | 22.004 | 28.0662 | 27.2808 | 0.8495 |
| S2_d32_v1 | 4 | 24.684 | 29.650 | 37.509 | 26.768 | 22.016 | 28.1254 | 27.3249 | 0.8497 |
| S1_d32_v2 | 4 | 24.632 | 29.541 | 37.495 | 26.707 | 22.029 | 28.0809 | 27.2743 | 0.8494 |
| S2_d32_v2 | 5 | 24.665 | 29.597 | 37.722 | 26.766 | 22.019 | 28.1538 | 27.3423 | 0.8497 |

## Diferencias promediadas entre réplicas

Para cada imagen se promedia la diferencia de las réplicas y luego se remuestrea por tarea. Con dos réplicas, el intervalo todavía no representa bien la variación entre semillas: comparar también las réplicas por separado.

| Comparación | Δ PSNR macro | IC 95 % | Δ PSNR ponderado | IC 95 % | Δ SSIM macro | Imágenes que mejoran |
|---|---:|---|---:|---|---:|---:|
| S2_d32 - S1_d32 | +0.066 | [+0.049, +0.084] | +0.056 | [+0.044, +0.070] | +0.0002 | 64 % |
| S2_d32 - S0_sel | +0.003 | [-0.037, +0.043] | +0.024 | [+0.011, +0.041] | +0.0001 | 51 % |
| S1_d32 - S0_sel | -0.063 | [-0.093, -0.034] | -0.032 | [-0.037, -0.026] | -0.0001 | 39 % |
| S2_d32 - B0 | +0.012 | [-0.143, +0.166] | +0.086 | [+0.064, +0.109] | -0.0010 | 69 % |

## Diferencias por réplica

| Comparación | Δ PSNR macro | IC 95 % | Δ PSNR ponderado | IC 95 % | Δ SSIM macro | Imágenes que mejoran |
|---|---:|---|---:|---|---:|---:|
| S0_sel - B0 | +0.009 | [-0.173, +0.190] | +0.061 | [+0.048, +0.076] | -0.0011 | 66 % |
| S2_d32_v1 - S1_d32_v1 | +0.059 | [+0.036, +0.083] | +0.044 | [+0.032, +0.059] | +0.0002 | 61 % |
| S1_d32_v1 - S0_sel | -0.070 | [-0.099, -0.044] | -0.028 | [-0.035, -0.020] | -0.0001 | 45 % |
| S2_d32_v1 - S0_sel | -0.011 | [-0.053, +0.030] | +0.016 | [+0.002, +0.033] | +0.0001 | 51 % |
| S2_d32_v1 - B0 | -0.002 | [-0.158, +0.151] | +0.077 | [+0.055, +0.101] | -0.0010 | 71 % |
| S2_d32_v2 - S1_d32_v2 | +0.073 | [+0.058, +0.090] | +0.068 | [+0.056, +0.082] | +0.0003 | 68 % |
| S1_d32_v2 - S0_sel | -0.056 | [-0.091, -0.021] | -0.035 | [-0.039, -0.030] | -0.0001 | 38 % |
| S2_d32_v2 - S0_sel | +0.017 | [-0.023, +0.057] | +0.033 | [+0.019, +0.050] | +0.0001 | 49 % |
| S2_d32_v2 - B0 | +0.026 | [-0.129, +0.179] | +0.094 | [+0.073, +0.118] | -0.0009 | 66 % |

## S2 − S1 por tarea

| Tarea | Réplica v1 | Réplica v2 | Media |
|---|---:|---:|---:|
| deblur | +0.019 [+0.00, +0.04] | +0.033 [+0.02, +0.05] | +0.026 |
| dehaze | +0.056 [+0.00, +0.12] | +0.056 [+0.01, +0.12] | +0.056 |
| denoise | +0.115 [+0.10, +0.14] | +0.228 [+0.19, +0.26] | +0.172 |
| derain | +0.093 [+0.07, +0.12] | +0.058 [+0.05, +0.07] | +0.076 |
| enhance | +0.012 [-0.08, +0.10] | -0.010 [-0.05, +0.03] | +0.001 |

## Calidad frente a costo

Latencia: mediana del benchmark v2 en L4, batch 1. FLOPs: programa previo a la optimización, 256×256.

| Modelo | Parámetros | GFLOP | ms 256 | ms 512 | PSNR macro | Δ vs S0_sel |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 14.18 M | 216.3 | 41.58 | 254.0 | 28.1274 | -0.009 |
| S0_sel | 14.27 M | 218.9 | 42.22 | 256.3 | 28.1366 | +0.000 |
| S1_d32_v1 | 11.98 M | 200.5 | 41.14 | 252.8 | 28.0662 | -0.070 |
| S2_d32_v1 | 12.26 M | 200.5 | 41.11 | 252.6 | 28.1254 | -0.011 |
| S1_d32_v2 | 11.98 M | 200.5 | 41.14 | 252.8 | 28.0809 | -0.056 |
| S2_d32_v2 | 12.26 M | 200.5 | 41.11 | 252.6 | 28.1538 | +0.017 |

Test evaluado una sola vez con la selección cerrada en validación. El test ya se había usado el 4 de octubre para B0 y el shared; S1 y S2 no se habían evaluado en test.
