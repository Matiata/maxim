# Comparación en val: variantes comprimidas y réplicas

Protocolo `a4f0820864f8e118cfeae1a824e49675342fd17cf0cedc2703d4ce778d34c357`. 2303 pares, crop central 256. Réplicas: v1, v2.

## Calidad por tarea, PSNR en dB

| Modelo | Época | deblur | dehaze | denoise | derain | enhance | Macro | Ponderado | SSIM macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | — | 24.387 | 50.579 | 34.197 | 27.238 | 22.973 | 31.8748 | 27.2203 | 0.8550 |
| S0_sel | — | 24.939 | 51.043 | 34.225 | 27.268 | 22.842 | 32.0633 | 27.4207 | 0.8576 |
| S1_d32_v1 | 3 | 24.986 | 50.912 | 34.252 | 27.129 | 22.830 | 32.0218 | 27.4159 | 0.8579 |
| S2_d32_v1 | 4 | 25.071 | 51.244 | 34.232 | 27.254 | 22.867 | 32.1334 | 27.4813 | 0.8580 |
| S1_d32_v2 | 4 | 24.922 | 50.771 | 34.232 | 27.168 | 22.836 | 31.9858 | 27.3909 | 0.8571 |
| S2_d32_v2 | 5 | 25.012 | 51.177 | 34.223 | 27.239 | 22.868 | 32.1040 | 27.4536 | 0.8574 |

## Diferencias promediadas entre réplicas

Para cada imagen se promedia la diferencia de las réplicas y luego se remuestrea por tarea. Con dos réplicas, el intervalo todavía no representa bien la variación entre semillas: comparar también las réplicas por separado.

| Comparación | Δ PSNR macro | IC 95 % | Δ PSNR ponderado | IC 95 % | Δ SSIM macro | Imágenes que mejoran |
|---|---:|---|---:|---|---:|---:|
| S2_d32 - S1_d32 | +0.115 | [+0.086, +0.154] | +0.064 | [+0.045, +0.086] | +0.0002 | 58 % |
| S2_d32 - S0_sel | +0.055 | [+0.024, +0.092] | +0.047 | [+0.023, +0.074] | +0.0001 | 46 % |
| S1_d32 - S0_sel | -0.060 | [-0.075, -0.044] | -0.017 | [-0.030, -0.003] | -0.0001 | 38 % |
| S2_d32 - B0 | +0.244 | [+0.171, +0.330] | +0.247 | [+0.154, +0.351] | +0.0027 | 56 % |

## Diferencias por réplica

| Comparación | Δ PSNR macro | IC 95 % | Δ PSNR ponderado | IC 95 % | Δ SSIM macro | Imágenes que mejoran |
|---|---:|---|---:|---|---:|---:|
| S0_sel - B0 | +0.189 | [+0.129, +0.255] | +0.200 | [+0.111, +0.300] | +0.0026 | 61 % |
| S2_d32_v1 - S1_d32_v1 | +0.112 | [+0.084, +0.145] | +0.065 | [+0.042, +0.093] | +0.0001 | 53 % |
| S1_d32_v1 - S0_sel | -0.042 | [-0.060, -0.022] | -0.005 | [-0.018, +0.011] | +0.0003 | 47 % |
| S2_d32_v1 - S0_sel | +0.070 | [+0.036, +0.109] | +0.061 | [+0.032, +0.093] | +0.0004 | 48 % |
| S2_d32_v1 - B0 | +0.259 | [+0.184, +0.347] | +0.261 | [+0.166, +0.368] | +0.0030 | 58 % |
| S2_d32_v2 - S1_d32_v2 | +0.118 | [+0.084, +0.164] | +0.063 | [+0.047, +0.080] | +0.0003 | 62 % |
| S1_d32_v2 - S0_sel | -0.078 | [-0.098, -0.058] | -0.030 | [-0.042, -0.015] | -0.0005 | 36 % |
| S2_d32_v2 - S0_sel | +0.041 | [+0.011, +0.075] | +0.033 | [+0.012, +0.056] | -0.0002 | 44 % |
| S2_d32_v2 - B0 | +0.229 | [+0.158, +0.313] | +0.233 | [+0.142, +0.336] | +0.0025 | 54 % |

## S2 − S1 por tarea

| Tarea | Réplica v1 | Réplica v2 | Media |
|---|---:|---:|---:|
| deblur | +0.084 [+0.03, +0.15] | +0.091 [+0.06, +0.13] | +0.087 |
| dehaze | +0.332 [+0.22, +0.49] | +0.406 [+0.24, +0.62] | +0.369 |
| denoise | -0.019 [-0.02, -0.01] | -0.009 [-0.01, -0.01] | -0.014 |
| derain | +0.124 [+0.10, +0.16] | +0.071 [+0.05, +0.09] | +0.098 |
| enhance | +0.037 [+0.02, +0.06] | +0.032 [+0.01, +0.06] | +0.034 |

## Calidad frente a costo

Latencia: mediana del benchmark v2 en L4, batch 1. FLOPs: programa previo a la optimización, 256×256.

| Modelo | Parámetros | GFLOP | ms 256 | ms 512 | PSNR macro | Δ vs S0_sel |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 14.18 M | 216.3 | 41.58 | 254.0 | 31.8748 | -0.189 |
| S0_sel | 14.27 M | 218.9 | 42.22 | 256.3 | 32.0633 | +0.000 |
| S1_d32_v1 | 11.98 M | 200.5 | 41.14 | 252.8 | 32.0218 | -0.042 |
| S2_d32_v1 | 12.26 M | 200.5 | 41.11 | 252.6 | 32.1334 | +0.070 |
| S1_d32_v2 | 11.98 M | 200.5 | 41.14 | 252.8 | 31.9858 | -0.078 |
| S2_d32_v2 | 12.26 M | 200.5 | 41.11 | 252.6 | 32.1040 | +0.041 |

Selección de S1/S2 por PSNR macro de validación; estos números son de la misma validación y por eso son optimistas. La confirmación independiente corresponde al test final.
