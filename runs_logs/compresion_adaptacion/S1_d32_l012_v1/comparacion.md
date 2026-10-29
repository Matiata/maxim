# Opción 1: S1_d32_l012, réplica v1

Validación, crop central 256. GFLOP: programa sin optimizar, batch 1. S0_sel es el profesor evaluado en esta corrida; B0 sólo aporta el costo.

| Modelo | Parámetros | vs B0 | GFLOP 256 | vs B0 | Época | PSNR macro | Δ vs profesor | Ponderado | deblur | dehaze | denoise | derain | enhance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 14.176 M | +0.0 % | — | — | — | — | — | — | — | — | — | — | — |
| S0_sel | 14.273 M | +0.7 % | — | — | — | 32.0633 | +0.000 | 27.4207 | 24.939 | 51.043 | 34.225 | 27.268 | 22.842 |
| S1_d32 | 11.981 M | -15.5 % | — | — | 3 | 32.0218 | -0.042 | 27.4160 | 24.987 | 50.911 | 34.252 | 27.129 | 22.830 |
| S2_d32 | 12.259 M | -13.5 % | — | — | 4 | 32.1334 | +0.070 | 27.4813 | 25.071 | 51.244 | 34.232 | 27.253 | 22.867 |
| S1_d32_l012 | 11.298 M | -20.3 % | — | — | 5 | 30.9386 | -1.125 | 26.9898 | 24.764 | 46.992 | 33.614 | 26.519 | 22.803 |
| S2_d32_l012 | 11.797 M | -16.8 % | — | — | 5 | 31.6856 | -0.378 | 27.4173 | 25.165 | 49.221 | 34.265 | 26.843 | 22.934 |

Criterio: Δ macro vs profesor ≥ −0.05 dB y ninguna tarea por debajo de −0.1 dB.
Δ macro -1.125 dB; peor tarea -4.051 dB (dehaze). **No favorable.**
