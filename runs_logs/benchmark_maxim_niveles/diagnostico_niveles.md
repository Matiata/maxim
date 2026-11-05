# Benchmark de niveles: expertos RCAB en uno, dos y tres niveles

- Estado: **complete**
- Protocolo: `46124f265709d980d2984009ea71f6dfbc4f2448db31c346401de0d3b5eca47d`
- Commit: `e8e9af5cf55b1bd3b0184ff841f1bdcfea84b170`
- GPU: NVIDIA L4, 580.82.07, 23034 MiB
- Sólo costo. Los módulos nuevos de S1/S2 tienen pesos aleatorios.

## Parámetros

| Variante | Totales | Usados por tarea | Ejecutados por tarea | Δ totales vs S0_sel | Δ totales vs B0 |
|---|---:|---:|---:|---:|---:|
| B0 | 14,176,178 | 14,176,178 | 14,176,178 | -0.7 % | +0.0 % |
| S0_sel | 14,272,993 | 14,195,541 | 14,195,541 | +0.0 % | +0.7 % |
| S1_d32 | 11,981,025 | 11,903,573 | 11,903,573 | -16.1 % | -15.5 % |
| S2_d32 | 12,258,529 | 11,903,573 | 11,903,573 | -14.1 % | -13.5 % |
| S1_d32_l12 | 11,426,273 | 11,348,821 | 11,348,821 | -19.9 % | -19.4 % |
| S2_d32_l12 | 11,848,161 | 11,348,821 | 11,348,821 | -17.0 % | -16.4 % |
| S1_d32_l012 | 11,297,761 | 11,220,309 | 11,220,309 | -20.8 % | -20.3 % |
| S2_d32_l012 | 11,797,473 | 11,220,309 | 11,220,309 | -17.3 % | -16.8 % |

## inference/256x256 (complete)

| Variante | Mediana ms | p95 ms | Dispersión entre bloques | Δ vs B0 | Δ vs S0_sel | GFLOP antes de optimizar | GFLOP compilado | GB accedidos | Kernels est. | Temp. MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 41.73 | 42.81 | 1.7 % | +0.0 % | -1.1 % | 216.33 | 129.89 | 15.68 | 2087 | 203 |
| S0_sel | 42.18 | 43.42 | 3.3 % | +1.1 % | +0.0 % | 218.91 | 129.95 | 15.73 | 2099 | 203 |
| S1_d32 | 41.32 | 42.73 | 3.7 % | -1.0 % | -2.0 % | 200.53 | 130.49 | 15.73 | 2107 | 203 |
| S2_d32 | 41.62 | 42.91 | 1.6 % | -0.3 % | -1.3 % | 200.53 | 130.49 | 15.74 | 2124 | 203 |
| S1_d32_l12 | 41.13 | 43.15 | 3.9 % | -1.4 % | -2.5 % | 182.56 | 131.59 | 15.83 | 2115 | 203 |
| S2_d32_l12 | 40.78 | 42.26 | 0.6 % | -2.3 % | -3.3 % | 182.56 | 131.60 | 15.83 | 2164 | 203 |
| S1_d32_l012 | 39.86 | 40.86 | 0.2 % | -4.5 % | -5.5 % | 165.88 | 133.87 | 16.23 | 2123 | 203 |
| S2_d32_l012 | 39.98 | 41.06 | 0.7 % | -4.2 % | -5.2 % | 165.88 | 133.90 | 16.23 | 2212 | 203 |

## inference/512x512 (complete)

| Variante | Mediana ms | p95 ms | Dispersión entre bloques | Δ vs B0 | Δ vs S0_sel | GFLOP antes de optimizar | GFLOP compilado | GB accedidos | Kernels est. | Temp. MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 257.68 | 263.04 | 1.3 % | +0.0 % | -0.6 % | 942.12 | 559.64 | 64.21 | 2183 | 603 |
| S0_sel | 259.33 | 262.57 | 0.0 % | +0.6 % | +0.0 % | 952.44 | 559.85 | 64.39 | 2195 | 603 |
| S1_d32 | 255.29 | 259.72 | 0.6 % | -0.9 % | -1.6 % | 878.13 | 562.00 | 64.47 | 2203 | 603 |
| S2_d32 | 255.65 | 259.45 | 0.3 % | -0.8 % | -1.4 % | 878.13 | 562.02 | 64.48 | 2236 | 603 |
| S1_d32_l12 | 251.39 | 255.07 | 0.1 % | -2.4 % | -3.1 % | 805.85 | 566.40 | 64.87 | 2211 | 603 |
| S2_d32_l12 | 251.36 | 255.09 | 0.2 % | -2.5 % | -3.1 % | 805.85 | 566.45 | 64.87 | 2284 | 603 |
| S1_d32_l012 | 251.55 | 254.51 | 0.2 % | -2.4 % | -3.0 % | 738.94 | 571.23 | 66.48 | 2227 | 603 |
| S2_d32_l012 | 249.72 | 252.67 | 0.0 % | -3.1 % | -3.7 % | 738.94 | 575.64 | 66.48 | 2332 | 603 |

## training/256x256_batch2 (complete)

Modo de medición: secuencial por corrida, de mayor a menor memoria temporal; tres bloques por corrida; mismo proceso y caché de autotuning.

| Variante | Mediana ms | p95 ms | Dispersión entre bloques | Δ vs B0 | Δ vs S0_sel | GFLOP antes de optimizar | GFLOP compilado | GB accedidos | Kernels est. | Temp. MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0:full | 452.87 | 457.64 | 0.6 % | +0.0 % | — | — | 758.84 | 108.37 | 11601 | 12 047 |
| S2_d32_l012:new_only | 256.12 | 258.72 | 0.2 % | -43.4 % | — | — | 560.41 | 79.53 | 5240 | 6 601 |
| S2_d32_l12:new_only | 238.30 | 240.53 | 0.1 % | -47.4 % | — | — | 533.42 | 73.68 | 4909 | 5 707 |
| S2_d32:new_only | 230.48 | 233.61 | 0.3 % | -49.1 % | — | — | 511.83 | 71.12 | 4570 | 5 401 |

| Corrida | Entrenables | Congelados |
|---|---:|---:|
| B0:full | 14,176,178 | 0 |
| S2_d32_l012:new_only | 624,640 | 11,172,833 |
| S2_d32_l12:new_only | 527,360 | 11,320,801 |
| S2_d32:new_only | 346,880 | 11,911,649 |

## Verificaciones

- OK · inference/256x256 · B0: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · S0_sel: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · S1_d32: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · S2_d32: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · S1_d32_l12: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · S2_d32_l12: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · S1_d32_l012: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · S2_d32_l012: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · S2 ejecuta tantas convoluciones como S1 · `{"S2_d32": 217, "S1_d32": 217}`
- OK · inference/256x256 · S2_d32_l12 ejecuta tantas convoluciones como S1_d32_l12 · `{"S2_d32_l12": 209, "S1_d32_l12": 209}`
- OK · inference/256x256 · S2_d32_l012 ejecuta tantas convoluciones como S1_d32_l012 · `{"S2_d32_l012": 201, "S1_d32_l012": 201}`
- OK · inference/512x512 · B0: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · S0_sel: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · S1_d32: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · S2_d32: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · S1_d32_l12: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · S2_d32_l12: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · S1_d32_l012: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · S2_d32_l012: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · S2 ejecuta tantas convoluciones como S1 · `{"S2_d32": 217, "S1_d32": 217}`
- OK · inference/512x512 · S2_d32_l12 ejecuta tantas convoluciones como S1_d32_l12 · `{"S2_d32_l12": 209, "S1_d32_l12": 209}`
- OK · inference/512x512 · S2_d32_l012 ejecuta tantas convoluciones como S1_d32_l012 · `{"S2_d32_l012": 201, "S1_d32_l012": 201}`
- OK · training/256x256_batch2 · B0:full: pérdida finita y un paso de optimizador · `{"loss": 0.5341626405715942, "grad_norm": 0.2594541311264038}`
- OK · training/256x256_batch2 · S2_d32_l012:new_only: pérdida finita y un paso de optimizador · `{"loss": 0.6303548216819763, "grad_norm": 0.09767128527164459}`
- OK · training/256x256_batch2 · S2_d32_l12:new_only: pérdida finita y un paso de optimizador · `{"loss": 0.563119649887085, "grad_norm": 0.027378225699067116}`
- OK · training/256x256_batch2 · S2_d32:new_only: pérdida finita y un paso de optimizador · `{"loss": 0.5374020338058472, "grad_norm": 0.008943146094679832}`

## Limitaciones

- Entradas sintéticas: miden costo y tiempo, no calidad.
- Los módulos nuevos de S1/S2 tienen pesos aleatorios: sólo costo.
- La memoria reportada es la estática del ejecutable (memory_analysis). No es un pico medido de buffers vivos.
- Los FLOPs 'lowered' cuentan el programa antes de optimizar y son comparables entre variantes; los 'compiled' dependen de las elecciones del backend.
- La cantidad de kernels se estima con fusiones y custom calls del programa compilado; no cuenta kernels internos de cuDNN.
- Entrenamiento: secuencial por corrida, de mayor a menor memoria temporal; tres bloques por corrida; mismo proceso y caché de autotuning. La alternancia entre variantes sólo se aplica a la inferencia.

## Próximo paso

Entregar `benchmark_niveles_results.json`, este reporte y la carpeta `hlo/`. Las latencias de las versiones por niveles se comparan sólo contra B0 y S2_d32 medidos en esta misma corrida.
