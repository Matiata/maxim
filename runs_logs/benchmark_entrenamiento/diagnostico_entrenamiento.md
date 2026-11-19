# Benchmark de entrenamiento: modelos comprimidos y adaptación

- Estado: **complete**
- Protocolo: `69426ab7e13b1c2a1db41434b01983b15c18928be4313f68f963c873567edfe4`
- Commit: `944f1eedb3118ae11b9ea9744c9385a6df8f6264`
- GPU: NVIDIA L4, 580.82.07, 23034 MiB
- Sólo costo. Los módulos nuevos de S1/S2 tienen pesos aleatorios.

## Parámetros

| Variante | Totales | Usados por tarea | Ejecutados por tarea | Δ totales vs S0_sel | Δ totales vs B0 |
|---|---:|---:|---:|---:|---:|
| B0 | 14,176,178 | 14,176,178 | 14,176,178 | -0.7 % | +0.0 % |
| S2_d32 | 12,258,529 | 11,903,573 | 11,903,573 | -14.1 % | -13.5 % |
| S2_d32_l12 | 11,848,161 | 11,348,821 | 11,348,821 | -17.0 % | -16.4 % |
| S0_sel | 14,272,993 | 14,195,541 | 14,195,541 | +0.0 % | +0.7 % |

## training/256x256_batch2 (complete)

Modo de medición: secuencial por corrida, de mayor a menor memoria temporal; tres bloques por corrida; mismo proceso y caché de autotuning.

| Variante | Mediana ms | p95 ms | Dispersión entre bloques | Δ vs B0 | Δ vs S0_sel | GFLOP antes de optimizar | GFLOP compilado | GB accedidos | Kernels est. | Temp. MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S2_d32:full | 465.03 | 470.48 | 0.1 % | +0.3 % | — | — | 755.95 | 108.86 | 11819 | 12 100 |
| B0:full | 463.43 | 470.22 | 0.8 % | +0.0 % | — | — | 752.40 | 108.25 | 11605 | 12 047 |
| S2_d32_l12:adapt | 338.02 | 339.69 | 0.3 % | -27.1 % | — | — | 789.58 | 105.69 | 7046 | 5 854 |
| S2_d32:adapt | 329.67 | 331.07 | 0.1 % | -28.9 % | — | — | 767.85 | 102.87 | 6685 | 5 453 |
| S2_d32:new_only | 229.04 | 230.47 | 0.0 % | -50.6 % | — | — | 505.38 | 71.00 | 4574 | 5 401 |

| Corrida | Entrenables | Congelados |
|---|---:|---:|
| S2_d32:full | 12,258,529 | 0 |
| B0:full | 14,176,178 | 0 |
| S2_d32_l12:adapt | 527,360 | 11,320,801 |
| S2_d32:adapt | 346,880 | 11,911,649 |
| S2_d32:new_only | 346,880 | 11,911,649 |

## Proyección a corridas completas

Mediana por paso × pasos de la corrida. Sólo tiempo de GPU: no incluye carga de datos, validación ni checkpoints.

| Corrida | Paso medido | Pasos | Horas de GPU |
|---|---|---:|---:|
| B0 desde cero | B0:full | 60,000 | 7.72 |
| B0, +10 épocas | B0:full | 20,000 | 2.57 |
| S2_d32 desde cero (hipotético) | S2_d32:full | 60,000 | 7.75 |
| Adaptación de S2_d32 | S2_d32:adapt | 10,000 | 0.92 |
| Adaptación de S2_d32_l12 | S2_d32_l12:adapt | 10,000 | 0.94 |

## Verificaciones

- OK · training/256x256_batch2 · S2_d32:full: pérdida finita y un paso de optimizador · `{"loss": 0.5374016761779785, "grad_norm": 0.2635512053966522}`
- OK · training/256x256_batch2 · B0:full: pérdida finita y un paso de optimizador · `{"loss": 0.5341627597808838, "grad_norm": 0.25941595435142517}`
- OK · training/256x256_batch2 · S2_d32_l12:adapt: pérdida finita y un paso de optimizador · `{"loss": 0.745055079460144, "grad_norm": 0.12985260784626007}`
- OK · training/256x256_batch2 · S2_d32:adapt: pérdida finita y un paso de optimizador · `{"loss": 0.7011626362800598, "grad_norm": 0.18566358089447021}`
- OK · training/256x256_batch2 · S2_d32:new_only: pérdida finita y un paso de optimizador · `{"loss": 0.5374013781547546, "grad_norm": 0.008950806222856045}`
- OK · training/256x256_batch2 · adapt y new_only entrenan los mismos parámetros · `{"new_only": 346880, "adapt": 346880}`

## Limitaciones

- Entradas sintéticas: miden costo y tiempo, no calidad.
- Los módulos nuevos de S1/S2 tienen pesos aleatorios: sólo costo.
- La memoria reportada es la estática del ejecutable (memory_analysis). No es un pico medido de buffers vivos.
- Los FLOPs 'lowered' cuentan el programa antes de optimizar y son comparables entre variantes; los 'compiled' dependen de las elecciones del backend.
- La cantidad de kernels se estima con fusiones y custom calls del programa compilado; no cuenta kernels internos de cuDNN.
- Entrenamiento: secuencial por corrida, de mayor a menor memoria temporal; tres bloques por corrida; mismo proceso y caché de autotuning. La alternancia entre variantes sólo se aplica a la inferencia.

## Próximo paso

Entregar `benchmark_entrenamiento_results.json` y este reporte. Los tiempos se comparan sólo dentro de esta corrida.
