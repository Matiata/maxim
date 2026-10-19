# Benchmark v2: compresión de MAXIM con expertos

- Estado: **complete**
- Protocolo: `86d9c15980815dff46ca1621a844b0352ebbbad1299bd01f4937fda881510e6c`
- Commit: `a4fce8b8f63744faa901321a39cf905a4830b640`
- GPU: NVIDIA L4, 580.82.07, 23034 MiB
- Sólo costo. Los módulos nuevos de S1/S2 tienen pesos aleatorios.

## Parámetros

| Variante | Totales | Usados por tarea | Ejecutados por tarea | Δ totales vs S0_sel | Δ totales vs B0 |
|---|---:|---:|---:|---:|---:|
| B0 | 14,176,178 | 14,176,178 | 14,176,178 | -0.7 % | +0.0 % |
| S0 | 14,272,993 | 14,195,541 | 14,272,993 | +0.0 % | +0.7 % |
| S0_sel | 14,272,993 | 14,195,541 | 14,195,541 | +0.0 % | +0.7 % |
| S1_d32 | 11,981,025 | 11,903,573 | 11,903,573 | -16.1 % | -15.5 % |
| S2_d32 | 12,258,529 | 11,903,573 | 11,903,573 | -14.1 % | -13.5 % |
| S2_d32_todos | 12,258,529 | 11,903,573 | 12,181,077 | -14.1 % | -13.5 % |
| S1_d64 | 12,049,377 | 11,971,925 | 11,971,925 | -15.6 % | -15.0 % |
| S2_d64 | 12,600,289 | 11,971,925 | 11,971,925 | -11.7 % | -11.1 % |
| S0_sel_sin_rcab2 | 11,911,649 | 11,834,197 | 11,834,197 | -16.5 % | -16.0 % |
| ref_sin_cross_gating | 11,891,410 | 11,891,410 | 11,891,410 | -16.7 % | -16.1 % |
| ref_sin_gmlp | 10,249,906 | 10,249,906 | 10,249,906 | -28.2 % | -27.7 % |
| B0@bf16 | 14,176,178 | 14,176,178 | 14,176,178 | -0.7 % | +0.0 % |
| S0_sel@bf16 | 14,272,993 | 14,195,541 | 14,195,541 | +0.0 % | +0.7 % |
| S2_d32@bf16 | 12,258,529 | 11,903,573 | 11,903,573 | -14.1 % | -13.5 % |
| B0_remat | 14,176,178 | 14,176,178 | 14,176,178 | -0.7 % | +0.0 % |

## inference/256x256 (complete)

| Variante | Mediana ms | p95 ms | Dispersión entre bloques | Δ vs B0 | Δ vs S0_sel | GFLOP antes de optimizar | GFLOP compilado | GB accedidos | Kernels est. | Temp. MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 41.58 | 42.85 | 1.8 % | +0.0 % | -1.5 % | — | 129.89 | 15.68 | 2087 | 203 |
| S0 | 43.91 | 45.16 | 1.6 % | +5.6 % | +4.0 % | — | 130.16 | 15.87 | 2124 | 203 |
| S0_sel | 42.22 | 43.37 | 1.6 % | +1.5 % | +0.0 % | — | 129.95 | 15.73 | 2099 | 203 |
| S1_d32 | 41.14 | 42.38 | 1.6 % | -1.0 % | -2.6 % | — | 130.49 | 15.73 | 2107 | 203 |
| S2_d32 | 41.11 | 42.22 | 1.0 % | -1.1 % | -2.6 % | — | 130.49 | 15.74 | 2124 | 203 |
| S2_d32_todos | 41.77 | 43.10 | 1.2 % | +0.5 % | -1.1 % | — | 132.79 | 16.05 | 2226 | 203 |
| S1_d64 | 40.84 | 41.95 | 1.0 % | -1.8 % | -3.3 % | — | 131.03 | 15.76 | 2107 | 203 |
| S2_d64 | 40.92 | 42.20 | 1.4 % | -1.6 % | -3.1 % | — | 131.04 | 15.76 | 2124 | 203 |
| S0_sel_sin_rcab2 | 40.17 | 41.25 | 1.1 % | -3.4 % | -4.8 % | — | 129.98 | 15.62 | 2055 | 203 |
| ref_sin_cross_gating | 33.68 | 34.49 | 1.0 % | -19.0 % | -20.2 % | — | 98.15 | 12.21 | 1748 | 111 |
| ref_sin_gmlp | 27.80 | 28.54 | 1.0 % | -33.1 % | -34.2 % | — | 66.61 | 8.17 | 1423 | 203 |
| B0@bf16 | 19.28 | 20.04 | 1.0 % | -53.6 % | -54.3 % | — | 139.23 | 8.22 | 2173 | 120 |
| S0_sel@bf16 | 19.42 | 20.11 | 0.2 % | -53.3 % | -54.0 % | — | 139.29 | 8.24 | 2195 | 120 |
| S2_d32@bf16 | 18.86 | 19.81 | 1.1 % | -54.6 % | -55.3 % | — | 139.86 | 8.26 | 2230 | 120 |

## inference/512x512 (complete)

| Variante | Mediana ms | p95 ms | Dispersión entre bloques | Δ vs B0 | Δ vs S0_sel | GFLOP antes de optimizar | GFLOP compilado | GB accedidos | Kernels est. | Temp. MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 254.01 | 255.02 | 0.1 % | +0.0 % | -0.9 % | — | 559.64 | 64.21 | 2183 | 603 |
| S0 | 263.15 | 264.78 | 0.4 % | +3.6 % | +2.7 % | — | 560.69 | 64.94 | 2221 | 603 |
| S0_sel | 256.28 | 257.95 | 0.1 % | +0.9 % | +0.0 % | — | 559.85 | 64.39 | 2195 | 603 |
| S1_d32 | 252.79 | 253.94 | 0.1 % | -0.5 % | -1.4 % | — | 562.00 | 64.47 | 2203 | 603 |
| S2_d32 | 252.62 | 253.86 | 0.1 % | -0.5 % | -1.4 % | — | 562.02 | 64.48 | 2236 | 603 |
| S2_d32_todos | 258.80 | 260.06 | 0.0 % | +1.9 % | +1.0 % | — | 571.00 | 66.28 | 2338 | 603 |
| S1_d64 | 252.97 | 254.00 | 0.1 % | -0.4 % | -1.3 % | — | 564.19 | 64.58 | 2203 | 603 |
| S2_d64 | 252.92 | 254.08 | 0.1 % | -0.4 % | -1.3 % | — | 564.21 | 64.58 | 2236 | 603 |
| S0_sel_sin_rcab2 | 251.97 | 253.08 | 0.1 % | -0.8 % | -1.7 % | — | 559.97 | 64.01 | 2147 | 603 |
| ref_sin_cross_gating | 206.73 | 208.07 | 0.2 % | -18.6 % | -19.3 % | — | 425.16 | 49.94 | 1830 | 443 |
| ref_sin_gmlp | 153.02 | 154.61 | 0.1 % | -39.8 % | -40.3 % | — | 315.78 | 32.26 | 1511 | 603 |
| B0@bf16 | 91.41 | 92.77 | 0.4 % | -64.0 % | -64.3 % | — | 632.56 | 33.63 | 2227 | 386 |
| S0_sel@bf16 | 91.88 | 93.24 | 0.2 % | -63.8 % | -64.1 % | — | 632.82 | 33.72 | 2249 | 386 |
| S2_d32@bf16 | 90.33 | 91.51 | 0.2 % | -64.4 % | -64.8 % | — | 635.07 | 33.76 | 2290 | 386 |

## training/256x256_batch2 (complete)

Modo de medición: secuencial por corrida, de mayor a menor memoria temporal; tres bloques por corrida; mismo proceso y caché de autotuning.

| Variante | Mediana ms | p95 ms | Dispersión entre bloques | Δ vs B0 | Δ vs S0_sel | GFLOP antes de optimizar | GFLOP compilado | GB accedidos | Kernels est. | Temp. MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0:full | 463.70 | 468.59 | 0.6 % | +4.0 % | +2.1 % | — | 761.34 | 109.55 | 11793 | 12 442 |
| S0_sel:full | 454.12 | 460.23 | 0.2 % | +1.9 % | +0.0 % | — | 759.21 | 108.65 | 11711 | 12 141 |
| S2_d32:full | 464.79 | 471.69 | 0.3 % | +4.3 % | +2.3 % | — | 762.40 | 108.98 | 11815 | 12 100 |
| B0:full | 445.69 | 450.18 | 0.3 % | +0.0 % | -1.9 % | — | 758.84 | 108.37 | 11601 | 12 047 |
| S1_d32:new_only | 227.26 | 228.30 | 0.1 % | -49.0 % | -50.0 % | — | 511.79 | 71.30 | 4574 | 5 409 |
| S2_d32:new_only | 227.52 | 228.53 | 0.1 % | -49.0 % | -49.9 % | — | 511.83 | 71.12 | 4570 | 5 401 |
| B0_remat:full | 529.51 | 534.02 | 0.1 % | +18.8 % | +16.6 % | — | 1 022.95 | 137.86 | 13574 | 2 270 |

| Corrida | Entrenables | Congelados |
|---|---:|---:|
| S0:full | 14,272,993 | 0 |
| S0_sel:full | 14,272,993 | 0 |
| S2_d32:full | 12,258,529 | 0 |
| B0:full | 14,176,178 | 0 |
| S1_d32:new_only | 69,376 | 11,911,649 |
| S2_d32:new_only | 346,880 | 11,911,649 |
| B0_remat:full | 14,176,178 | 0 |

## Verificaciones

- OK · inference/256x256 · B0: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · S0: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · S0_sel: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · S1_d32: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · S2_d32: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · S2_d32_todos: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · S1_d64: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · S2_d64: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · S0_sel_sin_rcab2: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · ref_sin_cross_gating: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · ref_sin_gmlp: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · B0@bf16: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · S0_sel@bf16: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · S2_d32@bf16: salida finita · `{"shape": [1, 256, 256, 3]}`
- OK · inference/256x256 · S0_sel ≡ S0 (selección efectiva de la cabeza) · `{"max_abs_diff": 0.0, "tolerance": 0.001}`
- OK · inference/256x256 · S2_d32 ≡ S2_d32_todos (selección efectiva del banco) · `{"max_abs_diff": 0.0001258254051208496, "tolerance": 0.001}`
- OK · inference/256x256 · B0@bf16: desvío frente a float32 · `{"max_abs_diff": 0.008617758750915527, "mean_abs_diff": 0.0016108086565509439, "nota": "desvío numérico, no calidad"}`
- OK · inference/256x256 · S0_sel@bf16: desvío frente a float32 · `{"max_abs_diff": 0.010409295558929443, "mean_abs_diff": 0.0017111565684899688, "nota": "desvío numérico, no calidad"}`
- OK · inference/256x256 · S2_d32@bf16: desvío frente a float32 · `{"max_abs_diff": 0.01039743423461914, "mean_abs_diff": 0.001691447920165956, "nota": "desvío numérico, no calidad"}`
- OK · inference/256x256 · S0 tiene 12 convoluciones más: 4 cabezas no seleccionadas × 3 · `{"S0": 237, "S0_sel": 225}`
- OK · inference/256x256 · S2 ejecuta tantas convoluciones como S1 · `{"S2_d32": 217, "S1_d32": 217}`
- OK · inference/256x256 · calcular todos los expertos agrega convoluciones · `{"S2_d32_todos": 249, "S2_d32": 217}`
- OK · inference/512x512 · B0: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · S0: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · S0_sel: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · S1_d32: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · S2_d32: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · S2_d32_todos: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · S1_d64: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · S2_d64: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · S0_sel_sin_rcab2: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · ref_sin_cross_gating: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · ref_sin_gmlp: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · B0@bf16: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · S0_sel@bf16: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · S2_d32@bf16: salida finita · `{"shape": [1, 512, 512, 3]}`
- OK · inference/512x512 · S0_sel ≡ S0 (selección efectiva de la cabeza) · `{"max_abs_diff": 0.0, "tolerance": 0.001}`
- OK · inference/512x512 · S2_d32 ≡ S2_d32_todos (selección efectiva del banco) · `{"max_abs_diff": 0.00012600421905517578, "tolerance": 0.001}`
- OK · inference/512x512 · B0@bf16: desvío frente a float32 · `{"max_abs_diff": 0.009243011474609375, "mean_abs_diff": 0.0016170041635632515, "nota": "desvío numérico, no calidad"}`
- OK · inference/512x512 · S0_sel@bf16: desvío frente a float32 · `{"max_abs_diff": 0.010846912860870361, "mean_abs_diff": 0.0017000712687149644, "nota": "desvío numérico, no calidad"}`
- OK · inference/512x512 · S2_d32@bf16: desvío frente a float32 · `{"max_abs_diff": 0.010741055011749268, "mean_abs_diff": 0.0016918998444452882, "nota": "desvío numérico, no calidad"}`
- OK · inference/512x512 · S0 tiene 12 convoluciones más: 4 cabezas no seleccionadas × 3 · `{"S0": 237, "S0_sel": 225}`
- OK · inference/512x512 · S2 ejecuta tantas convoluciones como S1 · `{"S2_d32": 217, "S1_d32": 217}`
- OK · inference/512x512 · calcular todos los expertos agrega convoluciones · `{"S2_d32_todos": 249, "S2_d32": 217}`
- OK · training/256x256_batch2 · B0:full: pérdida finita y un paso de optimizador · `{"loss": 0.5341626405715942, "grad_norm": 0.2594534456729889}`
- OK · training/256x256_batch2 · B0_remat:full: pérdida finita y un paso de optimizador · `{"loss": 0.5341626405715942, "grad_norm": 0.2594298720359802}`
- OK · training/256x256_batch2 · S0:full: pérdida finita y un paso de optimizador · `{"loss": 0.5345494747161865, "grad_norm": 0.2639930546283722}`
- OK · training/256x256_batch2 · S0_sel:full: pérdida finita y un paso de optimizador · `{"loss": 0.5345494747161865, "grad_norm": 0.26399287581443787}`
- OK · training/256x256_batch2 · S2_d32:full: pérdida finita y un paso de optimizador · `{"loss": 0.5374017953872681, "grad_norm": 0.263611376285553}`
- OK · training/256x256_batch2 · S1_d32:new_only: pérdida finita y un paso de optimizador · `{"loss": 0.5379585027694702, "grad_norm": 0.013751726597547531}`
- OK · training/256x256_batch2 · S2_d32:new_only: pérdida finita y un paso de optimizador · `{"loss": 0.5374020338058472, "grad_norm": 0.0089431032538414}`
- OK · training/256x256_batch2 · rematerialización conserva la pérdida · `{"abs_diff": 0.0}`

## Limitaciones

- Entradas sintéticas: miden costo y tiempo, no calidad.
- Los módulos nuevos de S1/S2 tienen pesos aleatorios: sólo costo.
- La memoria reportada es la estática del ejecutable (memory_analysis). No es un pico medido de buffers vivos.
- Los FLOPs 'lowered' cuentan el programa antes de optimizar y son comparables entre variantes; los 'compiled' dependen de las elecciones del backend.
- La cantidad de kernels se estima con fusiones y custom calls del programa compilado; no cuenta kernels internos de cuDNN.
- Entrenamiento: secuencial por corrida, de mayor a menor memoria temporal; tres bloques por corrida; mismo proceso y caché de autotuning. La alternancia entre variantes sólo se aplica a la inferencia.

## Próximo paso

Entregar `benchmark_v2_results.json`, este reporte y la carpeta `hlo/`. La elección de variantes para recuperar calidad se hace después de revisarlos.
