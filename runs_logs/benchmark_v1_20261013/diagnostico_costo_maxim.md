# Diagnóstico de costo MAXIM baseline/shared

- Estado: **complete**
- Protocolo SHA-256: `ca16d6043448dfafbf48fe3b730eae69692fe4ec1ee089d61b9ebbdaf88662d8`
- Commit: `98984438176afb21694dd1ec29b02fd9b40426c1`
- Nota: este reporte enumera mediciones; la selección de una sustitución se realiza después de revisar los artefactos.

## Parámetros

| Modelo | Totales | Entrenables | Ejecutados actualmente |
|---|---:|---:|---:|
| baseline | 14176178 | 14176178 | 14176178 |
| shared | 14272993 | 14272993 | 14272993 |

## Mediciones

| Clave | Mediana (ms) | p95 (ms) | Mediana (ms/imagen) | FLOPs compilador | Temp. estática (bytes) |
|---|---:|---:|---:|---:|---:|
| baseline/inference/256x256/final_dynamic/fixed | 41.618 | 42.726 | 41.618 | 121287868416 | 212712728 |
| baseline/inference/256x256/full_dynamic/fixed | 41.960 | 42.958 | 41.960 | 121287868416 | 214498584 |
| baseline/inference/512x512/final_dynamic/fixed | 253.763 | 257.666 | 253.763 | 550981074944 | 631913752 |
| baseline/inference/512x512/full_dynamic/fixed | 250.987 | 256.857 | 250.987 | 550981074944 | 635256600 |
| baseline/training/256x256_batch2/full_step | 422.429 | 426.324 | 211.215 | 752399089664 | 12632115176 |
| shared/inference/256x256/final_dynamic/deblur | 44.584 | 45.889 | 44.584 | 129179189248 | 225183272 |
| shared/inference/256x256/final_dynamic/dehaze | 44.613 | 45.824 | 44.613 | 129179189248 | 225183272 |
| shared/inference/256x256/final_dynamic/denoise | 44.115 | 45.251 | 44.115 | 129179189248 | 225183272 |
| shared/inference/256x256/final_dynamic/derain | 44.021 | 45.159 | 44.021 | 129179189248 | 225183272 |
| shared/inference/256x256/final_dynamic/enhance | 44.299 | 45.401 | 44.299 | 129179189248 | 225183272 |
| shared/inference/256x256/final_static/deblur | 44.363 | 45.535 | 44.363 | 129179189248 | 225183272 |
| shared/inference/256x256/final_static/dehaze | 45.588 | 47.073 | 45.588 | 129179189248 | 225183272 |
| shared/inference/256x256/final_static/denoise | 45.102 | 46.466 | 45.102 | 129179189248 | 225183272 |
| shared/inference/256x256/final_static/derain | 45.095 | 46.470 | 45.095 | 129179189248 | 225183272 |
| shared/inference/256x256/final_static/enhance | 46.296 | 47.432 | 46.296 | 129179189248 | 225183272 |
| shared/inference/256x256/full_dynamic/deblur | 44.483 | 46.092 | 44.483 | 129179189248 | 210569256 |
| shared/inference/256x256/full_dynamic/dehaze | 45.435 | 46.928 | 45.435 | 129179189248 | 210569256 |
| shared/inference/256x256/full_dynamic/denoise | 44.776 | 46.332 | 44.776 | 129179189248 | 210569256 |
| shared/inference/256x256/full_dynamic/derain | 44.605 | 45.926 | 44.605 | 129179189248 | 210569256 |
| shared/inference/256x256/full_dynamic/enhance | 44.638 | 46.061 | 44.638 | 129179189248 | 210569256 |
| shared/inference/512x512/final_dynamic/deblur | 251.366 | 253.045 | 251.366 | 504859852800 | 565142056 |
| shared/inference/512x512/final_dynamic/dehaze | 252.803 | 254.379 | 252.803 | 504859852800 | 565142056 |
| shared/inference/512x512/final_dynamic/denoise | 252.898 | 254.328 | 252.898 | 504859852800 | 565142056 |
| shared/inference/512x512/final_dynamic/derain | 252.728 | 254.425 | 252.728 | 504859852800 | 565142056 |
| shared/inference/512x512/final_dynamic/enhance | 252.784 | 254.410 | 252.784 | 504859852800 | 565142056 |
| shared/inference/512x512/final_static/deblur | 253.181 | 254.850 | 253.181 | 504859852800 | 565142056 |
| shared/inference/512x512/final_static/dehaze | 253.074 | 254.794 | 253.074 | 504859852800 | 565142056 |
| shared/inference/512x512/final_static/denoise | 253.005 | 254.672 | 253.005 | 504859852800 | 565142056 |
| shared/inference/512x512/final_static/derain | 252.634 | 254.502 | 252.634 | 504859852800 | 565142056 |
| shared/inference/512x512/final_static/enhance | 252.980 | 254.366 | 252.980 | 504859852800 | 565142056 |
| shared/inference/512x512/full_dynamic/deblur | 249.667 | 251.538 | 249.667 | 504859852800 | 584933928 |
| shared/inference/512x512/full_dynamic/dehaze | 253.015 | 254.750 | 253.015 | 504859852800 | 584933928 |
| shared/inference/512x512/full_dynamic/denoise | 253.266 | 255.064 | 253.266 | 504859852800 | 584933928 |
| shared/inference/512x512/full_dynamic/derain | 253.268 | 254.853 | 253.268 | 504859852800 | 584933928 |
| shared/inference/512x512/full_dynamic/enhance | 253.308 | 254.779 | 253.308 | 504859852800 | 584933928 |
| shared/training/256x256_batch2/full_step | 443.726 | 447.553 | 221.863 | 739155050496 | 13185530088 |

## Limitaciones registradas

- NVML informa memoria residente/reservada del proceso, no sólo buffers vivos. Para comparar picos estrictamente, ejecutar un modelo por runtime limpio mediante MODEL_FILTER.
- cost_analysis y memory_analysis dependen de la versión/backend; campos ausentes se registran como no disponibles.
- Las entradas sintéticas sirven para costo y tiempo, no para calidad. La calidad se toma únicamente de metadata ya existente.
- Profiling JAX desactivado: en Colab una incompatibilidad nativa del PLUGIN_Profiler_Api puede abortar el kernel. Latencia, FLOPs, HLO y memoria NVML se miden sin el profiler.

## Próximo paso

Entregar `benchmark_results.json`, este reporte y las trazas para revisar costos dominantes. No implementar S1/S2 antes de esa revisión.
