# Plan de mejora de MAXIM + Mixture of Experts

Última actualización: 11 de septiembre de 2026.

## Objetivo

Comparar MAXIM S-2 con MAXIM+MoE bajo el mismo dataset, supervisión multietapa/multiescala y protocolo experimental. El MoE debe producir especialización observable y, finalmente, routing aprendido útil.

No se realizará la ablación MoE con una única cabeza. La baseline relevante es MAXIM original sin MoE, conservando su supervisión profunda.

## Estado actual

| Bloque | Estado | Resultado / acción |
|---|---|---|
| Auditoría y corrección de datos | Completado | Pares problemáticos corregidos o aceptados por criterio de dominio |
| Ampliación SIDD `denoise` | Completado | 5.060 pares nuevos normalizados, combinados y verificados en Drive |
| Supervisión profunda del MoE | Completado | Cinco salidas auxiliares y una salida final MoE para S-2 |
| Diagnósticos del router | Completado | JSON, heatmaps y métricas globales/por tarea |
| Routing oracle | Completado | Routing perfecto; muestreo uniforme corregido |
| Fine-tuning con LR bajo | Completado | Mejora global marginal; LR no es el cuello de botella principal |
| Smoke test del dataset ampliado | Completado | Listas, hashes, rutas y batches reales validados desde Drive |
| Baseline MAXIM comparable | Preparado | Sanity check y luego corrida desde cero en `maxim/maximTrainer_no_moe.ipynb` |
| Expertos con mayor capacidad | Pendiente | Probar cabezas residuales de 2–3 convoluciones |
| Router supervisado y ajuste conjunto | Pendiente | Ejecutar después de seleccionar la mejor arquitectura oracle |

## Datasets finales

La carga ampliada fue revisada directamente en Drive y confirmada como correcta.

| Tarea | Train | Test | Total |
|---|---:|---:|---:|
| deblur | 8.680 | 3.150 | 11.830 |
| dehaze | 909 | 101 | 1.010 |
| denoise | 4.450 | 767 | 5.217 |
| derain | 3.600 | 400 | 4.000 |
| enhance | 5.469 | 30 | 5.499 |
| **Total** | **23.108** | **4.448** | **27.556** |

### Cambios de `denoise`

- Se seleccionaron 34 escenas de SIDD Full sRGB.
- Una escena oficial contenía 110 pares, por lo que se obtuvieron 5.060 y no 5.100 pares nuevos.
- El split nuevo aporta 4.310 train y 750 test, separados por escena.
- Se conservaron los 157 pares anteriores; ocho se reasignaron para evitar fuga entre splits.
- Las listas combinadas finales son `datasets/drive_merge/train.txt` y `datasets/drive_merge/test.txt`.
- SHA-256: train `3f7924a6a20171883cb0a07e3b6c9d1e64d89879526c4baa3f72721403040afa`; test `45fb0b36422f23d017ec2d95957043c2b64301e9423d16766f594cab618ed419`.
- Ruta en Drive: `MyDrive/Facultad/tesis/Datasets/Classifier/denoise`.

### Auditoría cerrada

- `derain`: se corrigieron y validaron 200 GT `rain_light_*`; los `rain_heavy_*` estaban bien.
- `enhance`: se corrigió `a3291-LS051026_day_2_arive38.png`; el duplicado `a2931-jn_20081025_Kent_Shelter_413.png` fue eliminado junto con su entrada.
- `deblur`: los desplazamientos observados corresponden al blur esperado.
- `dehaze`: las imágenes suaves o visualmente duplicadas se aceptaron como efecto del haze.
- Las augmentations aplican transformaciones alineadas a input y GT.

No se repetirá la auditoría completa.

## Evidencia experimental acumulada

### Routing oracle con muestreo corregido

La primera corrida reveló que repetir el dataset después de mezclarlo agotaba primero las tareas pequeñas. La corrección repite cada fuente antes de `sample_from_datasets`. En 120.000 muestras, cada tarea recibió aproximadamente 20 % de exposición.

| Tarea | Oracle corregido |
|---|---:|
| deblur | 24,635 dB |
| dehaze | 41,782 dB |
| denoise | 39,036 dB |
| derain | 26,887 dB |
| enhance | 21,938 dB |
| **Global ponderado** | **25,384 dB** |

El routing mantuvo 100 % de accuracy, confianza 1 y entropía 0. La exposición desigual queda descartada como causa del rezago de `deblur`, `derain` y `enhance`.

### Fine-tuning con optimizador y LR reiniciados

Cinco épocas adicionales con cosine decay `1e-5 → 1e-6` elevaron el PSNR global de `25,384` a `25,422 dB`. La salida final mejoró sólo 0,37 % en L1; por lo tanto, prolongar el mismo fine-tuning no promete una mejora grande. La capacidad de las cabezas es la siguiente hipótesis.

El PSNR global está dominado por `deblur`. Todas las corridas futuras deben informar también promedio macro y resultados por tarea.

## Notebooks preparados

### `maxim/moeExperimentRunner.ipynb`

Este archivo reemplaza a `maxim/moeFineTune.ipynb` como runner de pruebas del MoE.

Configuración predeterminada segura:

- `RUN_DATASET_SMOKE_TEST=True`;
- `START_TRAINING=False`;
- valida 4.450/767 entradas de `denoise`, ausencia de duplicados e intersección y hashes de las listas;
- construye el loader real y decodifica batches de train y test desde Drive;
- comprueba shapes `256×256×3`, valores finitos y rango `[0,1]`;
- con **Run all**, las celdas de modelo, checkpoint y entrenamiento se omiten mientras `START_TRAINING=False`.

Smoke test aprobado en Colab: 4.450 entradas train y 767 test únicas, ambos SHA-256 correctos, 0 archivos faltantes en `imgs/` y `GT/`, y decodificación correcta de 8 muestras train y 4 test mediante el loader real. Resultado final: `DENOISE DATASET SMOKE TEST PASSED`.

Para una corrida posterior se debe elegir un `OUTPUT_DIR` nuevo, revisar `SOURCE_CHECKPOINT_DIR` y cambiar `START_TRAINING=True`. El fine-tuning con SIDD ampliado es una prueba adicional y no sustituye una corrida MoE desde cero para la comparación principal.

### `maxim/maximTrainer_no_moe.ipynb`

Preparado como baseline MAXIM S-2 comparable:

- mismas versiones JAX/Flax, seed, batch size 2, crop 256, dropout 0.1 y weight decay `1e-4`;
- 30 épocas, 2.000 pasos por época, LR `2e-4` y warmup de tres épocas;
- inicialización aleatoria, sin restaurar el baseline anterior;
- mismo pipeline de carga y augmentations que el MoE;
- muestreo uniforme correcto, repitiendo cada tarea antes de mezclar;
- misma pérdida profunda normalizada: L1 final + promedio ponderado de cinco auxiliares;
- validación exhaustiva y determinista de 4.448 pares;
- PSNR/L1 globales y por tarea, PSNR macro y conteos por tarea;
- no contiene validaciones estructurales ni conteos esperados hardcodeados del dataset; esas comprobaciones pertenecen exclusivamente a `moeExperimentRunner.ipynb`;
- checkpoints y `training.log` bajo el `OUTPUT_DIR` configurado;
- `RUN_SANITY_CHECK=True`; revisar `START_TRAINING` y el nombre de salida antes de usar **Run all**.

## Próximos pasos

1. Ejecutar el sanity check del baseline; si pasa, lanzar MAXIM S-2 desde cero.
2. No comparar esa baseline nueva contra el MoE histórico como resultado final: el MoE comparable también debe entrenarse con los 27.556 pares actuales y el mismo presupuesto.
3. Implementar expertos residuales de 2–3 convoluciones y hacer primero un smoke test oracle y una corrida corta.
4. Seleccionar la arquitectura oracle por PSNR por tarea, promedio macro, costo y memoria.
5. Entrenar el router como clasificador de cinco tareas; registrar accuracy y matriz de confusión.
6. Ajustar conjuntamente backbone, expertos y router; luego ejecutar ablaciones de entropía, balance y temperatura.
7. Ejecutar las corridas finales y actualizar la tesis con tablas, curvas, heatmaps y costos.

## Criterios para corridas comparables

- Mismo split, seed, crop, augmentations, batch size, arquitectura S-2 y presupuesto.
- Muestreo uniforme entre tareas durante train; validación completa concatenada sin sampling.
- Mismos pesos de supervisión profunda.
- Cero pares faltantes y conteos exactos por tarea.
- Reportar PSNR/SSIM global ponderado, macro y por tarea.
- Guardar entorno, logs, checkpoints y diagnósticos reproducibles.

## Riesgos activos

- Las cabezas de una convolución pueden ser insuficientes para `deblur`, `derain` y `enhance`.
- El router puede aprender atajos de dataset/cámara en lugar de degradación.
- El PSNR global ponderado puede ocultar regresiones en tareas pequeñas.
- Batch size 2 vuelve ruidosas las regularizaciones de balance por batch.
- Cambios automáticos de dependencias de Colab pueden romper reproducibilidad; conservar el pin JAX/Flax.
