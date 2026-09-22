# Plan de mejora de MAXIM + Mixture of Experts

Última actualización: 22 de septiembre de 2026.

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
| Baseline MAXIM comparable | Completado | 30 épocas; mejor checkpoint en época 28: 27,455 dB ponderado y 30,822 dB macro |
| MoE oracle comparable | Completado | 30 épocas; mejor resultado en época 30: 27,362 dB ponderado y 30,547 dB macro |
| Expertos con mayor capacidad | Completado | `ResidualExpertHead` entrenado 30 épocas; mejora el macro y algunas tareas, pero no el global ponderado |
| Verificación del archivo SIDD que interrumpió la época 27 | Completado | El par `imgs/`–`GT/` existe, está alineado y ambos PNG se decodifican completamente; el fallo fue transitorio de lectura/montaje |
| Cabeza compartida + expertos residuales | Implementado en `main` | MAXIM conserva su predicción final y los expertos aprenden sólo correcciones residuales condicionadas |
| Expertos latentes K>5 | Planificado | Branch propuesta `exp/latent-experts-k8`; comparar K=5 y K=8 con routing aprendido sin etiquetas de tarea |
| Router supervisado tarea→experto | Secundario / pendiente | Conservar sólo como ablación diagnóstica; no es el siguiente camino principal de mejora |

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
- `SIDD_0062_003_S6_03200_02500_4400_L_001.png`: se verificaron las copias de `imgs/` y `GT/`. Ambas son RGB de 8 bits, miden `5328×3000`, se descargan y decodifican completamente y muestran la misma escena alineada. La entrada tiene el ruido intenso esperado y el GT es la versión limpia. SHA-256 de la descarga completa: `imgs` `fe9f4876ac160cb506055dc63aff9c0b2439fead173b78f80bc3ce6edc48bb28`; `GT` `d104a921bebc0cd85412aa92ab9778540ffe530e66ce78c4179bbca757d09f5a`. El primer intento de descarga durante la revisión también se truncó y el segundo fue correcto, reforzando la hipótesis de una falla transitoria de acceso a Drive.

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

### Baseline MAXIM S-2 sin MoE

La corrida comparable finalizó sus 30 épocas y evaluó los 4.448 pares de test en cada validación. El mejor checkpoint corresponde a la época 28.

| Tarea | PSNR del mejor checkpoint |
|---|---:|
| deblur | 24,554 dB |
| dehaze | 42,824 dB |
| denoise | 38,087 dB |
| derain | 26,428 dB |
| enhance | 22,216 dB |
| **Global ponderado** | **27,455 dB** |
| **Promedio macro** | **30,822 dB** |

La época 30 terminó con 27,406 dB ponderado y 30,768 dB macro, apenas 0,049 dB por debajo del mejor valor, por lo que la corrida llegó a una meseta estable. El log acumulado contiene dos resultados para las épocas 13 y 27: los primeros pertenecen a ramas interrumpidas que no quedaron guardadas. Las reanudaciones restauraron respectivamente los checkpoints completos de las épocas 12 y 26, repitieron esas épocas y continuaron con estado del optimizador, `state.step` y LR coherentes. Para curvas y tablas deben usarse solamente los segundos resultados de las épocas 13 y 27.

Sobre la cadena final de 60.000 pasos y 120.000 muestras, cada tarea recibió entre 19,73 % y 20,59 % de exposición. El muestreo uniforme funcionó como estaba previsto y no introduce un desbalance relevante en esta baseline.

### MoE oracle con dataset ampliado

La corrida `moe_oracle_all_S-2_scratch` del 18/09 completó 30 épocas con el mismo split actual de 4.448 pares, 2.000 pasos por época, S-2, batch, seed y supervisión profunda que la baseline. El mejor resultado fue la época 30.

| Tarea | MAXIM sin MoE (ép. 28) | MoE oracle (ép. 30) | MoE − MAXIM |
|---|---:|---:|---:|
| deblur | 24,554 | 24,659 | +0,105 |
| dehaze | 42,824 | 41,934 | −0,890 |
| denoise | 38,087 | 36,970 | −1,117 |
| derain | 26,428 | 26,931 | +0,503 |
| enhance | 22,216 | 22,240 | +0,024 |
| **Global ponderado** | **27,455** | **27,362** | **−0,093** |
| **Promedio macro** | **30,822** | **30,547** | **−0,275** |

El MoE oracle queda apenas 0,093 dB por debajo de la baseline en el global, por lo que no hay una mejora global demostrada todavía. Sí mejora `derain` en 0,503 dB y `deblur` en 0,105 dB, mientras que pierde en `dehaze` y especialmente `denoise`. El routing oracle fue perfecto en validación (accuracy 1, confianza 1, entropía 0 y uso tarea→experto diagonal), así que esta diferencia ya no se explica por errores de routing.

En los 120.000 ejemplos de entrenamiento, la exposición por tarea fue 19,77 %, 19,94 %, 19,97 %, 19,86 % y 20,46 %, respectivamente. El muestreo tampoco explica el resultado.

### Variante de expertos residuales

Se implementó `ResidualExpertHead` en `maxim/models/moe.py`. Cada experto conserva el ancho de canales de las features de MAXIM, aplica dos convoluciones `3×3` ocultas con GELU, suma una conexión residual interna y termina en una `output_conv` `3×3`. La salida sigue siendo compatible con el residual final `output + input` y la inicialización opcional desde la convolución de salida de MAXIM.

`maxim/moeTrainer.ipynb` queda configurado con `EXPERT_HEAD_VARIANT="residual"`, `EXPERT_NUM_HIDDEN_LAYERS=2` y un `OUTPUT_DIR` separado (`v2_residual_expert_heads`). Antes de una corrida completa se debe ejecutar el preflight/smoke test y comprobar las shapes de la salida final y la carga del checkpoint; no se pudo hacer la inicialización JAX en el entorno local porque JAX sólo está instalado en Colab.

La corrida posterior completó las 30 épocas. En la época 27 ocurrió un fallo transitorio de lectura de `SIDD_0062_003_S6_03200_02500_4400_L_001.png`; el entrenamiento se reanudó desde el checkpoint 26 y finalizó correctamente. El archivo fue revisado posteriormente y es válido. La métrica se considera válida porque la época 27 incompleta no quedó guardada y se ejecutó nuevamente desde el checkpoint completo de la época 26.

| Tarea | Baseline MAXIM | MoE residual (mejor global, ép. 27) | MoE residual − baseline |
|---|---:|---:|---:|
| deblur | 24,554 | 24,611 | +0,057 |
| dehaze | 42,824 | 45,757 | +2,933 |
| denoise | 38,087 | 35,568 | −2,519 |
| derain | 26,428 | 26,244 | −0,184 |
| enhance | 22,216 | 22,720 | +0,504 |
| **Global ponderado** | **27,455** | **27,114** | **−0,341** |
| **Promedio macro** | **30,822** | **30,980** | **+0,158** |

El resultado confirma que aumentar la capacidad de los expertos cambia la especialización, pero no mejora el objetivo global: `dehaze` y `enhance` avanzan, mientras `denoise` cae 2,52 dB. El routing sigue siendo perfecto (accuracy 1, confianza 1, entropía 0) y la exposición fue uniforme, por lo que el cuello de botella está en la capacidad/eficiencia de aprendizaje por experto, no en el router ni en el sampler. Cada experto ve sólo aproximadamente una quinta parte de las muestras; esto puede explicar la regresión de `denoise` frente a la cabeza compartida.

### Arquitectura seleccionada para la siguiente prueba

La próxima arquitectura conservará una cabeza de reconstrucción compartida equivalente a la salida de MAXIM y añadirá expertos residuales que aprenderán solamente correcciones condicionadas:

```text
features F = MAXIM_S2(x)
base       = H_shared(F)
correction = sum_e p_e(F) * DeltaE_e(F)
y_hat      = x + base + correction
```

Componentes y motivación:

- `H_shared` recibe todas las muestras y conserva el conocimiento común de restauración aprendido por MAXIM.
- Cada `DeltaE_e` es un bloque residual pequeño; no debe reconstruir por sí solo la imagen completa.
- El router produce los pesos `p_e` sin recibir la etiqueta de tarea en el experimento de especialización latente.
- Si las correcciones se inicializan cerca de cero, el modelo comienza funcionalmente en el comportamiento de la baseline en vez de degradarlo al introducir el MoE.
- La salida residual global `x + ...` se mantiene compatible con MAXIM.
- La hipótesis es que esta estructura evita que cada experto dependa únicamente de aproximadamente 20 % de los ejemplos, que es la principal limitación observada en el MoE rígido tarea→experto.

La implementación quedó en `maxim/models/moe.py` como `SharedResidualMaximMoE`. La rama base de MAXIM permanece activa para todas las muestras y los expertos usan `ResidualExpertHead(zero_init_output=True)`, por lo que su contribución inicial es cero. La notebook `maxim/moeTrainer.ipynb` usa `EXPERT_HEAD_VARIANT="shared_residual"`, `CORRECTION_SCALE=1.0` y el directorio `v3_shared_residual_head`. La inicialización desde `best_checkpoint/checkpoint_28` queda como siguiente ajuste para una corrida warm-start; esta primera implementación conserva la ruta de inicialización actual y permite validar la arquitectura de forma aislada.

La branch de respaldo `exp/latent-experts-k8` fue creada desde el commit `9fbeb31` (`v2 training`) y no contiene todavía la implementación de la cabeza compartida. El trabajo experimental actual continúa en `main`, como se decidió.

### Branch experimental de especialización latente

Branch propuesta: `exp/latent-experts-k8`.

La branch debe permitir parametrizar `NUM_EXPERTS` para ejecutar como mínimo:

1. `K=5`, routing aprendido: control que separa el efecto del nuevo routing del efecto de agregar expertos.
2. `K=8`, routing aprendido: experimento principal con más expertos que tareas.

Configuración inicial propuesta:

- routing aprendido sin `task_id` y sin pérdida de clasificación de tarea;
- selección top-2, no top-1 rígido;
- routing preferentemente por patches/features espaciales para permitir que una misma imagen use varias especialidades;
- cabeza compartida siempre activa;
- expertos residuales ligeros;
- temperatura alta al comienzo y annealing gradual;
- regularización de balance calculada sobre patches o estadísticas acumuladas entre pasos, no sólo sobre batch, porque el batch size es 2;
- registrar tanto utilización soft como frecuencia top-1/top-2;
- analizar después del entrenamiento la asociación de cada experto con tarea, severidad, brillo, contraste, textura y frecuencia espacial.

No se impondrá inicialmente que los expertos sean estadísticamente independientes de las tareas. La ausencia de `task_id` garantiza que la asignación no está cableada; cualquier correlación tarea→experto será una especialización aprendida que deberá analizarse, no un error por sí misma.

### Comparación provisional con MoE

Como referencia histórica, antes del reentrenamiento del MoE se había comparado la baseline de la época 28 con un routing oracle medido antes de ampliar `denoise`. Esa comparación queda reemplazada por la sección anterior para las conclusiones principales.

| Tarea | MAXIM sin MoE | MoE oracle histórico | Diferencia MAXIM−MoE |
|---|---:|---:|---:|
| deblur | 24,554 | 24,635 | −0,081 |
| dehaze | 42,824 | 41,782 | +1,042 |
| denoise | 38,087 | 39,036 | −0,949 |
| derain | 26,428 | 26,887 | −0,459 |
| enhance | 22,216 | 21,938 | +0,278 |
| **Global ponderado** | **27,455** | **25,384** | **+2,071** |

La lectura histórica no debe usarse para la tabla final de resultados.

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

La corrida final completó las 30 épocas. El sistema de reanudación utilizado:

- elegir entre `none`, `latest`, `best` y `specific` sin construir rutas duplicadas;
- restaurar parámetros, `batch_stats`, momentos del optimizador y `state.step`;
- derivar las épocas completas desde `state.step / steps_per_epoch` y comprobar que coincidan con el nombre del checkpoint;
- recuperar `best_psnr` y su época desde `best_metric.json`;
- impedir que una métrica peor sobrescriba el mejor checkpoint histórico;
- guardar cada nuevo mejor modelo también en el directorio plano usado por `latest`.

Las reanudaciones observadas fueron coherentes con los checkpoints disponibles. El artefacto que debe conservarse para evaluación final es `best_checkpoint/checkpoint_28`; `checkpoint_30` representa el final del entrenamiento, pero no el mejor resultado de validación.

## Handoff de implementación

### Decisión tomada

No continuar optimizando la arquitectura donde cada experto reemplaza completamente la cabeza final y está asignado rígidamente a una tarea. El oracle ya mostró que ese diseño no supera la baseline global y que aumentar la capacidad aislada de las cabezas no resuelve la pérdida de eficiencia por experto.

El siguiente camino principal es **MAXIM con cabeza compartida más expertos residuales de especialización latente**. El router supervisado de cinco tareas queda reservado como ablación para medir la diferencia entre routing oracle y routing aprendido, pero no bloquea este experimento.

### Orden de trabajo

1. Crear la branch `exp/latent-experts-k8` desde el estado estable actual. **Completado:** apunta a `9fbeb31` y el trabajo continúa en `main`.
2. Implementar `SharedResidualMoEHead` sin cambiar inicialmente el backbone ni las cinco salidas auxiliares. **Completado en `main`:** implementado como `SharedResidualMaximMoE`.
3. Añadir una prueba de equivalencia: con todas las correcciones anuladas, la nueva salida debe coincidir numéricamente con MAXIM cargado desde `best_checkpoint/checkpoint_28`. **Pendiente:** requiere cargar el checkpoint de la baseline y ejecutar JAX en Colab.
4. Implementar router sin etiquetas con `NUM_EXPERTS`, `TOP_K` y temperatura configurables.
5. Ejecutar preflight de shapes, carga de checkpoint, gradientes y decodificación completa del dataset.
6. Ejecutar un smoke training corto con `K=5`, comprobando que la pérdida baja, que todos los expertos reciben gradiente y que no aparecen valores no finitos.
7. Ejecutar el mismo smoke test con `K=8` y revisar utilización, entropía, expertos efectivos y distribución por tarea.
8. Realizar una corrida piloto con backbone congelado al comienzo y después descongelado con LR menor que expertos/router.
9. Sólo si no hay colapso y el modelo conserva el rendimiento inicial de la baseline, ejecutar las corridas completas comparables K=5 y K=8.
10. Como control de presupuesto, continuar el checkpoint baseline durante la misma cantidad adicional de pasos que reciba el MoE inicializado desde él.
11. Después de seleccionar K y routing, ejecutar ablaciones de top-1/top-2, temperatura y coeficiente de balance.
12. Ejecutar el router supervisado tarea→experto como ablación secundaria y actualizar la tesis con resultados, curvas, heatmaps y costo computacional.

### Condiciones para detener o corregir una corrida piloto

- algún experto no recibe gradientes o queda sin uso durante varias evaluaciones;
- el número de expertos efectivos colapsa de forma persistente cerca de 1;
- el router satura prematuramente con confianza cercana a 1 antes de que los expertos se diferencien;
- la salida inicial no reproduce la baseline cuando las correcciones están anuladas;
- aparecen regresiones grandes antes de descongelar el backbone;
- aumenta el costo sin una comparación K=5 equivalente que permita atribuir la mejora.

## Criterios para corridas comparables

- Mismo split, seed, crop, augmentations, batch size, arquitectura S-2 y presupuesto.
- Muestreo uniforme entre tareas durante train; validación completa concatenada sin sampling.
- Mismos pesos de supervisión profunda.
- Cero pares faltantes y conteos exactos por tarea.
- Reportar PSNR/SSIM global ponderado, macro y por tarea.
- En corridas inicializadas desde la baseline, comparar contra una continuación de la baseline con el mismo número de pasos adicionales.
- Reportar parámetros totales, parámetros activos, FLOPs aproximados, tiempo por paso y memoria pico.
- Para routing latente, reportar utilización soft, top-1/top-2, entropía, expertos efectivos y matrices tarea×experto.
- Guardar entorno, logs, checkpoints y diagnósticos reproducibles.

## Riesgos activos

- El router puede aprender atajos de dataset/cámara en lugar de propiedades de degradación.
- Con K=8 y batch size 2, una pérdida de balance calculada únicamente por batch es estadísticamente inadecuada; debe operar sobre patches o acumulación temporal.
- Expertos idénticos y una contribución inicial exactamente nula pueden producir simetría o gradientes débiles para el router; usar pequeñas diferencias de inicialización y verificar gradientes.
- Top-2 incrementa el cómputo activo; la comparación debe incluir costo y no sólo parámetros totales.
- La especialización emergente puede seguir correlacionándose con las cinco tareas; esto debe medirse antes de introducir restricciones que podrían perjudicar la calidad.
- El PSNR global ponderado puede ocultar regresiones en tareas pequeñas.
- Las lecturas directas desde Drive pueden fallar transitoriamente aun con archivos válidos; preferir copiar el dataset a almacenamiento local de Colab o reforzar reintentos y preflight de decodificación completa.
- Cambios automáticos de dependencias de Colab pueden romper reproducibilidad; conservar el pin JAX/Flax.
