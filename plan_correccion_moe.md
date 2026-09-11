# Plan unificado de mejora de MAXIM + Mixture of Experts

Última actualización: 10 de septiembre de 2026.

Este documento reemplaza a `plan_mejoras.md` y concentra el diagnóstico, el orden de trabajo, los avances y los criterios de éxito de la tesis.

## Objetivo de esta iteración

La variante MAXIM+MoE debe:

1. conservar la supervisión multietapa y multiescala del MAXIM original;
2. producir una especialización observable de los expertos;
3. aprender un routing útil y medible;
4. compararse con MAXIM original bajo la misma supervisión y el mismo protocolo experimental;
5. entrenarse y evaluarse sobre pares de imágenes verificados.

El cambio de formato de la tesis queda postergado hasta cerrar la implementación y obtener evidencia experimental.

## Estado general

| Bloque | Estado | Resultado o siguiente acción |
|---|---|---|
| Auditoría de datasets y augmentations | Completado | 22.496 pares utilizables; excepciones corregidas o aceptadas explícitamente |
| Ampliación de `denoise` con SIDD Full | En curso | 16/34 escenas y 2.400 pares normalizados; procesamiento reanudado sobre las 18 escenas restantes con cuatro workers |
| Métricas del router | Completado | Métricas por experto y tarea, JSON y heatmaps integrados en `maxim/moeTrainer.ipynb` |
| Entorno reproducible JAX/Flax | Completado | `moeTrainer.ipynb` fija una versión compatible y volvió a ejecutar correctamente |
| Supervisión multietapa/multiescala | Completado | Implementación y smoke test JAX/Flax validados en Colab |
| Routing oráculo | Completado | Dos corridas reales de 30 épocas analizadas; la segunda confirmó exposición uniforme y routing perfecto |
| Fine-tuning oracle con LR reducido | Completado | Cinco épocas ejecutadas; PSNR global `25,384 → 25,422 dB`, una mejora demasiado pequeña para explicar las tareas rezagadas |
| Expertos con mayor capacidad | Próximo experimento | Probar cabezas residuales de dos o tres convoluciones bajo routing oracle antes del router aprendido |
| Router supervisado | Pendiente | Entrenarlo como clasificador de las cinco tareas |
| Ajuste conjunto y ablaciones | Pendiente | Comparar regularización, temperatura y estrategia de routing |
| Corrida final y material de tesis | Pendiente | Ejecutar sólo después de cerrar datos, código y smoke tests |

## Avances verificados

### Auditoría de datasets

Se creó y ejecutó `tesis/diagnostico_datasets_colab.ipynb` sobre los cinco datasets. La corrida original contabilizó 22.497 pares. Tras confirmar y eliminar un duplicado de `enhance/train` —incluidos sus archivos y su entrada en `train.txt`—, el conjunto utilizable quedó en 22.496 pares:

| Tarea | Train | Test | Total |
|---|---:|---:|---:|
| deblur | 8.680 | 3.150 | 11.830 |
| dehaze | 909 | 101 | 1.010 |
| denoise | 142 | 15 | 157 |
| derain | 3.600 | 400 | 4.000 |
| enhance | 5.469 | 30 | 5.499 |
| **Total** | **18.800** | **3.696** | **22.496** |

Resultados confirmados:

- todos los pares listados existen y se pueden decodificar;
- no hay entradas repetidas dentro de `train.txt` o `test.txt`;
- ninguna imagen necesita padding para un crop de `256 × 256`;
- `random_crop`, `random_flip` y `random_rotation` aplican la misma transformación a input y GT; ambos notebooks pasaron 50 pruebas aleatorias;
- los desplazamientos detectados en `deblur` corresponden al desenfoque de la entrada y se aceptan como comportamiento esperado;
- los 500 pares visualmente suaves o idénticos de `dehaze` se aceptan por decisión de dominio y no bloquean el entrenamiento;
- se detectaron 80 pares de `derain` con dimensiones u orientación incompatibles y uno de `enhance`;
- se detectó un par input=GT adicional en `enhance`, luego confirmado como duplicado accidental y eliminado;
- hay 53 archivos físicos no listados o sin contraparte. No participan en el entrenamiento actual, pero deben quedar documentados.

La inspección directa confirmó que `rain_light_train-34x2.png` emparejaba escenas distintas. Los casos cercanos `33x2` a `36x2` también mezclaban contenidos diferentes. El problema no era una rotación: parte de `derain` había quedado asociada con el GT equivocado.

Se creó `tesis/diagnostico_derain_matching.py`. El script:

- mide correspondencia visual con SIFT y RANSAC;
- busca un GT alternativo dentro del mismo split;
- exige un mínimo de inliers y margen contra el segundo candidato;
- genera `propuestas_emparejamiento_derain.csv`;
- genera `revision_propuestas_derain.pdf` con input, GT actual y GT propuesto;
- no modifica el dataset en su modo predeterminado;
- al usar `--apply`, respalda cada GT reemplazado y guarda un manifiesto con hashes.

La revisión del PDF determinó que las sugerencias sobre `rain_heavy_*` eran falsos positivos y que los errores reales estaban limitados a destinos `rain_light_*`. El CSV de trabajo se filtró a 206 casos light y el diagnóstico completo se conservó por separado.

La reparación de `derain` quedó completada:

- se conservaron sin cambios los seis GT correctos `rain_light_train-22x2.png` a `27x2.png`;
- se reemplazaron 79 GT de train: índices 3–9 y 28–99;
- se reemplazaron 121 GT de test: índices 1–2, 10–27 y 100–200;
- para train se usó como fuente limpia `rain_heavy_train-(N+1200)x2.png`;
- para test se usó `rain_heavy_test-Nx2.png`;
- los casos especiales 51, 78, 83 y 96 se confirmaron visualmente; para 78 y 96 se descartaron las propuestas erróneas del matcher y se aplicaron los índices 1278 y 1296;
- `tesis/reparar_derain_light.py` restringió todas las escrituras a destinos `rain_light_*`, realizó backups y verificó las copias con SHA-256;
- la comprobación posterior informó 0 reemplazos pendientes y 200 pares iguales a su fuente esperada;
- la reparación completa también fue confirmada manualmente mediante inspección visual.

Estado de esta corrección: **completada y validada automática y visualmente**. No quedan correcciones pendientes en `derain`.

En `enhance`, `a3291-LS051026_day_2_arive38.png` fue corregida manualmente después de confirmar la incompatibilidad geométrica. El par input=GT inesperado `train/a2931-jn_20081025_Kent_Shelter_413.png` fue revisado y confirmado como duplicado; se eliminaron ambos archivos y su referencia de `train.txt`.

Por decisión experimental no se repetirá la auditoría completa: se aceptan los resultados ya recolectados y las validaciones automática y manual de las correcciones realizadas.

Los 22.496 pares corresponden al conjunto base ya auditado. La ampliación de `denoise` descrita a continuación todavía no forma parte de ese total ni de la corrida de fine-tuning ya completada.

### Ampliación del dataset de denoise

Se inició una ampliación con SIDD Full para corregir la diferencia extrema entre los 142 pares de entrenamiento de `denoise` y el tamaño de las demás tareas. La versión pública contiene 160 escenas con 150 pares noisy/GT por escena. Se seleccionaron de forma determinista 34 escenas distribuidas sobre la lista pública, equivalentes a 5.100 pares.

Estado al 10 de septiembre de 2026:

- `maxim/download_sidd_subset.py` descarga únicamente `NOISY_SRGB` y `GT_SRGB`; no descarga RAW ni metadata;
- la primera descarga fue pausada para evitar llenar el disco y luego reemplazada por el flujo progresivo descrito abajo;
- son 68 archivos ZIP y aproximadamente 197,63 GiB comprimidos;
- `datasets/SIDD_Full_5100_sRGB_archives/manifest.json` conserva escenas, mirrors, tamaños esperados y estado por archivo;
- cada ZIP terminado se valida por tamaño, integridad y presencia de exactamente 150 PNG;
- la selección apunta a aproximadamente 5.100 pares, pero algunas escenas oficiales contienen menos de 150 imágenes; el total definitivo se fijará al completar la normalización y validación.

Durante la primera descarga quedaron tres instancias simultáneas escribiendo sobre los mismos destinos. La carpeta alcanzó aproximadamente 313 GiB. Se detuvieron los tres grupos de procesos y se eliminaron 52 ZIP sobredimensionados y 10 parciales no confiables, liberando 302,43 GiB. Se conservaron seis ZIP —11,44 GiB— que coinciden con el tamaño oficial, pasan la verificación CRC completa y contienen 150 PNG cada uno. En ese momento sólo la escena `0054` tenía ambos archivos noisy/GT completos; los otros cuatro ZIP eran GT cuyo noisy se aprovechará al llegar a esas escenas. Después de la limpieza quedaron aproximadamente 323 GiB libres.

`maxim/download_sidd_subset.py` ahora usa un lock exclusivo de sistema: una segunda instancia aborta antes de descargar. También fuerza HTTP/1.1 para evitar los errores de stream HTTP/2 observados.

Se agregó `maxim/download_normalize_sidd_subset.py` para continuar con uso de disco acotado. El nuevo flujo toma una escena por worker, descarga sus archivos noisy y GT secuencialmente, valida tamaño e integridad, extrae los 150 pares con nombres idénticos en `imgs/` y `GT/`, actualiza los listados y sólo entonces elimina ambos ZIP. Usa el mismo lock exclusivo y cuatro workers, la mitad de la concurrencia anterior.

Antes del reinicio del equipo se completaron y verificaron 16 escenas: 2.400 pares, distribuidos en 2.100 de train y 300 de test. Sus ZIP ya fueron eliminados. El proceso se reanudó sobre las 18 escenas restantes con cuatro workers. El split quedó fijado por escena en 29 escenas de train y cinco de test; se eligió una escena de test por cada cámara (`S6`, `N6`, `IP`, `GP`, `G4`). Los conteos máximos previstos eran 4.350/750, pero se ajustarán al contenido real de los archivos oficiales.

La escena `0087` reveló que no todos los archivos públicos contienen 150 imágenes: su ZIP noisy oficial es íntegro, coincide con el tamaño publicado y contiene 110 PNG contiguos. El pipeline ahora acepta cantidades variables entre escenas, pero sólo normaliza cuando noisy y GT presentan exactamente los mismos índices.

Trabajo pendiente para integrar SIDD:

1. completar las 33 escenas restantes con el flujo progresivo;
2. verificar los 5.100 pares, los conteos finales y la ausencia de duplicados;
3. subir el dataset normalizado a Drive y actualizar `TASK_DIRS` o la ruta de `denoise`.

Esta ampliación se inició en paralelo al fine-tuning y no modificó sus datos. Su efecto se medirá en una corrida posterior con el nuevo split cerrado.

### Métricas del router

`maxim/moeTrainer.ipynb` ya conserva las dimensiones de tarea y experto durante la recolección y la agregación. La implementación incluye:

- probabilidad media por experto;
- frecuencia top‑1 por experto;
- entropía media y entropía normalizada por `log(K)`;
- cantidad efectiva de expertos, calculada como `exp(entropía)`;
- probabilidad máxima media como medida de confianza;
- conteo de muestras por tarea;
- matrices tarea × experto de uso soft y frecuencia top‑1;
- entropía y confianza desagregadas por tarea;
- agregación exacta por número de muestras, incluso con un último batch más pequeño;
- impresión de diagnósticos para train y validación;
- un JSON por época en `router_diagnostics/`;
- cuatro heatmaps por época: uso soft y top‑1 para train y validación.

El antiguo escalar cercano a `0.2` ya no se interpreta como uso individual de los cinco expertos.

Quedan fuera de este bloque, porque requieren otros cambios:

- diversidad entre las salidas y los parámetros de los expertos;
- accuracy y matriz de confusión del router supervisado;
- métricas de routing oráculo frente a routing aprendido.

### Compatibilidad del entorno

La inicialización de `moe_model` falló con JAX 0.11 porque Flax todavía accedía a `jax.core.get_opaque_trace_state`, eliminado en esa versión. `maxim/moeTrainer.ipynb` ya fija una versión compatible y volvió a funcionar correctamente. Las corridas deben conservar ese pin para que una actualización de Colab no cambie el entorno.

## Diagnóstico del modelo

### Falta de especialización explícita

En la implementación original los cinco expertos se combinaban de forma densa y recibían gradiente en casi todos los ejemplos. Los comentarios `denoise`, `deblur`, etc. no imponían una especialidad aunque el dataset ya producía `task_id`.

La prueba causal con routing oráculo ya confirmó que la asignación tarea→experto funciona y que las cabezas pueden especializarse, especialmente en `dehaze` y `denoise`. El fine-tuning de LR bajo produjo sólo cambios marginales, por lo que la optimización no explica por sí sola el rezago de `deblur`, `derain` y `enhance`; la capacidad de las cabezas pasa a ser la siguiente hipótesis. Si el oracle con cabezas más capaces es bueno pero el router aprendido no lo reproduce, el cuello de botella estará en el clasificador/router.

### Pérdida de supervisión profunda

MAXIM original genera predicciones en varias etapas y escalas. La variante MoE previa usaba sólo las características finales del decoder y calculaba L1 sobre una única reconstrucción, mezclando la incorporación del MoE con la eliminación de pérdidas auxiliares. Esta diferencia ya fue corregida.

Se descartó el baseline monocabecera: la implementación final reincorporó directamente la supervisión profunda al MoE. La comparación relevante es MAXIM original frente a MAXIM+MoE con las mismas salidas auxiliares, la misma ponderación de pérdidas y el mismo protocolo. Así, la diferencia principal es la reconstrucción final mediante expertos y router.

### Regularización del router

La configuración original del router aprendido usa aproximadamente:

$$
\mathcal{L} = \mathcal{L}_{\text{tarea}}
- 10^{-3}\,\mathcal{H}(p)
+ 10^{-2}\,\mathcal{L}_{\text{balance}},
$$

con temperatura `1.5`. La entropía, el balance y la temperatura favorecen distribuciones suaves. Con batch size 2, el balance por batch puede ser ruidoso y empujar al router hacia `0.2` para cada experto. En las corridas oracle ambos pesos fueron cero, por lo que esta regularización no intervino en sus resultados.

En la fase de routing aprendido no se cambiará el signo de la entropía como primera prueba. Primero se medirá el efecto de quitarla o reducirla, usando las métricas ya implementadas.

## Plan de trabajo actualizado

### 1. Cerrar la auditoría de datos

1. [x] Terminar el diagnóstico automático de `derain`.
2. [x] Revisar visualmente las propuestas y descartar los falsos positivos heavy.
3. [x] Aplicar las 200 correspondencias light confirmadas, con backup, manifiesto y verificación posterior.
4. [x] Corregir manualmente `enhance/a3291-LS051026_day_2_arive38.png`.
5. [x] Confirmar como duplicado `enhance/train/a2931-jn_20081025_Kent_Shelter_413.png` y eliminar sus archivos y su entrada de `train.txt`.
6. [x] Cerrar la auditoría sin repetirla, aceptando los datos recolectados y documentando las decisiones de dominio sobre `deblur` y `dehaze`.

Criterio de finalización: todos los pares quedan contabilizados; cada excepción está corregida o aceptada explícitamente.

Estado: cumplido. La auditoría de datasets queda cerrada con 22.496 pares utilizables.

### 1.b. Ampliar `denoise` con SIDD Full

1. [x] Identificar los archivos sRGB noisy/GT de las 160 escenas públicas.
2. [x] Seleccionar 34 escenas distribuidas, equivalentes a 5.100 pares.
3. [x] Preparar descarga paralela, reanudable, con manifiesto y validación por ZIP.
4. [x] Corregir la concurrencia con lock exclusivo y reducirla de ocho a cuatro workers.
5. [ ] Descargar y procesar progresivamente las 18 escenas restantes; 16/34 completadas.
6. [ ] Normalizar todos los pares al esquema común y liberar cada archivo comprimido ya procesado; 2.400 pares completados.
7. [x] Crear un split por escena y generar incrementalmente los listados de entrenamiento y prueba.
8. [ ] Ejecutar las comprobaciones estructurales finales sobre los 5.100 pares.
9. [ ] Subir la versión normalizada a Drive e incorporarla a una nueva corrida.

Estado: en curso. Hay 2.400 pares normalizados (`2.100 train + 300 test`) y el proceso fue reanudado después del reinicio del equipo.

### 2. Restaurar la supervisión multietapa y multiescala

Modificar la interfaz de MAXIM para exponer las características finales y las salidas auxiliares. Para MAXIM S-2 con tres escalas se busca conservar:

- tres salidas auxiliares de la primera etapa;
- dos salidas de menor resolución de la segunda etapa;
- una salida final de resolución completa producida por el MoE.

La pérdida inicial será:

$$
\mathcal{L}_{\text{total}} =
\mathcal{L}_{\text{MoE-final}}
+ \lambda_{\text{aux}}
\sum_{(s,i)\neq(\text{final},\text{full})}
w_{s,i}\,\mathcal{L}_{1}(\hat{y}_{s,i}, y_i)
+ \alpha\,\mathcal{L}_{\text{router}}.
$$

Se debe normalizar la suma auxiliar para que no domine la reconstrucción final.

Implementación iniciada el 6 de septiembre y validada el 7 de septiembre de 2026:

- `MAXIM(..., return_features=True)` devuelve la jerarquía completa de predicciones junto con las características finales del decoder;
- `MaximMoE` conserva las salidas de todas las etapas y escalas y reemplaza únicamente `outputs[-1][-1]` por la reconstrucción mezclada de los expertos;
- para S-2 con tres escalas se preservan cinco salidas auxiliares y una salida final MoE;
- la pérdida separa `final_loss` del promedio ponderado y normalizado `auxiliary_loss`;
- `AUXILIARY_LOSS_WEIGHT` controla explícitamente el aporte auxiliar y comienza en 1.0;
- entrenamiento, validación, JSON y logs conservan por separado pérdida total, reconstrucción, salida final y auxiliares;
- la evaluación externa extrae correctamente la nueva salida final anidada;
- la estructura del notebook, la sintaxis y la fórmula de ponderación pasaron validaciones locales;
- el smoke test real en Colab produjo las seis salidas esperadas de S-2 con formas `64²`, `128²` y `256²` en ambas etapas;
- una corrida corta de 15 épocas y dos pasos por época completó inicialización, forward, backward, validación, métricas, heatmaps y checkpoints sin excepciones ni errores de memoria;
- `final_loss` y `auxiliary_loss` permanecieron finitas y descendieron durante la prueba; los `NaN` visibles corresponden únicamente a métricas por tarea sin muestras en los pequeños lotes de diagnóstico;
- el cargador confirmó los conteos corregidos, incluido `enhance/train` con 5.469 pares presentes y ninguna ruta faltante.

Pruebas mínimas completadas:

- seis salidas con formas correctas;
- pérdidas finitas en forward y backward;
- gradientes en backbone, salidas auxiliares, expertos y router según la fase;
- entrenamiento corto sin `NaN` ni errores de memoria;
- checkpoint reproducible.

Criterio de finalización: el MoE reemplaza sólo la salida final y todas las salidas auxiliares seleccionadas contribuyen a la pérdida.

Estado: cumplido. La supervisión profunda queda lista para utilizarse en los experimentos de routing.

### 3. Especializar expertos con routing oráculo

Convertir `task_id` en una asignación one-hot y enviar cada muestra a su experto. Mantener un backbone compartido y retirar temporalmente entropía y balance.

Antes de entrenar, verificar que el orden real de `TASK_DIRS` coincida con el significado de cada `task_id` y con la lectura de los heatmaps.

Mapeo canónico de `TASK_DIRS` y expertos: `0=deblur`, `1=dehaze`, `2=denoise`, `3=derain`, `4=enhance`.

Implementación realizada el 7 de septiembre de 2026:

- `MaximMoE` acepta `routing_mode="oracle"` y exige un `task_id` por imagen;
- en modo oráculo genera directamente `jax.nn.one_hot(task_id, 5)` y no ejecuta ni inicializa el router aprendido;
- el notebook pasa `task_id` por inicialización, entrenamiento, validación e inferencia manual;
- las pérdidas de entropía y balance conservan sus métricas diagnósticas, pero sus pesos son cero y no afectan la optimización;
- el orden de las cabezas quedó ligado directamente a `TASK_NAMES`, evitando comentarios semánticos inconsistentes;
- se agregó `routing_accuracy`, que debe ser exactamente 100 % en esta fase, además de las matrices tarea × experto;
- los checkpoints se guardan en `moe_oracle_training`, separados de las corridas de routing aprendido;
- la sintaxis Python de todas las celdas y la estructura JSON del notebook se validaron localmente antes de la prueba funcional en Colab.

Smoke test validado en Colab el 7 de septiembre de 2026, usando 15 épocas, dos pasos por época y batch size 2:

- la corrida completó las 15 épocas, forward, backward, validación, checkpoints, JSON y heatmaps sin excepciones;
- `routing_mode` fue `oracle` y los pesos de entropía y balance fueron 0.0 en los 15 JSON;
- train y validación mantuvieron `routing_accuracy=100 %`, entropía 0.0 y confianza 1.0 en todas las épocas;
- los heatmaps inicial y final fueron estrictamente diagonales para todas las tareas observadas;
- las pérdidas total, final y auxiliar y el PSNR permanecieron finitos;
- `loss == reconstruction_loss` en todas las épocas, confirmando que las métricas de balance y entropía no modificaron el objetivo;
- la pérdida de validación bajó de 64.4913 a 11.3300 y el PSNR subió de -34.79 a -19.79 dB, suficiente para confirmar flujo de gradientes, no para medir calidad final;
- las 60 muestras de entrenamiento del smoke test incluyeron las cinco tareas: 13 deblur, 14 dehaze, 6 denoise, 7 derain y 20 enhance;
- por el corte `MAX_EVAL_BATCHES=5` y el muestreo proporcional, cada validación sólo vio nueve muestras de deblur y una de derain. Los `NaN` de las otras filas indican ausencia de muestras, no un fallo del routing.

La corrida real se configuró con:

- MAXIM S-2 desde cero, `dropout_rate=0.1`, batch size 2, 30 épocas y 2.000 pasos por época, igual que la corrida MAXIM S-2 de comparación;
- 60.000 pasos de optimización totales, warmup de tres épocas, learning rate `2e-4`, AdamW y weight decay `1e-4`;
- entrenamiento con muestreo uniforme entre tareas;
- validación completa de los 3.696 pares, con batch size 1 y center crop determinista de `256 × 256`;
- concatenación determinista de los cinco datasets de validación para garantizar cobertura exhaustiva;
- métricas agregadas y por tarea para PSNR y L1 final, además de los diagnósticos de routing;
- abortar la corrida si validación no procesa exactamente 3.696 muestras o alguna tarea queda ausente;
- protección contra sobrescritura accidental de checkpoints existentes;
- preflight automático de una muestra para comprobar pérdidas finitas y routing correcto antes de iniciar los 60.000 pasos;
- duplicación de stdout y stderr del loop principal en `training.log`: el progreso sigue visible en Colab y se persiste en Drive, anexando una cabecera por cada sesión o reanudación;
- salida separada en `moe_oracle_all_S-2_scratch`;
- outputs del smoke test retirados del notebook para que la próxima salida corresponda inequívocamente a la corrida real.

#### Primera corrida real y detección del error de muestreo

La primera corrida completó las 30 épocas y obtuvo `25,381 dB` de PSNR global en validación. Sin embargo, los conteos de train de la época 30 fueron `[1247, 125, 7, 1316, 1305]`: `dehaze` y especialmente `denoise` casi no participaron.

La causa fue que se repetía el dataset ya mezclado. Cuando una fuente pequeña se agotaba, desaparecía del muestreo y la distribución que debía ser uniforme se volvía dependiente del tamaño de cada dataset. Se corrigió repitiendo cada fuente individual antes de `sample_from_datasets`. Se retiraron además los controles `UNIFORM_SAMPLING_MAX_RELATIVE_DEVIATION` y `EPOCH_DEFINITION`: si `STEPS_PER_EPOCH_OVERRIDE` está definido se usa ese presupuesto y, en caso contrario, la longitud de época mantiene el comportamiento proporcional esperado.

#### Segunda corrida real con muestreo uniforme corregido

La corrida corregida también completó 30 épocas y 60.000 pasos. En las 120.000 muestras observadas durante todo el entrenamiento, los conteos por tarea fueron:

| Tarea | Muestras train | Porcentaje aproximado |
|---|---:|---:|
| deblur | 23.875 | 19,90 % |
| dehaze | 23.864 | 19,89 % |
| denoise | 23.899 | 19,92 % |
| derain | 24.032 | 20,03 % |
| enhance | 24.330 | 20,28 % |
| **Total** | **120.000** | **100 %** |

La época 30 también quedó equilibrada: `[843, 802, 766, 801, 788]` sobre 4.000 muestras. Por lo tanto, el desbalance de exposición entre tareas queda descartado como explicación principal de las diferencias finales de calidad.

Resultados de validación al terminar ambas corridas:

| Tarea | Primera corrida | Muestreo corregido | Diferencia |
|---|---:|---:|---:|
| deblur | 24,989 dB | 24,635 dB | -0,354 dB |
| dehaze | 33,881 dB | 41,782 dB | +7,901 dB |
| denoise | 31,461 dB | 39,036 dB | +7,575 dB |
| derain | 26,259 dB | 26,887 dB | +0,628 dB |
| enhance | 23,255 dB | 21,938 dB | -1,317 dB |
| **Global ponderado** | **25,381 dB** | **25,384 dB** | **+0,003 dB** |

El routing permaneció estrictamente oráculo: accuracy 100 %, confianza 1, entropía 0 y matrices tarea × experto diagonales. La corrección produjo mejoras muy grandes en `dehaze` y `denoise`, una mejora menor en `derain` y ningún avance en `deblur` y `enhance`. El PSNR global casi no cambió porque la validación está dominada por los 3.150 ejemplos de `deblur`; en adelante no debe interpretarse sin la tabla por tarea.

Estado del hito: **cumplido**. El routing oráculo funciona, todas las tareas reciben una cantidad comparable de actualizaciones y las cabezas muestran capacidad de especialización. El fine-tuning posterior descartó que una optimización más suave sea suficiente; la capacidad de los expertos queda como hipótesis activa.

#### Fine-tuning con learning rate reiniciado

Para probar la hipótesis de que el learning rate de la corrida desde cero limita el ajuste final, se creó `maxim/moeFineTune.ipynb`, reutilizando la estructura y las funciones auxiliares ya verificadas.

El notebook:

- restaura el mejor checkpoint oracle de la corrida con muestreo corregido;
- conserva los parámetros y `batch_stats` del modelo;
- descarta los momentos anteriores de AdamW y reinicia `state.step` en cero;
- ejecuta cinco épocas adicionales de 2.000 pasos;
- usa cosine decay sin warmup desde `1e-5` hasta `1e-6`;
- escribe checkpoints, métricas, diagnósticos y log en un directorio nuevo para no sobrescribir la corrida base.

La corrida terminó correctamente el 9 de septiembre de 2026. Completó 10.000 pasos adicionales, las cinco validaciones exhaustivas de 3.696 muestras y conservó routing accuracy de 100 %, confianza 1 y entropía 0.

Comparación entre el checkpoint de partida y la época 5, seleccionada como mejor por PSNR global:

| Tarea | Oracle de partida | Fine-tuning época 5 | Diferencia |
|---|---:|---:|---:|
| deblur | 24,635 dB | 24,660 dB | +0,025 dB |
| dehaze | 41,782 dB | 42,056 dB | +0,274 dB |
| denoise | 39,036 dB | 39,079 dB | +0,043 dB |
| derain | 26,887 dB | 26,982 dB | +0,095 dB |
| enhance | 21,938 dB | 21,756 dB | -0,183 dB |
| **Global ponderado** | **25,384 dB** | **25,422 dB** | **+0,038 dB** |

La pérdida total de validación bajó de `0,08267` a `0,08113` (-1,86 %), pero casi toda la reducción provino de la pérdida auxiliar (-3,15 %); la pérdida de la salida final sólo bajó 0,37 %. Esto confirma que reiniciar el optimizador y reducir el learning rate es estable, pero no resuelve el límite de calidad de las tareas rezagadas.

El mejor punto no es idéntico para todas las tareas. `enhance` llegó a `22,309 dB` en la época 2 (+0,371 dB frente al inicio) y luego cayó a `21,756 dB`, mientras el criterio global eligió la época 5. Esto evidencia que seleccionar checkpoints sólo por PSNR ponderado —dominado por `deblur`— puede ocultar regresiones en tareas pequeñas. Las próximas corridas deben registrar también un criterio macro o checkpoints por tarea, aunque la comparación principal continúe informando todas las métricas por separado.

Estado del hito: **completado**. El learning rate influye ligeramente, pero no es el cuello de botella principal. No se justifica prolongar este mismo fine-tuning esperando una mejora grande.

Criterio de decisión:

- conservar la época 5 como mejor checkpoint global de esta prueba y la época 2 como referencia del comportamiento de `enhance`;
- aumentar cada cabeza a dos o tres convoluciones con conexión residual y repetir primero un smoke test y luego una comparación oracle corta;
- registrar selección global ponderada, resumen macro y mejor época por tarea para evitar que `deblur` determine por sí solo el checkpoint;
- no atribuir mejoras de `denoise` a la expansión de SIDD hasta entrenar explícitamente con el dataset nuevo;
- después de caracterizar el oracle con cabezas más capaces, avanzar al router supervisado usando ese modelo como referencia superior de routing.

### 3.b. Aumentar la capacidad de los expertos

1. [ ] Diseñar una cabeza de dos o tres convoluciones con conexión residual, conservando tres canales de salida y la interfaz actual del MoE.
2. [ ] Mantener una variante seleccionable de una sola convolución para que la ablación sea directa.
3. [ ] Registrar cantidad adicional de parámetros y memoria para separar capacidad de costo computacional.
4. [ ] Ejecutar un smoke test oracle que compruebe formas, pérdidas finitas, gradientes y checkpoints.
5. [ ] Ejecutar primero una comparación corta con los mismos datos y muestreo uniforme.
6. [ ] Si la señal es positiva, realizar una corrida comparable y medir especialmente `deblur`, `derain` y `enhance`.

Criterio de decisión: avanzar al router supervisado con la arquitectura oracle que ofrezca el mejor compromiso entre calidad por tarea y costo. Si las cabezas profundas tampoco mejoran, revisar la cantidad de características expuestas por el decoder o permitir adaptación específica por tarea antes de aumentar más el entrenamiento.

### 4. Entrenar el router como clasificador

Congelar inicialmente backbone y expertos. Entrenar el router con cross-entropy sobre las cinco tareas y registrar:

- accuracy global y por tarea;
- matriz de confusión tarea × predicción;
- entropía y confianza;
- matrices tarea × experto ya implementadas.

Criterio de finalización: el router reconoce las tareas de validación con una asignación estable y sin monopolio global de un experto.

### 5. Ajustar el modelo completo

Inicializar con los expertos y el router de las fases anteriores. Descongelar el modelo y combinar reconstrucción, pérdidas auxiliares y supervisión del router. Evaluar una transición gradual desde routing oráculo hacia routing aprendido.

Una formulación inicial es:

$$
\mathcal{L}_{\text{router}} =
\mathcal{L}_{\text{CE-task}}
+ \lambda_{\text{bal}}\mathcal{L}_{\text{balance}}
- \lambda_{\text{ent}}\mathcal{H}(p).
$$

Los términos de balance y entropía se conservarán sólo si mejoran reconstrucción y especialización.

### 6. Ejecutar ablaciones cortas

Todas las variantes deben usar la misma semilla, split, presupuesto y protocolo:

| Variante | $\lambda_{ent}$ | $\lambda_{bal}$ | Temperatura | Propósito |
|---|---:|---:|---:|---|
| Control actual | $10^{-3}$ | $10^{-2}$ | 1.5 | Reproducir el comportamiento existente |
| Sin regularización | 0 | 0 | 1.0 | Medir el efecto neto de ambos términos |
| Balance débil | 0 | $10^{-3}$ | 1.0 | Evitar monopolio sin forzar mezcla uniforme |
| Router más selectivo | 0 | $10^{-3}$ | 0.5 | Favorecer asignaciones marcadas |

La selección usará PSNR/SSIM de validación junto con las métricas de especialización. Un heatmap marcado no compensa una peor reconstrucción.

### 7. Ejecutar la corrida final y actualizar la tesis

La corrida final con routing aprendido comenzará después de cerrar el nuevo dataset de `denoise`, seleccionar la mejor configuración oracle y validar el router supervisado. Debe producir:

- checkpoints restaurables;
- PSNR y SSIM globales y por tarea;
- curvas de train y validación;
- heatmaps tarea × experto;
- matriz de confusión del router;
- tabla de ablaciones;
- comparación con MAXIM original bajo el mismo protocolo de supervisión.

## Métricas y criterios de éxito

### Calidad de reconstrucción

- PSNR y SSIM globales y por tarea;
- mismo split, crop, seed y presupuesto en todas las comparaciones;
- evolución de train y validación para detectar sobreajuste;
- el oracle fine-tuned alcanzó `25,422 dB` globales, pero este valor está ponderado en un 85,2 % por `deblur` y no reemplaza la comparación por tarea;
- conservar las mejoras fuertes ya observadas en `dehaze` (`42,056 dB`) y `denoise` (`39,079 dB`);
- alcanzar o superar la referencia comparable por tarea, con atención prioritaria a `deblur`, `derain` y `enhance`;
- informar tanto el promedio ponderado por muestras como un resumen macro por tarea para que el tamaño del split no oculte regresiones.

### Especialización

- matriz tarea × experto no uniforme;
- frecuencia top-1 por experto y por tarea;
- entropía menor que `ln(5)`, sin exigir que sea cercana a cero;
- confianza compatible con la accuracy del router;
- ningún experto domina todas las tareas;
- salidas o parámetros de los expertos muestran diversidad medible.

### Estabilidad técnica

- pérdidas auxiliares y final finitas;
- gradientes en los componentes esperados;
- checkpoints reproducibles;
- inferencia con dimensiones válidas;
- ausencia de errores de memoria en la configuración elegida;
- versiones del entorno registradas en cada corrida.

## Riesgos y decisiones

- **GT incorrectos en `derain` — mitigado:** se repararon y verificaron los 200 GT light afectados mediante hashes e inspección visual. Se decidió no repetir la auditoría completa.
- **Expertos demasiado simples — hipótesis activa:** cada cabeza es una única convolución `3 × 3` y el fine-tuning de LR bajo no recuperó las tareas rezagadas. El próximo experimento probará dos o tres convoluciones con residual.
- **Selección de checkpoint sesgada por tamaño:** el PSNR global está dominado por `deblur`; en el fine-tuning eligió la época 5 aunque `enhance` había alcanzado su mejor valor en la época 2. Registrar además criterio macro y mejores épocas por tarea.
- **Split de SIDD y fuga de contenido:** los 150 pares de una misma escena están fuertemente relacionados; train/test debe separarse por escena y no por imagen.
- **Volumen de SIDD:** los 5.100 pares seleccionados ocupan ~197,63 GiB comprimidos. La normalización debe hacerse por escena, liberando archivos procesados para no agotar el disco.
- **Atajos por dataset:** el router puede reconocer cámara, compresión o colorimetría en vez de degradación. Evaluar ejemplos cruzados cuando sea posible.
- **Balance ruidoso:** con batch size 2, considerar acumulación temporal o un batch efectivo mayor.
- **Pérdidas auxiliares dominantes:** controlar `lambda_aux` y los pesos por escala.
- **Entorno cambiante:** las dependencias abiertas de Colab pueden romper una corrida reproducible.
- **Costo de entrenamiento:** decidir con smoke tests y corridas cortas antes de reservar un entrenamiento completo.

## Material para la próxima reunión

- resumen final de la auditoría y corrección de `derain`;
- estado de las métricas del router con un JSON y heatmap real;
- esquema de MAXIM+MoE con supervisión auxiliar;
- definición de la pérdida;
- resultados de las dos corridas oracle y evidencia del muestreo uniforme corregido;
- resultado del fine-tuning con learning rate reiniciado: mejora global marginal y diferencias por tarea;
- estado de la ampliación de `denoise` con SIDD Full;
- tabla breve de ablaciones;
- siguiente experimento completo con configuración cerrada.
