# Plan de mejora de MAXIM + Mixture of Experts

Última actualización: 28 de septiembre de 2026.

## Objetivo

Comparar MAXIM S-2 con MAXIM+MoE bajo el mismo dataset, supervisión multietapa/multiescala, presupuesto de optimización y protocolo de evaluación. El MoE debe producir especialización observable sin sacrificar la ruta común de reconstrucción.

Las siguientes iteraciones se harán en tres bloques, en este orden:

1. corregir y hacer explícitas las métricas y el protocolo de selección de checkpoints;
2. convertir la branch K=8 en un MoE token-choice real, con routing independiente para cada token de features de MAXIM;
3. actualizar la variante con cabeza compartida para inicializarla desde un checkpoint baseline y compararla contra una continuación equivalente del baseline.

## Estado actual

| Bloque | Estado | Resultado / acción |
|---|---|---|
| Auditoría y corrección de datos | Completado | Pares problemáticos corregidos o aceptados por criterio de dominio |
| Ampliación SIDD `denoise` | Completado | 5.060 pares nuevos normalizados, combinados y verificados en Drive |
| Supervisión profunda del MoE | Completado | Cinco salidas auxiliares y una salida final MoE para S-2 |
| Muestreo uniforme de entrenamiento | Completado | Cada tarea recibe aproximadamente 20 % de exposición |
| Baseline MAXIM comparable | Completado | Mejor checkpoint: 27,455 dB ponderado y 30,822 dB macro |
| MoE oracle K=5 simple | Completado | 27,362 dB ponderado y 30,547 dB macro |
| MoE oracle K=5 residual | Completado | 27,114 dB ponderado y 30,980 dB macro |
| MoE K=5 con cabeza compartida | Completado | 27,350 dB ponderado y 30,720 dB macro |
| MoE latente K=8 top-2 | Completado | 27,060 dB ponderado y 30,800 dB macro; cuatro de ocho expertos quedaron inactivos |
| Corrección del protocolo de métricas | Completado | Ponderado/macro explícitos, selección por validación, test final separado y splits agrupados verificados en Drive |
| MoE token-choice K=8 sobre features | Implementado; smoke funcional aprobado | Router y gates `[B,N,K]`, expertos `C→C`, top-2 por token, balance denso, exploración y warm-up denso; falta smoke training con MAXIM real |
| Cabeza compartida inicializada desde baseline | Pendiente | Cargar parámetros baseline, inicializar residuales en cero y entrenar con control baseline de igual presupuesto adicional |

## Datasets finales

| Tarea | Train | Evaluación actual | Total |
|---|---:|---:|---:|
| deblur | 8.680 | 3.150 | 11.830 |
| dehaze | 909 | 101 | 1.010 |
| denoise | 4.450 | 767 | 5.217 |
| derain | 3.600 | 400 | 4.000 |
| enhance | 5.469 | 30 | 5.499 |
| **Total** | **23.108** | **4.448** | **27.556** |

No se debe balancear la evaluación descartando imágenes de `deblur` ni duplicando imágenes de tareas pequeñas. Se conservarán todas las muestras y se reportarán dos agregados diferentes:

- **PSNR ponderado:** promedio de los PSNR por tarea ponderado por la cantidad de imágenes de cada tarea; equivale al promedio por imagen con la implementación actual.
- **PSNR macro:** promedio de los cinco PSNR por tarea; asigna 20 % a cada tarea.

Los resultados por tarea y sus conteos son obligatorios. Para `enhance`, cuyo conjunto actual tiene sólo 30 imágenes, se debe informar además que la estimación tiene mayor incertidumbre.

## Evidencia experimental acumulada

| Modelo | Deblur | Dehaze | Denoise | Derain | Enhance | Ponderado | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAXIM baseline | 24,554 | 42,824 | **38,087** | 26,428 | 22,216 | **27,455** | 30,822 |
| MoE K=5 simple oracle | **24,659** | 41,934 | 36,970 | **26,931** | 22,240 | 27,362 | 30,547 |
| MoE K=5 residual oracle | 24,611 | **45,757** | 35,568 | 26,244 | **22,720** | 27,114 | **30,980** |
| MoE K=5 compartida + residual | 24,610 | 42,660 | 37,060 | 26,780 | 22,490 | 27,350 | 30,720 |
| MoE K=8 residual aprendido top-2 | 24,230 | 45,200 | 37,190 | 25,740 | 21,640 | 27,060 | 30,800 |

### Conclusión de la corrida K=8 actual

- El router encontró asociaciones sin usar `task_id` para seleccionar expertos.
- `deblur` y `dehaze` se concentraron en el par `E1/E4`.
- `denoise` y `derain` se concentraron en el par `E2/E6`.
- `enhance` mantuvo routing más disperso.
- `E0`, `E3`, `E5` y `E7` quedaron con uso cero: K=8 se comportó en la práctica como cuatro expertos organizados en dos pares.
- El valor `expertos efectivos=2` del log mide diversidad por imagen y es esperable con top-2; no mide cuántos expertos se usan globalmente.
- El K=8 actual usa cabezas residuales independientes, no la cabeza compartida. Por eso no debe interpretarse como una ablación exclusiva de `K` o del router.
- La corrida queda como control histórico. No continuarla desde su checkpoint después de cambiar el router o la arquitectura.

## Bloque 1 — Corregir métricas y protocolo de evaluación

### Cambios de implementación

1. Reemplazar el campo ambiguo `psnr` por nombres explícitos:
   - `psnr_weighted`;
   - `psnr_macro`;
   - `task_psnr`;
   - `task_count`.
2. Mantener el PSNR por imagen actual y agregarlo por tarea mediante sumas y conteos.
3. Calcular:

   `psnr_weighted = sum(task_psnr[t] * task_count[t]) / sum(task_count)`

   `psnr_macro = mean(task_psnr[t] for t in tareas_presentes)`

4. No calcular el macro a partir del PSNR ponderado.
5. Guardar en cada JSON de época ambos agregados, los cinco resultados por tarea, los conteos y el criterio usado para seleccionar el checkpoint.
6. Cambiar el checkpoint principal para seleccionar por `psnr_macro` de validación. Guardar opcionalmente un segundo checkpoint `best_weighted_checkpoint` para análisis.
7. Actualizar `best_metric.json` para incluir como mínimo:
   - época;
   - `selection_metric`;
   - `psnr_macro`;
   - `psnr_weighted`;
   - `task_psnr`;
   - `task_count`.
8. Aplicar exactamente la misma implementación en baseline, MoE K=5, K=8 y cabeza compartida.

### Validación y test

El notebook actual lee `test.txt` después de cada época y usa ese valor para elegir el mejor checkpoint. Metodológicamente ese conjunto está funcionando como validación.

Se debe:

1. agregar soporte explícito para `train.txt`, `val.txt` y `test.txt`;
2. usar `val.txt` durante entrenamiento y para seleccionar checkpoints;
3. evaluar `test.txt` sólo una vez con el checkpoint ya elegido;
4. evitar fugas por escena al construir `val.txt`;
5. registrar conteos y SHA-256 de las tres listas.

Decisión práctica: no cambiar silenciosamente los splits dentro del código. Si todavía no existe `val.txt`, el notebook debe detenerse con un mensaje claro. La creación del nuevo split debe hacerse con un script determinista, agrupado por escena y versionado. Si el nuevo `val.txt` se extrae del train actual, las comparaciones finales estrictas requerirán volver a entrenar el baseline y las variantes seleccionadas con ese mismo split.

Los resultados históricos se conservarán bajo el protocolo anterior y no se mezclarán en una misma tabla final con resultados obtenidos bajo un split nuevo.

### Pruebas de aceptación

- Test sintético con tareas de tamaños distintos que confirme los valores ponderado y macro esperados.
- Con las métricas de la época 27 K=8, el código debe reproducir aproximadamente `27,06 dB` ponderado y `30,80 dB` macro.
- El log debe imprimir claramente `validation ponderado`, `validation macro` y el criterio de guardado.
- Ninguna variable denominada `test_dataset` debe usarse dentro del bucle por época.
- La evaluación final de test no debe actualizar ni reemplazar checkpoints.

## Bloque 2 — MoE token-choice K=8 sobre features

Branch de trabajo: `exp/latent-experts-k8`.

### Cambio arquitectónico

El router actual recibe una imagen RGB completa, aplica global average pooling y produce un único vector `[B, K]`. Esa implementación es routing por imagen y debe reemplazarse, no adaptarse.

La nueva arquitectura debe realizar token-choice exclusivamente en el espacio de features:

1. ejecutar `MAXIM(..., return_features=True)`;
2. tomar el feature map del último decoder antes de la reconstrucción final, con forma `[B, H, W, C]`;
3. interpretarlo como `N = H × W` tokens de dimensión `C`, obteniendo `[B, N, C]`;
4. aplicar el mismo router liviano a cada token, sin mezclar tokens y sin global pooling;
5. producir logits `[B, N, K]` y seleccionar `TOP_K=2` expertos de forma independiente para cada token;
6. aplicar expertos de features con contrato `R^C → R^C` a los tokens seleccionados;
7. combinar las salidas top-2 con sus gates normalizados;
8. restaurar el feature map `[B, H, W, C]`;
9. usar conexión residual en features y luego una cabeza de reconstrucción compartida para generar RGB.

Forma conceptual:

`F [B,H,W,C] → tokens Z [B,N,C] → router [B,N,K] → top-2 token experts → Z' [B,N,C] → F' → shared output head → RGB`

No debe entrar al router:

- la imagen RGB;
- `task_id`;
- una representación promediada de toda la imagen.

Los expertos de esta branch deben operar en el espacio de features. Las cabezas convolucionales que producen una imagen RGB completa por experto no implementan token routing y no deben reutilizarse como si fueran expertos por token. La implementación recomendada es un MLP residual por experto, por ejemplo `Dense(C→rC) → GELU → Dense(rC→C)`, o su equivalente pointwise `1×1`.

Los tokens ya contienen contexto espacial extraído por MAXIM. El primer experimento debe permitir que el gradiente de reconstrucción atraviese el router y llegue al backbone; si aparece inestabilidad se agregará como ablación una opción `ROUTER_STOP_GRADIENT`, pero no debe quedar activada por defecto sin evidencia.

### Evitar nuevamente expertos muertos

El routing por token aumenta el conjunto de decisiones de balance desde `B` imágenes hasta `B × N` tokens, pero el hard top-k todavía puede dejar expertos muertos. En la misma branch se debe:

1. conservar las probabilidades densas `dense_router_probs` antes del top-k;
2. calcular la pérdida de balance sobre esas probabilidades densas, no sobre gates ya enmascarados;
3. devolver por separado `dense_router_probs` y `sparse_gates` para métricas;
4. agregar exploración durante entrenamiento mediante ruido pequeño en logits o noisy top-k;
5. implementar un warm-up configurable del routing: recomendado dos épocas densas o top-4 y luego top-2;
6. mantener top-2 determinista durante validación/test;
7. emitir una advertencia si un experto permanece por debajo de 1 % de los tokens durante tres épocas consecutivas;
8. si se implementa capacidad finita por experto, registrar `capacity_factor`, tokens descartados y política de overflow.

Aunque el batch de imágenes sea 2, el balance ahora se estima sobre todos los tokens del batch. La agregación y la normalización deben usar el número real de tokens, no sólo la cantidad de imágenes.

### Métricas del router

Registrar por separado:

- uso soft denso global por token;
- carga sparse global por token;
- frecuencia de inclusión top-2 por token, no sólo frecuencia top-1;
- matriz tarea × experto agregando tokens, sólo para análisis posterior;
- entropía por token y promedio por imagen;
- entropía de la distribución global media;
- expertos efectivos por token y expertos efectivos globales;
- porcentaje de tokens asignados a cada par top-2;
- expertos con gradiente y norma media del gradiente por época.

Para demostrar especialización más allá de reconocer la tarea, agregar análisis dentro de cada tarea: posición espacial, magnitud/energía del feature, bordes/textura y severidad de degradación cuando exista esa metadata.

### Eficiencia

La implementación semántica mínima puede evaluar los ocho expertos sobre todos los tokens y aplicar luego los gates top-2. Eso permite validar token routing, pero no reduce cómputo. La versión verdaderamente sparse debe despachar/gather tokens por experto y recomponerlos con scatter, usando shapes estáticos compatibles con JAX y una política explícita de capacidad/overflow. No mezclar la optimización de ejecución con la primera validación funcional.

### Pruebas de aceptación

- El router debe recibir exclusivamente tokens `[B,N,C]` derivados del feature map.
- No debe existir global average pooling sobre `H×W` antes del router.
- Shapes correctos para batch 1 y batch 2 y para al menos dos tamaños espaciales válidos.
- Los logits y gates deben tener forma `[B,N,K]`, nunca `[B,K]`.
- `dense_router_probs` suma 1 y tiene ocho valores positivos por token.
- `sparse_gates` suma 1 y tiene exactamente `TOP_K` valores no nulos por token.
- El balance denso debe producir gradiente para los ocho logits del router.
- Durante un smoke test deben observarse gradientes finitos y no nulos en el router y en los expertos seleccionados.
- Dos tokens de una misma imagen deben poder seleccionar pares de expertos distintos en un test controlado.
- La salida reconstruida debe conservar la resolución y el contrato RGB de MAXIM.
- Reiniciar desde cero en un `OUTPUT_DIR` nuevo; no restaurar el checkpoint K=8 colapsado.

## Bloque 3 — Cabeza compartida inicializada desde baseline

Este bloque se implementará en el notebook de cabeza compartida sobre `main` o una branch nueva derivada de `main`, sin mezclar todavía los cambios experimentales del router K=8.

### Inicialización

1. Elegir un checkpoint baseline fuente y registrar ruta, época, métrica y hash/configuración.
2. Cargar todos los parámetros compatibles del backbone y de las salidas auxiliares.
3. Copiar la convolución de salida baseline a la cabeza compartida.
4. Inicializar la última convolución de cada corrección residual en cero, de modo que inicialmente:

   `salida_MoE = salida_baseline + 0`

5. Comprobar con una entrada fija y `train=False` que la diferencia máxima entre baseline y MoE inicial sea menor que `1e-6`.
6. Fallar si la cobertura de parámetros cargados no coincide con la esperada; no aceptar warm starts parciales silenciosos.

### Comparación justa por presupuesto adicional

Definir una única constante `EXTRA_EPOCHS` antes de mirar los resultados. Valor recomendado inicial: **10 épocas**, equivalentes a 20.000 pasos adicionales.

Crear dos corridas desde exactamente el mismo checkpoint baseline:

- **Control baseline continuado:** baseline + `EXTRA_EPOCHS`.
- **MoE cabeza compartida:** baseline inicial + cabeza compartida/residuales + `EXTRA_EPOCHS`.

Para que la comparación sea limpia:

- restaurar los mismos parámetros fuente en ambas corridas;
- crear un optimizador nuevo en ambas, porque la arquitectura MoE no puede heredar de forma idéntica el estado completo del optimizador baseline;
- usar el mismo LR, schedule, warm-up, weight decay, orden de datos, seed y número de pasos;
- decidir y documentar si todo el backbone queda entrenable; por defecto, entrenar conjuntamente backbone, cabeza compartida y expertos;
- reportar tanto el resultado absoluto como la mejora o regresión respecto del checkpoint inicial.

No comparar directamente baseline de 30 épocas contra MoE con 30 épocas baseline más épocas adicionales. El control válido es baseline continuado durante el mismo presupuesto extra.

### Checkpoints y salidas

Usar directorios nuevos y separados, por ejemplo:

- `baseline_S-2_from_ckpt_extra10`;
- `moe_shared_residual_from_baseline_extra10`.

Guardar en ambos:

- checkpoint inicial evaluado;
- mejor checkpoint macro de validación;
- último checkpoint;
- métricas ponderada/macro y por tarea;
- configuración completa y procedencia del checkpoint fuente;
- tiempo por paso, tiempo de evaluación y memoria pico si está disponible.

### Pruebas de aceptación

- Equivalencia inicial baseline/MoE menor que `1e-6`.
- Conteo exacto de pasos adicionales igual en ambas corridas.
- Mismo split y mismo protocolo de métricas.
- Ningún checkpoint previo debe ser sobrescrito.
- La corrida debe poder reanudarse conservando correctamente `EXTRA_EPOCHS`, paso global y criterio de mejor checkpoint.

## Orden de trabajo para Codex

1. Crear una branch/commit común de métricas y tests. **Completado.**
2. Aplicar esas métricas al baseline, K=5, K=8 y cabeza compartida. **Completado.**
3. Implementar soporte `val/test` sin generar un split silenciosamente. **Completado y verificado en Drive.**
4. En `exp/latent-experts-k8`, implementar token-choice `[B,N,K]` sobre features, expertos `C→C`, balance denso y top-2 por token. **Implementado.**
5. Ejecutar tests unitarios y un smoke test corto del K=8; no lanzar todavía una corrida completa si aparecen expertos muertos. **Tests locales y smoke funcional JAX/Colab aprobados; falta el smoke training con MAXIM real.**
6. En la branch de cabeza compartida, implementar warm start exacto desde baseline.
7. Preparar los dos notebooks de continuación con `EXTRA_EPOCHS=10`: baseline control y MoE compartido.
8. Verificar equivalencia inicial, presupuesto y directorios de salida.
9. Sólo después de esos chequeos, lanzar las corridas completas.

## Criterios para corridas comparables

- Mismo split, seed, crop, augmentations, batch size, arquitectura S-2 y presupuesto de pasos.
- Muestreo uniforme entre tareas durante train.
- Validación exhaustiva y determinista, separada del test final.
- Mismos pesos de supervisión profunda.
- Cero pares faltantes y conteos exactos por tarea.
- Reportar PSNR ponderado, macro y por tarea; añadir SSIM bajo el mismo esquema cuando se implemente.
- Guardar entorno, commit, configuración, logs, checkpoints y diagnósticos reproducibles.
- Tratar diferencias menores a 0,1 dB como empate técnico mientras exista una sola seed.

## Riesgos y detalles a no olvidar

- La evaluación histórica usó `test.txt` para elegir checkpoints; debe declararse y no confundirse con un test final ciego.
- Crear un `val.txt` desde train cambia el protocolo y obliga a repetir las comparaciones finales bajo el nuevo split.
- El macro corrige la ponderación entre tareas, pero no la baja representatividad de `enhance` con 30 imágenes.
- El router por token puede seguir aprendiendo atajos de dataset; analizar asignaciones dentro de cada tarea y dentro de cada imagen, no sólo tarea × experto.
- Top-2 con hard mask puede dejar expertos sin gradiente; el balance pre-top-k y la exploración son obligatorios.
- Batch size 2 limita la diversidad de imágenes por batch, aunque el balance por token use `B×N` observaciones.
- La corrida K=8 actual calcula los ocho expertos aunque sólo mezcle dos.
- Hubo fallas transitorias leyendo imágenes desde Drive. Antes de corridas largas, copiar o cachear el dataset en almacenamiento local y validar decodificación.
- Reanudar una época desde checkpoint no reprodujo exactamente el mismo resultado; registrar estado del pipeline y seeds de TensorFlow/NumPy/JAX.
- Mantener versiones fijadas de JAX, Flax, Optax y TensorFlow en Colab.
- No mezclar cambios de métricas, router y cabeza compartida en una sola corrida sin controles intermedios.
