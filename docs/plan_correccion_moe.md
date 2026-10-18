# Plan de mejora de MAXIM + Mixture of Experts

Última actualización: 8 de octubre de 2026.

## Objetivo

Comparar MAXIM S-2 con MAXIM+MoE bajo el mismo dataset, supervisión multietapa/multiescala, presupuesto de optimización y protocolo de evaluación. El MoE debe producir especialización observable sin sacrificar la ruta común de reconstrucción.

Los bloques de PSNR/SSIM, separación train/val/test, token-choice K=8 y warm start de cabeza compartida están implementados. El baseline nuevo desde cero y ambas continuaciones de diez épocas finalizaron.

**Próximo paso autorizado: test final del baseline +10 y shared head +10**, con checkpoints fijados por PSNR macro de validación antes de consultar test. Usar el notebook `test_final_baseline_vs_shared.ipynb` entregado con este plan. No lanzar nuevas extensiones ni modificar la selección después de ver test.

Checkpoints principales fijados el 04/10/2026:

1. Baseline `baseline_S-2_from_newsplit_ckpt_extra10/best_checkpoint/checkpoint_10`.
2. Shared `moe_shared_residual_from_newsplit_baseline_extra10/best_checkpoint/checkpoint_8`.

Ambas continuaciones declaran como fuente el baseline nuevo S-2, mejor macro en época 30, entrenado desde cero con el nuevo split. Las corridas históricas quedan separadas de esta comparación.

## Estado actual

| Bloque | Estado | Resultado / acción |
|---|---|---|
| Auditoría y corrección de datos | Completado | Pares problemáticos corregidos o aceptados por criterio de dominio |
| Ampliación SIDD `denoise` | Completado | 5.060 pares nuevos normalizados, combinados y verificados en Drive |
| Supervisión profunda del MoE | Completado | Cinco salidas auxiliares y una salida final MoE para S-2 |
| Muestreo uniforme de entrenamiento | Completado | Cada tarea recibe aproximadamente 20 % de exposición |
| Baseline MAXIM histórico | Completado; referencia histórica | Mejor checkpoint: 27,455 dB ponderado y 30,822 dB macro bajo el protocolo anterior; repetición con nuevo split completada |
| MoE oracle K=5 simple | Completado | 27,362 dB ponderado y 30,547 dB macro |
| MoE oracle K=5 residual | Completado | 27,114 dB ponderado y 30,980 dB macro |
| MoE K=5 con cabeza compartida | Completado | 27,350 dB ponderado y 30,720 dB macro |
| MoE latente K=8 top-2 | Completado | 27,060 dB ponderado y 30,800 dB macro; cuatro de ocho expertos quedaron inactivos |
| Corrección del protocolo de métricas PSNR | Completado | Ponderado/macro explícitos, selección por validación y test final separado |
| Conteos del nuevo split train/val/test | Verificado el 02/10/2026 | 20.523 train, 2.303 val y 4.730 test; tabla completa contrastada con las listas de Drive |
| SSIM durante validación | Completado y presente en las tres runs nuevas | RGB, ventana 11, sigma 1,5; por tarea, macro y ponderado; selección sigue por PSNR macro |
| MoE token-choice K=8 sobre features | Corrida completa finalizada | 30 épocas / 60.000 pasos; mejor macro en época 30: 31,754 dB macro y 27,025 dB ponderado; ocho expertos activos al final |
| Cabeza compartida desde baseline histórico | Corrida finalizada; ablación exploratoria | +10 épocas / 20.000 pasos; mejor macro en época extra 6: 31,775 dB macro y 27,202 dB ponderado; repetición desde el nuevo checkpoint completada |
| Baseline desde cero con nuevo split | Completado | Mejor macro: época 30; 31,694158 dB macro / 27,230948 dB ponderado |
| Shared head +10 desde nuevo baseline | Completado | Mejor macro: época extra 8; 32,063355 dB macro / 27,420716 dB ponderado |
| Baseline +10 desde nuevo baseline | Completado | Mejor macro: época extra 10; 31,874760 dB macro / 27,220281 dB ponderado |
| Test final baseline +10 vs shared +10 | Notebook preparado; pendiente de ejecutar en Colab | Checkpoints fijados; 4.730 pares; PSNR/SSIM por tarea, macro y ponderado |

## Datasets y separación de protocolos

### Inventario histórico, anterior a la creación de val

Estos conteos corresponden al esquema anterior. No usarlos como conteos de entrenamiento del nuevo split.

| Tarea | Train histórico | Evaluación histórica | Total |
|---|---:|---:|---:|
| deblur | 8.680 | 3.150 | 11.830 |
| dehaze | 909 | 101 | 1.010 |
| denoise | 4.450 | 767 | 5.217 |
| derain | 3.600 | 400 | 4.000 |
| enhance | 5.469 | 30 | 5.499 |
| **Total** | **23.108** | **4.448** | **27.556** |

### Dataset después del nuevo split train/val/test — verificado en Drive el 2 de octubre de 2026

Los conteos se obtuvieron directamente de las 15 listas `train.txt`, `val.txt` y `test.txt`, contando una entrada no vacía por imagen/pareja de restauración. Son imágenes distintas incluidas en las listas, no crops ni muestras repetidas por el muestreo de entrenamiento.

| Tarea | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| deblur | 7.812 | 868 | 3.150 | 11.830 |
| dehaze | 561 | 66 | 383 | 1.010 |
| denoise | 3.987 | 463 | 767 | 5.217 |
| derain | 3.240 | 360 | 400 | 4.000 |
| enhance | 4.923 | 546 | 30 | 5.499 |
| **Total** | **20.523** | **2.303** | **4.730** | **27.556** |

Fuente: [carpeta Classifier con las cinco tareas y sus listas](https://drive.google.com/drive/folders/1BcjHsRHCKB5UGLg5g9zGopiM-rlGlyjA).

Los conteos de val coinciden con los reportados por shared head y token-choice. En las listas revisadas no hay entradas duplicadas dentro de un split ni nombres de imagen compartidos entre train/val/test de la misma tarea; esto no reemplaza la verificación previa de separación por escena. En `dehaze`, el split actual es 561/66/383, diferente de los 909/101 del esquema histórico: no calcular train nuevo restando únicamente val al train histórico. El total de pares se conserva en 27.556.

Conservar estas listas sin regenerarlas. Sus SHA-256 y conteos ya fueron verificados en las corridas controladas; repetir esa comprobación en toda corrida posterior, porque una igualdad de conteos no prueba identidad de las listas. Mantener el chequeo agrupado por escena ya previsto.

No se debe balancear la evaluación descartando imágenes de `deblur` ni duplicando imágenes de tareas pequeñas. Se conservarán todas las muestras y se reportarán dos agregados diferentes:

- **PSNR ponderado:** promedio de los PSNR por tarea ponderado por la cantidad de imágenes de cada tarea; equivale al promedio por imagen con la implementación actual.
- **PSNR macro:** promedio de los cinco PSNR por tarea; asigna 20 % a cada tarea.

SSIM ya está implementado **durante validación** bajo el mismo esquema: promedio por imagen dentro de cada tarea, macro entre las cinco tareas y ponderado por sus conteos. No reemplazar PSNR ni cambiar la selección de checkpoints. El alcance de esta actualización es únicamente SSIM; no incorporar LPIPS, DISTS, MS-SSIM ni nuevos outputs por imagen.

Los resultados por tarea y sus conteos son obligatorios. El conjunto histórico de evaluación de `enhance` tiene sólo 30 imágenes, mientras que el nuevo `val` reportado tiene 546. Declarar la incertidumbre según el conjunto efectivamente evaluado y sus escenas independientes; no trasladar el conteo histórico de 30 imágenes a la validación nueva.

## Evidencia experimental histórica — protocolo anterior

| Modelo | Deblur | Dehaze | Denoise | Derain | Enhance | Ponderado | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAXIM baseline | 24,554 | 42,824 | **38,087** | 26,428 | 22,216 | **27,455** | 30,822 |
| MoE K=5 simple oracle | **24,659** | 41,934 | 36,970 | **26,931** | 22,240 | 27,362 | 30,547 |
| MoE K=5 residual oracle | 24,611 | **45,757** | 35,568 | 26,244 | **22,720** | 27,114 | **30,980** |
| MoE K=5 compartida + residual | 24,610 | 42,660 | 37,060 | 26,780 | 22,490 | 27,350 | 30,720 |
| MoE K=8 residual aprendido top-2 | 24,230 | 45,200 | 37,190 | 25,740 | 21,640 | 27,060 | 30,800 |

### Conclusión de la corrida histórica K=8 por imagen

- El router encontró asociaciones sin usar `task_id` para seleccionar expertos.
- `deblur` y `dehaze` se concentraron en el par `E1/E4`.
- `denoise` y `derain` se concentraron en el par `E2/E6`.
- `enhance` mantuvo routing más disperso.
- `E0`, `E3`, `E5` y `E7` quedaron con uso cero: K=8 se comportó en la práctica como cuatro expertos organizados en dos pares.
- El valor `expertos efectivos=2` del log mide diversidad por imagen y es esperable con top-2; no mide cuántos expertos se usan globalmente.
- El K=8 histórico por imagen usa cabezas residuales independientes, no la cabeza compartida. Por eso no debe interpretarse como una ablación exclusiva de `K` o del router.
- La corrida queda como control histórico. No continuarla desde su checkpoint después de cambiar el router o la arquitectura.

## Resultados revisados el 1 de octubre de 2026 — validación nueva

| Modelo / checkpoint | Deblur | Dehaze | Denoise | Derain | Enhance | Ponderado | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline histórico reevaluado / inicial shared, época 0 | 23,891 | 49,110 | 34,236 | 26,886 | 22,693 | 26,878 | 31,363 |
| Shared head desde baseline histórico +10, mejor macro época extra 6 | 24,440 | 50,106 | 34,408 | 27,208 | 22,711 | 27,202 | 31,775 |
| Token-choice K=8 desde cero, mejor macro época 30 | 24,325 | 50,972 | 33,782 | 26,941 | 22,749 | 27,025 | 31,754 |

Esta tabla es **exploratoria**, no la comparación final controlada. La inicialización shared reproduce exactamente al baseline fuente, por lo que su evaluación inicial sirve como referencia baseline reevaluada. El baseline histórico fue entrenado antes de extraer el nuevo `val` del train anterior; shared hereda esa exposición. Token-choice se entrenó desde cero bajo el nuevo split. Evaluarlos sobre val nuevo no elimina la exposición previa ni iguala los presupuestos de entrenamiento. Mantener esta tabla separada de la histórica y de la futura comparación controlada.

### Baseline nuevo desde cero: finalizado el 3 de octubre de 2026

- Fuente fijada para las dos continuaciones: `maxim_no_moe/baseline_S-2_newsplit_scratch/best_checkpoint/checkpoint_30`, seleccionada exclusivamente por PSNR macro de validación.
- Presupuesto completado: 30 épocas × 2.000 pasos = 60.000 pasos, S-2, batch 2, seed 42, crop 256, LR `2e-4`, warm-up de tres épocas, weight decay `1e-4` y muestreo uniforme.
- Época 30: **31,6942 dB macro / 27,2309 dB ponderado**; SSIM **0,8541 macro / 0,8147 ponderado**. La época 29 obtuvo 31,5417/27,1799 dB, por lo que el máximo final no es una selección ambigua.
- Por tarea en época 30: deblur 24,790 dB / 0,7438 SSIM; dehaze 49,853 / 0,9762; denoise 34,121 / 0,8410; derain 27,060 / 0,8423; enhance 22,647 / 0,8673.
- El muestreo de la última época fue aproximadamente uniforme: 843/802/766/801/788 ejemplos por tarea. No hay indicios de que el resultado se explique por una tarea ausente o submuestreada de forma material.
- Frente al token-choice K=8 desde cero, el baseline queda **−0,060 dB macro** y **+0,206 dB ponderado**: empate técnico macro con una seed. Baseline mejora deblur (+0,465), denoise (+0,339) y derain (+0,119); token-choice mejora dehaze (+1,119) y enhance (+0,102). No comparar SSIM porque la corrida token-choice anterior no lo registró.
- Los notebooks de las dos continuaciones validan la misma fuente y registran su hash de parámetros. Ambos reinician optimizador y paso, evalúan y guardan el checkpoint inicial como época 0 y usan LR `1e-5`, warm-up de una época y 20.000 pasos adicionales.

### Shared head desde checkpoint histórico: finalizada

- Fuente: baseline S-2 `best_checkpoint/checkpoint_28`; hash de parámetros `b5259196ddfd3a31065417b1cf30f1d793dccdeaa494af9dabe3e45777d8b935`.
- Cobertura: 1.760/1.760 tensores cargados; diferencia máxima inicial baseline/MoE: **0,0**.
- Presupuesto: 10 épocas × 2.000 pasos = 20.000 pasos adicionales, batch 2, seed 42, LR `1e-5`, warm-up de una época, weight decay `1e-4`, backbone entrenable.
- Mejor macro: época extra 6, **31,774724 dB macro / 27,202477 dB ponderado**. Mejora frente al inicial en la misma validación: **+0,411511 / +0,324804 dB**, respectivamente.
- Mejor ponderado: época extra 5, **27,230651 dB ponderado / 31,737481 dB macro**. Conservar época 6 como checkpoint principal, según el criterio fijado.
- Todas las tareas mejoraron frente al inicial; `enhance` cambió sólo +0,018 dB. La época 10 terminó en 31,759362 dB macro, cerca del máximo de época 6.
- Routing **oracle por tarea**, distinto del routing aprendido por token. Los resultados apoyan continuar la evaluación de shared head; todavía no separan el efecto del MoE del efecto de los pasos extra.
- La repetición de shared y su control desde el nuevo baseline limpio ya finalizó; no ejecutar el control desde la fuente histórica.

### Token-choice K=8 sobre features: finalizada

- 30 épocas × 2.000 pasos = 60.000 pasos, desde cero, S-2, batch 2, seed 42, crop 256, muestreo uniforme por tarea.
- Router sobre features del último decoder, ocho expertos `C→C`, top-2 por token, ruido `0,05`, dos épocas de warm-up denso y peso de balance `0,01`.
- Mejor macro: época 30, **31,753934 dB macro / 27,025320 dB ponderado**.
- Mejor ponderado: época 29, **27,035803 dB ponderado / 31,745835 dB macro**. El checkpoint principal sigue siendo época 30.
- Frente a la referencia baseline reevaluada: **+0,390721 dB macro / +0,147647 dB ponderado**. Por tarea: deblur +0,435; dehaze +1,863; denoise −0,454; derain +0,055; enhance +0,056 dB.
- Frente a shared histórico +10: empate técnico macro (−0,021 dB) y menor ponderado (−0,177 dB). Los presupuestos y la exposición previa a val difieren; no atribuir estas diferencias exclusivamente a la arquitectura.
- Los ocho expertos recibieron gradientes no nulos en todas las épocas. La inclusión top-2 final por experto fue: E0 28,19 %; E1 34,28 %; E2 24,47 %; E3 30,15 %; E4 19,29 %; E5 14,03 %; E6 36,25 %; E7 13,35 %. Estos porcentajes suman 200 %, porque cada token selecciona dos expertos.
- Hubo uso bajo transitorio en validación: E2 permaneció debajo del 1 % durante las épocas 3–5 y luego se recuperó. No hubo colapso persistente al final; en entrenamiento todos los expertos permanecieron por encima del 1 % de inclusión.
- Preferencias por tarea en validación: dehaze favorece E0/E3; denoise E1/E6; enhance E1/E2. Son indicios de diferenciación; falta analizar asignaciones dentro de cada imagen para demostrar especialización espacial.
- `router_effective_experts_per_token ≈ 8` describe las probabilidades **densas anteriores al top-k**. No significa ocho expertos activos por token: la mezcla sparse usa dos. Las probabilidades densas son casi uniformes; el balance global no demuestra por sí solo routing útil.
- Macro época 24 → 30: 31,123678 → 31,753934 dB. Época 29 → 30: +0,008098 dB. Aún no hay evidencia suficiente de una meseta prolongada ni de que más épocas resuelvan la pérdida en denoise.
- El log registra reanudaciones desde pasos 24.000 y 48.000, y una falla de lectura de imagen desde Drive en la época 25, que posteriormente se completó al reanudar desde época 24. Los pasos descartados del intento fallido no forman parte de los 60.000 pasos conservados en el entrenamiento final.

### Fuentes de los resultados

- [Shared head: carpeta de corrida, log, metadata y diagnósticos](https://drive.google.com/drive/folders/1bVH7cC98aiX9MzbLpbd80S5s1rw3eWIf).
- [Shared head: mejor checkpoint macro](https://drive.google.com/file/d/1mMM0zE_a_CfRvYmQd_uRNa6jID_GgMaH/view).
- [Token-choice: carpeta de corrida, log y diagnósticos](https://drive.google.com/drive/folders/19lot9AHdiVxiQZG0vmr_8iaW4z_qsMVh).
- [Token-choice: mejor checkpoint macro](https://drive.google.com/file/d/1tJ5giC0pG62RfKnqMaypx4BuJb9ThbmG/view).

## Comparación controlada revisada el 4 de octubre de 2026

Todos los resultados de esta sección son de **validación nueva**, no de test. Cada continuación se seleccionó por PSNR macro de validación entre el checkpoint inicial y sus diez épocas adicionales. Se usa un crop central 256×256 por par; los conteos de imágenes no representan evaluación exhaustiva de todos sus píxeles.

| Modelo / checkpoint seleccionado | PSNR macro (dB) | PSNR ponderado (dB) | SSIM macro | SSIM ponderado |
|---|---:|---:|---:|---:|
| Fuente nueva, baseline desde cero época 30 | 31,694158 | 27,230948 | 0,854124 | 0,814671 |
| Baseline +10, época extra 10 | 31,874760 | 27,220281 | 0,854977 | 0,813282 |
| Shared head +10, época extra 8 | 32,063355 | 27,420716 | 0,857620 | 0,818572 |
| **Shared − baseline +10** | **+0,188594** | **+0,200435** | **+0,002643** | **+0,005289** |

La reevaluación inicial de las continuaciones dio baseline 31,694258 dB macro / 27,231001 dB ponderado y shared 31,694261 / 27,231001, prácticamente idénticas. La diferencia entre esa reevaluación y la metadata original de la fuente es aproximadamente 0,0001 dB macro. Frente al inicial reevaluado, baseline mejora +0,180503 dB macro y shared +0,369093 dB macro. No atribuir toda la mejora respecto de la fuente a los expertos: el control también mejora al seguir entrenando.

| Tarea | n val | PSNR baseline +10 | PSNR shared +10 | Δ dB | SSIM baseline +10 | SSIM shared +10 | Δ SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|
| deblur | 868 | 24,387094 | 24,938868 | +0,551774 | 0,733799 | 0,750056 | +0,016257 |
| dehaze | 66 | 50,578749 | 51,043390 | +0,464642 | 0,979523 | 0,980146 | +0,000623 |
| denoise | 463 | 34,196532 | 34,224843 | +0,028312 | 0,840995 | 0,838928 | −0,002067 |
| derain | 360 | 27,238290 | 27,268150 | +0,029860 | 0,846283 | 0,847040 | +0,000757 |
| enhance | 546 | 22,973137 | 22,841521 | −0,131616 | 0,874287 | 0,871931 | −0,002357 |

### Condiciones verificadas y límites

- Los 15 hashes SHA-256 de train/val/test coinciden entre ambas continuaciones y con los manifiestos de la fuente registrados por shared. Conteos: 20.523 train, 2.303 val y 4.730 test.
- Ambas declaran la misma fuente `baseline_S-2_newsplit_scratch/best_checkpoint/checkpoint_30`. Shared registra hash fuente `4f18d86352d896ea0425af3cd37c113ec0f395151c5b6750ed058e4c384aec6f`, cobertura 1.760/1.760 tensores y equivalencia inicial de salida **0,0**. El baseline declara la misma ruta; su metadata no registra por separado ese hash de parámetros.
- LR `1e-5`, warm-up 1 época, weight decay `1e-4`, seed 42, batch 2 y 2.000 pasos/época coinciden. Presupuesto nominal de ambas: 10 épocas / 20.000 pasos adicionales; backbone entrenable. El notebook final verifica los pasos del checkpoint elegido: 20.000 baseline y 16.000 shared; elegir la época 8 no altera el presupuesto ejecutado de diez épocas.
- Shared falló durante la época 4 por lectura de imagen desde Drive y reanudó desde época 3 / paso 6.000. Baseline reinició durante la época 5 y retomó desde la época 4. Los intentos incompletos no deben sumarse automáticamente al presupuesto conservado; no está probado que ambas runs mantengan idéntico orden de muestras tras reanudar.
- Shared utiliza routing **oracle por tarea**, cinco expertos residuales, dos convoluciones ocultas, escala de corrección 1. No confundirlo con token-choice K=8 aprendido.
- No se midió aquí una relación calidad/costo de inferencia ni se hizo una ablación de igual número de parámetros.

### ¿Aporta valor el MoE?

La evidencia actual permite afirmar: **“La variante shared residual con selección oracle obtuvo una mejora modesta frente al baseline continuado con el mismo presupuesto nominal: +0,189 dB macro y +0,200 dB ponderado en validación, acompañada por mejora agregada de SSIM.”**

La ganancia se concentra en deblur y dehaze; denoise/derain están cerca del empate y enhance empeora respecto del control. En deblur, shared mejora aproximadamente +0,149 dB frente a la fuente, mientras el control pierde aproximadamente −0,403 dB: parte de la ventaja relativa consiste en preservar mejor esa tarea durante la continuación.

El umbral práctico de 0,1 dB del plan es una regla de lectura, **no una prueba estadística**. La ventaja global supera ese umbral, pero una sola seed no permite afirmar robustez entre entrenamientos. Tampoco demuestra routing aprendido útil ni que el beneficio provenga exclusivamente de la mezcla, porque también aumenta la capacidad del modelo.

Si la ventaja se mantiene en test, describirla como mejora observada de esta variante en el protocolo fijado, con los cambios por tarea y sus límites. Para una conclusión más fuerte, una réplica con más seeds y una ablación de capacidad comparable serían trabajo futuro; no hace falta lanzar esas runs para ejecutar este test final. Si se usan los resultados de este test para diseñar nuevos cambios, deja de ser un test ciego para esa fase posterior.

### Fuentes

- [Baseline nuevo desde cero](https://drive.google.com/drive/folders/1AisQxdZKgT0nGxg3JtH3dr0AgA-GwBaQ).
- [Baseline +10: run completa](https://drive.google.com/drive/folders/1PwaRhH1PnTcuuYSTgdV6vXgiQ8p4nPsy).
- [Baseline +10: best checkpoint macro](https://drive.google.com/drive/folders/1g0bVMt2erbSwCBACU5yRdWr0QdaliKNX).
- [Shared +10: run completa](https://drive.google.com/drive/folders/19w59aFyr0brgfKePVJwLP_koZoBE6nsd).
- [Shared +10: best checkpoint macro](https://drive.google.com/drive/folders/19dxiPGN_CQ3jlChGGMBloaXUii1koXZG).

## Próximo paso — test final con selección cerrada

1. Abrir `test_final_baseline_vs_shared.ipynb` en Colab GPU. Ejecutar las celdas en orden. Usa el fork `Matiata/maxim` fijado al commit `98984438176afb21694dd1ec29b02fd9b40426c1`, revisado para construir el notebook; no se conoce el commit exacto ejecutado en los runtimes de las continuaciones. Las shapes/keys/dtypes se validan contra los checkpoints y no hay fallback a parámetros aleatorios.
2. Mantener las rutas ya cargadas bajo `MyDrive/Facultad/tesis`. Evaluar exactamente `best_checkpoint/checkpoint_10` baseline y `best_checkpoint/checkpoint_8` shared, con metadata de selección y pasos verificados. No usar “latest” ni el mejor ponderado como sustitución.
3. Comprobar hashes y conteos de las 15 listas contra los manifiestos fijados. Sólo `test.txt` entra al bucle de inferencia; train/val se leen para comprobar identidad, sin evaluación ni augmentations.
4. Copiar a cache local exclusivamente los pares de test; validar decodificación y fallar ante pares faltantes. No descartar imágenes ni regenerar listas.
5. Aplicar exactamente el crop central 256×256 y padding reflect inferior/derecho para imágenes pequeñas del notebook de entrenamiento. RGB float32 /255, batch 1, `train=False`, última salida de última etapa, sin clipping. No cambiar ahora a full resolution, tiles o varios crops; si se evalúa eso luego, declararlo como protocolo diferente.
6. Restaurar sólo variables para inferencia; no entrenar, crear optimizadores, guardar checkpoints ni cambiar selección. Verificar cobertura total de keys, shapes y dtypes; guardar fingerprint de las variables usadas.
7. Evaluar las 4.730 parejas: deblur 3.150; dehaze 383; denoise 767; derain 400; enhance 30. Acumular sólo PSNR/SSIM por tarea; macro da 20 % a cada tarea, ponderado usa los conteos. PSNR conserva el piso MSE `1e-6` en escala 0–255; SSIM ventana gaussiana 11, sigma 1,5, RGB, VALID, rango 1.
8. Guardar un único `test_final_results.json` en `ckpts/test_final_baseline_vs_shared_20261004`. Incluye configuración, hashes, rutas/selección y resultados agregados, sin imágenes ni métricas por imagen. El mismo archivo conserva tareas completadas si se interrumpe; sólo se repite la tarea incompleta. Si está completo, se muestran los resultados sin reevaluar.
9. Reportar tablas por tarea, macro y ponderado y diferencias shared − baseline. Leer test sin reinterpretarlo como validación ni elegir nuevas épocas en función de él. `Enhance` tiene sólo 30 pares de test; no trasladar el conteo 546 de val al análisis final.
10. Validación local del notebook: sintaxis/estructura y verificaciones sintéticas de métricas, preprocesamiento, agregación y restauración cuando las dependencias están disponibles. **La evaluación GPU de los checkpoints de Drive queda pendiente de ejecutar en Colab; no afirmar que los resultados de test ya existen.**

## Bloque 1 — Métricas y protocolo de evaluación (implementado)

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

### SSIM durante validación — contrato implementado

1. Calcular un único SSIM por imagen entre la **salida final reconstruida** y su GT, en la misma pasada de validación que ya produce PSNR. No evaluar las salidas auxiliares con SSIM, no hacer otro forward del modelo ni releer imágenes.
2. Usar la misma región, resolución, alineación y tratamiento de padding/clipping que el protocolo de evaluación de PSNR; no introducir un crop o un ajuste por imagen distinto para favorecer SSIM. Calcular sobre RGB normalizado en float32, con rango fijo `data_range=1.0`; promediar SSIM de los tres canales. Registrar esta convención una sola vez en la configuración.
3. Usar SSIM de una escala con ventana gaussiana 11×11, sigma 1,5, `K1=0,01` y `K2=0,03`, con reducción espacial y por canal a un escalar por imagen. Reutilizar una implementación vectorizada/compilada de las librerías presentes; si se usa TensorFlow, `tf.image.ssim` con `max_val=1.0` y `return_index_map=False`, o una implementación JAX equivalente verificada. Referencia: [documentación oficial de SSIM en TensorFlow](https://www.tensorflow.org/api_docs/python/tf/image/ssim).
4. Acumular únicamente sumas de SSIM y los mismos `task_count` de validación usados para PSNR. Para cada tarea `t`:

   `task_ssim[t] = sum_ssim[t] / task_count[t]`

   `ssim_weighted = sum(task_ssim[t] * task_count[t]) / sum(task_count)`

   `ssim_macro = mean(task_ssim[t] for t in tareas_presentes)`

   Con las cinco tareas previstas, cada tarea pesa 20 % en macro. No promediar batches sin ponderar su cantidad de imágenes ni calcular un único SSIM sobre imágenes promediadas.
5. Agregar sólo `task_ssim`, `ssim_macro` y `ssim_weighted` al objeto de validación del JSON de época existente. Incluir esos tres campos en la metadata existente del checkpoint seleccionado cuando haya valores disponibles; los resultados históricos sin SSIM quedan sin ese dato, nunca con un cero inventado.
6. En el resumen ya existente de cada época, agregar SSIM macro/ponderado; en las cinco líneas ya existentes por tarea, agregar su SSIM. Usar cuatro decimales y no imprimir SSIM en cada paso/batch ni crear otro archivo de logs.
7. **La selección sigue siendo `psnr_macro` de validación.** SSIM es una métrica complementaria: no agregar checkpoints `best_ssim`, una nueva loss ni términos que modifiquen el entrenamiento. El cálculo se aplicó a las tres runs nuevas y debe conservarse en futuras evaluaciones de validación de token-choice.
8. Calcular SSIM sólo durante validación, sin gradientes ni cómputo por paso de train. Mantener sólo los acumuladores pequeños por tarea; no conservar arrays de resultados por imagen. No guardar mapas SSIM, imágenes, gráficos ni nuevos diagnósticos del router. No sumar otras métricas o dependencias pesadas.

### Validación y test

El notebook histórico leía `test.txt` después de cada época y usaba ese valor para elegir el mejor checkpoint. Metodológicamente ese conjunto funcionaba como validación. El soporte explícito train/val/test ya está implementado; las reglas siguientes siguen siendo obligatorias.

Se debe:

1. agregar soporte explícito para `train.txt`, `val.txt` y `test.txt`;
2. usar `val.txt` durante entrenamiento y para seleccionar checkpoints;
3. evaluar `test.txt` sólo una vez con el checkpoint ya elegido;
4. evitar fugas por escena al construir `val.txt`;
5. registrar conteos y SHA-256 de las tres listas.

Decisión práctica: no cambiar silenciosamente los splits dentro del código. Si todavía no existe `val.txt`, el notebook debe detenerse con un mensaje claro. La creación del nuevo split debe hacerse con un script determinista, agrupado por escena y versionado. Si el nuevo `val.txt` se extrae del train actual, las comparaciones finales estrictas requerirán volver a entrenar el baseline y las variantes seleccionadas con ese mismo split.

Los resultados históricos se conservarán bajo el protocolo anterior y no se mezclarán en una misma tabla final con resultados obtenidos bajo un split nuevo.

### Pruebas de aceptación

- Test sintético con tareas de tamaños distintos que confirme los valores ponderado y macro esperados para PSNR y SSIM.
- SSIM de imágenes idénticas aproximadamente 1, con resultados finitos; comprobar equivalencia batch/por imagen y el promedio entre tareas de distintos tamaños.
- SSIM reutiliza el mismo forward y las mismas imágenes de val que PSNR. Los conteos coinciden; no se agregan archivos, mapas ni impresiones por paso. Verificar el sobrecosto usando el tiempo de validación ya registrado, sin generar nuevos reportes de rendimiento.
- Con las métricas de la época 27 K=8, el código debe reproducir aproximadamente `27,06 dB` ponderado y `30,80 dB` macro.
- El log debe imprimir claramente `validation ponderado`, `validation macro` y el criterio de guardado.
- Ninguna variable denominada `test_dataset` debe usarse dentro del bucle por época.
- La evaluación final de test no debe actualizar ni reemplazar checkpoints.

## Bloque 2 — MoE token-choice K=8 sobre features

Branch de trabajo: `exp/latent-experts-k8`.

### Cambio arquitectónico

El router histórico recibía una imagen RGB completa, aplicaba global average pooling y producía un único vector `[B, K]`. Ya fue reemplazado en esta branch por token-choice sobre features. Las especificaciones siguientes documentan el contrato implementado y deben preservarse.

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

### Estado de ejecución

El smoke training real de tres épocas y 20 pasos por época finalizó correctamente. Las dos primeras épocas usaron mezcla densa y la tercera cambió a top-2 por token. No hubo errores ni valores no finitos; el router y los ocho expertos recibieron gradientes. En la época top-2, la inclusión por experto permaneció por encima del umbral de experto muerto y el PSNR macro de validación mejoró en las tres épocas (`-22,05 → -14,48 → -10,77 dB`). Los valores absolutos bajos son esperables tras sólo 60 actualizaciones desde cero.

La corrida completa `moe_token_choice_k8_top2_S-2_scratch`, iniciada el 28/09/2026, finalizó el 01/10/2026 con 30 épocas, 2.000 pasos por época, batch 2, dos épocas de routing denso y top-2 desde la tercera. El mejor macro corresponde a época 30. Los resultados, gradientes y recuperación del uso temprano bajo de E2 están documentados en la sección de resultados. Conservar la corrida como referencia sobre el nuevo split; no modificar ni sobrescribir sus checkpoints.

## Bloque 3 — Cabeza compartida inicializada desde baseline

Este bloque se implementó en el notebook de cabeza compartida sobre `main`, separado de los cambios experimentales del router K=8. Finalizaron la corrida histórica y la repetición desde el nuevo baseline, junto con su control +10. Las especificaciones siguientes documentan el contrato ejecutado; conservar la corrida histórica como ablación exploratoria.

### Inicialización

1. Elegir un checkpoint baseline fuente y registrar ruta, época, métrica y hash/configuración.
2. Cargar todos los parámetros compatibles del backbone y de las salidas auxiliares.
3. Copiar la convolución de salida baseline a la cabeza compartida.
4. Inicializar la última convolución de cada corrección residual en cero, de modo que inicialmente:

   `salida_MoE = salida_baseline + 0`

5. Comprobar con una entrada fija y `train=False` que la diferencia máxima entre baseline y MoE inicial sea menor que `1e-6`.
6. Fallar si la cobertura de parámetros cargados no coincide con la esperada; no aceptar warm starts parciales silenciosos.

### Comparación justa por presupuesto adicional

Fijar una única constante `EXTRA_EPOCHS=10` antes de las nuevas continuaciones: **10 épocas**, equivalentes a 20.000 pasos adicionales en cada variante. Mantener ese presupuesto aunque la corrida shared se ejecute antes que el control.

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

- `baseline_S-2_newsplit_scratch`;
- `moe_shared_residual_from_newsplit_baseline_extra10`;
- `baseline_S-2_from_newsplit_ckpt_extra10`.

Los nombres deben distinguir explícitamente el nuevo split y evitar colisiones con las corridas históricas. Guardar el identificador/hash de la fuente común en ambas continuaciones.

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

## Procedimiento de las tres corridas ya completadas — conservar como referencia

### Preparación común

1. Mantener los cambios ya completados de PSNR, separación val/test, token-choice y warm start exacto. SSIM de validación por tarea/macro/ponderado del Bloque 1 ya se implementó antes de estas corridas. No rediseñar el router ni cambiar el split en este bloque.
2. Leer los manifiestos actuales; fijar conteos y SHA-256 por tarea de train/val/test. Verificar separación por escena y ausencia de pares faltantes. Usar exactamente esas listas en las tres corridas y cotejarlas con las de token-choice cuando estén registradas.
3. Validar decodificación y copiar/cachear los datos en disco local antes de entrenar para evitar fallas transitorias de Drive.
4. Fijar entorno, commit, seed 42, S-2, batch 2, crop 256, 2.000 pasos por época, muestreo uniforme, augmentations y pesos de supervisión profunda. Para el baseline desde cero, conservar el LR y schedule de entrenamiento desde cero usados por token-choice y registrarlos explícitamente.

### Corrida 1 — Baseline con el nuevo split (completada)

- Entrenar MAXIM S-2 **desde cero**, sin restaurar el baseline histórico ni ninguna variante MoE: 30 épocas / 60.000 pasos.
- Usar sólo train para gradientes y val para evaluación/selección. Seleccionar el checkpoint principal por `psnr_macro`; guardar opcionalmente el mejor ponderado y el último.
- Al terminar, **fijar un único checkpoint fuente: el mejor macro del nuevo baseline**. Registrar ruta exacta, época, paso, métricas por tarea, hash de parámetros, configuración y hashes de listas. No cambiar esta fuente después de mirar shared.
- Comparar este baseline de 30 épocas con el token-choice de 30 épocas como comparación bajo el nuevo protocolo, condicionada a confirmar iguales listas y configuración común.

### Corrida 2 — Shared head +10 desde el nuevo checkpoint (completada)

- Restaurar el backbone, salidas auxiliares y salida compartida desde la fuente nueva fijada en la corrida 1. Inicializar las últimas convoluciones residuales en cero.
- Verificar cobertura completa esperada y equivalencia de salida inicial menor que `1e-6`, con entrada fija y `train=False`.
- Crear un optimizador nuevo; entrenar backbone, cabeza compartida y expertos durante **10 épocas / 20.000 pasos adicionales**.
- Parámetros de continuación comunes: LR `1e-5`, warm-up de una época, weight decay `1e-4`, seed 42, batch 2; fijar y registrar el mismo optimizador y schedule para la corrida 3. Conservar routing oracle de esta variante.
- Evaluar y guardar el checkpoint inicial, el mejor macro, el mejor ponderado opcional y el último. Registrar mejora respecto del inicial.

### Corrida 3 — Baseline +10 desde el mismo nuevo checkpoint (completada)

- Restaurar **exactamente la fuente de la corrida 1 usada por shared**. No partir del último checkpoint por conveniencia, de shared ni de la fuente histórica.
- Crear un optimizador nuevo y entrenar **10 épocas / 20.000 pasos adicionales**, con el mismo LR, schedule, warm-up, weight decay, seed, datos, orden de muestreo, batch y supervisión que la corrida 2. Mantener el backbone entrenable.
- Guardar las mismas salidas y registrar la mejora respecto del mismo inicial. No restaurar el estado antiguo del optimizador únicamente en el control.

### Comparación y criterio de decisión

- Comparación principal de continuaciones: **shared +10 frente a baseline +10 desde la misma fuente nueva**, cada uno seleccionado por macro de val.
- Comparación desde cero: baseline nuevo de 30 épocas frente a token-choice de 30 épocas. No comparar token-choice de 30 épocas con modelos continuados como si tuvieran igual presupuesto.
- Reportar PSNR y SSIM de validación por tarea, macro y ponderado en los resúmenes existentes, junto con los conteos y la metadata ya prevista de pasos/fuente. No agregar otros archivos o métricas. Las continuaciones comparten el presupuesto heredado de la fuente y los 20.000 pasos extra.
- Tratar diferencias menores a 0,1 dB como empate técnico con una sola seed. Revisar especialmente denoise y cuánto de la mejora macro depende de dehaze.
- Decidir extensiones u otras ablaciones usando validación. No usar test para decidir qué variante continuar ni para seleccionar checkpoints. Cuando se cierre la selección experimental, evaluar los checkpoints fijados en test sin modificar esa selección.
- Las corridas compartidas históricas quedan archivadas. El control desde la fuente nueva ya finalizó; no ejecutar la continuación de la fuente histórica.

## Criterios para corridas comparables

- Mismo split, seed, crop, augmentations, batch size, arquitectura S-2 y presupuesto de pasos.
- Muestreo uniforme entre tareas durante train.
- Validación exhaustiva y determinista, separada del test final.
- Mismos pesos de supervisión profunda.
- Cero pares faltantes y conteos exactos por tarea.
- Reportar PSNR y SSIM de validación ponderados, macro y por tarea con idénticos conteos. SSIM complementa PSNR, sin cambiar el criterio de selección ni ampliar los outputs.
- Guardar entorno, commit, configuración, logs, checkpoints y diagnósticos reproducibles.
- Tratar diferencias menores a 0,1 dB como empate técnico mientras exista una sola seed.

## Riesgos y detalles a no olvidar

- La evaluación histórica usó `test.txt` para elegir checkpoints; debe declararse y no confundirse con un test final ciego.
- El nuevo `val.txt` fue extraído del train anterior. El baseline histórico y shared heredaron exposición a esas imágenes; las fuentes nuevas ya se entrenaron desde cero con el nuevo split. Continuar el checkpoint histórico no elimina esa exposición.
- El macro corrige la ponderación entre tareas, pero no la representatividad de cada conjunto. `Enhance` tenía 30 imágenes en la evaluación histórica y tiene 546 en val nuevo; reportar el conteo y escenas del conjunto usado.
- El router por token puede seguir aprendiendo atajos de dataset; analizar asignaciones dentro de cada tarea y dentro de cada imagen, no sólo tarea × experto.
- Top-2 con hard mask puede dejar expertos sin gradiente; el balance pre-top-k y la exploración son obligatorios.
- Batch size 2 limita la diversidad de imágenes por batch, aunque el balance por token use `B×N` observaciones.
- La implementación token-choice calcula los ocho expertos aunque sólo mezcle dos; no atribuirle ahorro de cómputo sparse sin implementar y medir dispatch/gather/scatter.
- Hubo fallas transitorias leyendo imágenes desde Drive. Antes de corridas largas, copiar o cachear el dataset en almacenamiento local y validar decodificación.
- Reanudar una época desde checkpoint no reprodujo exactamente el mismo resultado; registrar estado del pipeline y seeds de TensorFlow/NumPy/JAX.
- Mantener versiones fijadas de JAX, Flax, Optax y TensorFlow en Colab.
- No mezclar cambios de métricas, router y cabeza compartida en una sola corrida sin controles intermedios.
