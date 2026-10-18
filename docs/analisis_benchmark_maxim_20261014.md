# Análisis del benchmark MAXIM baseline/shared

Fecha: 14 de octubre de 2026. Segunda revisión del mismo día, que reemplaza la interpretación y la propuesta de la primera.

Fuente: `benchmark_maxim_baseline_shared_20261013-20261014T175836Z-1-001.zip`.
SHA-256 del ZIP: `f8371d7bcecda70826f9dc472ebde91bb93ccd40466821a59a8ab4eef0990e17`.
Protocolo: `ca16d6043448dfafbf48fe3b730eae69692fe4ec1ee089d61b9ebbdaf88662d8`.

Objetivo de la fase: reducir tamaño y complejidad de MAXIM aprovechando expertos por tarea, o al menos medir el impacto de los expertos sobre el costo. El test final ya ejecutado es un diagnóstico para orientar esa optimización.

## 1. Resumen

1. Los datos son íntegros y las mediciones son estables dentro de cada configuración.
2. Shared ejecuta hoy los cinco expertos. Seleccionar la tarea no reduce trabajo.
3. **El modelo está limitado por ancho de banda de memoria y por lanzamiento de kernels, no por cómputo.** Reducir FLOPs no garantiza reducir latencia.
4. **Baseline y shared se midieron con ejecutables distintos para el mismo backbone**, porque cada sesión de Colab hizo su propio autotuning de cuDNN. Las diferencias de latencia de ±5 % y de FLOPs del compilador entre modelos no son atribuibles a los expertos.
5. La traza baseline sí permite atribuir el tiempo por familia de bloques. El reparto de tiempo y el de parámetros son distintos.
6. El candidato de la primera revisión, dos parejas de convoluciones RCAB a resolución completa, representa ~0,26 % de los parámetros y ~0,5 % del tiempo de kernels. Es demasiado chico para cualquier objetivo.
7. **Para reducir tamaño con expertos, el lugar indicado son las convoluciones RCAB del nivel más profundo, con 128 canales.** Concentran ~16,6 % de los parámetros con ~1,3 % del tiempo de kernels.

## 2. Integridad y alcance

El JSON tiene estado `complete`, ambos modelos completos y 36/36 mediciones completas: cinco baseline y 31 shared. Dos ejecuciones shared interrumpidas fueron recuperadas. El profiling shared quedó desactivado por la incompatibilidad nativa documentada. Hay trazas baseline de inferencia y entrenamiento, y HLO de todos los ejecutables.

Las cuatro ejecuciones registran NVIDIA L4, driver 580.82.07, JAX/jaxlib 0.10.2, Flax 0.11.2 y Optax 0.2.8. Son sesiones separadas: baseline en una y shared en otras tres. Cada configuración tuvo diez calentamientos y tres bloques de cien ejecuciones. Las entradas son sintéticas en float32 y residen en GPU.

Las comparaciones full/final-only y dinámica/estática dieron diferencia máxima absoluta 0. No se recalcularon PSNR/SSIM. La calidad de la metadata es la de la selección histórica en validación.

## 3. Tiempo y parámetros medidos

Shared es la media de las cinco medianas por tarea.

| Medición | Baseline | Shared dinámico | Cambio shared/baseline |
|---|---:|---:|---:|
| Inferencia 256×256, ms/imagen | 41,618 | 44,326 | +6,51 % |
| Inferencia 512×512, ms/imagen | 253,763 | 252,516 | −0,49 % |
| Entrenamiento 256×256 batch 2, ms/paso | 422,429 | 443,726 | +5,04 % |
| Parámetros totales | 14.176.178 | 14.272.993 | +0,683 % |

Cada experto tiene 19.363 parámetros: dos convoluciones 3×3 de 32→32 y una 3×3 de 32→3. Los HLO conservan las tres convoluciones de cada uno de los cinco expertos, también en los ejecutables con tarea fijada. La ruta actual apila las cinco correcciones y luego multiplica por las compuertas.

### Variabilidad que limita la lectura

- Dentro de una sesión, los bloques de una misma configuración varían menos de 1 %.
- Los cinco ejecutables shared con tarea fijada a 256 tienen el mismo costo compilado, pero sus medianas van de 44,36 a 46,30 ms. Esa dispersión de ~4 % aparece entre configuraciones equivalentes de la misma sesión.
- Baseline full y final-only invierten su orden entre 256 y 512.

Con esta variabilidad, ni el +6,5 % a 256 ni el −0,5 % a 512 permiten afirmar un sobrecosto o un ahorro de shared.

## 4. Ejecutables distintos para el mismo backbone

Los HLO compilados registran el algoritmo cuDNN elegido para cada convolución. Baseline y shared usan algoritmos distintos para convoluciones idénticas del backbone.

| Inferencia final-only | Custom calls de convolución | Custom calls cuBLAS | Algoritmo cuDNN más frecuente |
|---|---:|---:|---|
| Baseline 256 | 222 | 10 | algo 20 |
| Shared 256 | 237 | 4 | algo 11 |
| Baseline 512 | 222 | 12 | algo 15 |
| Shared 512 | 237 | 30 | algo 15 |

XLA elige esos algoritmos midiendo tiempos al compilar, y cada sesión hizo su propia elección. Esto explica que shared, que hace estrictamente más trabajo, aparezca con menos FLOPs que baseline a 512 y en entrenamiento. `cost_analysis()` cuenta el programa ya optimizado, que difiere entre sesiones.

**Consecuencia para el protocolo:** las variantes deben compilarse y medirse en el mismo proceso, alternando el orden. Los FLOPs deben tomarse además del programa previo a la optimización, que es homogéneo entre variantes.

## 5. Límite por memoria y por lanzamiento de kernels

El JSON incluye los bytes accedidos estimados por el compilador. La primera revisión no los usó.

| Inferencia baseline | 256×256 | 512×512 |
|---|---:|---:|
| GFLOP del compilador | 121,3 | 551,0 |
| GB accedidos según el compilador | 15,8 | 64,1 |
| FLOP por byte | 7,7 | 8,6 |
| Cómputo logrado | 2,9 TFLOP/s | 2,2 TFLOP/s |
| Ancho de banda implícito | 380 GB/s | 253 GB/s |

Como referencia, una L4 ofrece unos 30 TFLOP/s en FP32 y unos 300 GB/s de memoria. Con 8 FLOP por byte, el modelo queda muy por debajo del punto en que el cómputo limita. A 512 opera cerca del ancho de banda máximo. A 256 una parte de los datos queda en la caché L2 de 48 MB, y por eso 512 tarda 6,1 veces más con 4 veces más píxeles.

La traza baseline muestra 2.314 kernels por inferencia a 256. Los kernels suman ~32,6 ms por llamada frente a 41,6 ms de latencia sincronizada. El resto es overhead de lanzamiento y huecos entre kernels.

Por tipo de kernel, sobre el tiempo de GPU:

| Tipo | Inferencia 256 | Entrenamiento batch 2 |
|---|---:|---:|
| GEMM | 30 % | 25 % |
| Convolución cuDNN | 25 % | 21 % |
| Elementales | 19 % | 12 % |
| Transposiciones | 13 % | 26 % |
| Reducciones | 12 % | 15 % |

Transposiciones, reducciones y operaciones elementales son mayoritariamente movimiento de memoria. Algunas convoluciones de entrenamiento usan kernels FP32 sin tensor cores.

**Consecuencias:**
- Bajar FLOPs no se traduce automáticamente en milisegundos.
- Un módulo con más kernels pequeños puede ser más lento aunque tenga menos FLOPs. Esto aplica a las convoluciones depthwise.
- La precisión bf16 y la rematerialización atacan directamente el cuello de memoria. Son referencias útiles para dimensionar lo que aportan los expertos.

## 6. Atribución por familia de bloques

### Método

Cada evento de la traza trae el nombre de la operación HLO o del command buffer. El HLO compilado trae, en su metadata, el scope de Flax de cada operación. El nombre del kernel no sirve para atribuir, porque XLA reutiliza un kernel para fusiones idénticas y lo nombra según la primera. Atribuir por nombre carga casi todo a stage_0.

La atribución usada alinea por posición los kernels de cada llamada con las fusiones y custom calls del programa compilado, en orden de ejecución. Las tres llamadas perfiladas dan 2.314 kernels cada una, sin desajustes en la alineación. Las dos etapas quedan balanceadas, como corresponde a S-2. Es una aproximación: los valores finos por bloque individual tienen ruido.

### Tiempo y parámetros por familia

Tiempo sobre los kernels de inferencia baseline a 256. Parámetros tomados de los nombres de pesos del HLO.

| Familia | % parámetros | % tiempo de kernels |
|---|---:|---:|
| Cross-gating | 25,1 | 25 |
| Convoluciones RCAB conv1/conv2, todos los niveles | 21,9 | ~5 |
| gMLP local por bloques | 14,5 | 14 |
| gMLP global por grilla | 14,5 | 12 |
| Proyecciones y LayerNorm de los bloques multi-eje | 8,7 | 6 |
| RCAB, atención de canales y LayerNorm | 3,0 | ~11 |
| Redimensionamiento | 1,4 | 10 |
| SAM, fusión entre etapas y resto sin familia asignada | 10,9 | ~16 |

En entrenamiento, los bloques gMLP multi-eje dominan el tiempo, seguidos por cross-gating.

### Convoluciones RCAB por nivel

| Nivel de la U-Net | Canales | Resolución a 256 | Parámetros conv1/conv2 | % del modelo | % tiempo de kernels |
|---|---:|---:|---:|---:|---:|
| 0 | 32 | 256×256 | 147.456 | 1,0 | ~1,4 |
| 1 | 64 | 128×128 | 589.824 | 4,2 | ~2,2 |
| 2 | 128 | 64×64 | 2.359.296 | 16,6 | ~1,3 |

Una convolución 3×3 cuesta los mismos FLOPs en los tres niveles, porque al bajar la resolución se duplican los canales. En el nivel profundo tiene 16 veces más pesos. Cada nivel tiene ocho parejas: dos RCAB en el encoder y dos en el decoder, en cada una de las dos etapas.

## 7. Implicaciones para la sustitución con expertos

### Candidato descartado

La primera revisión proponía sustituir dos parejas de C=32 en `stage_1_decoder_block_0`. Representan 36.992 parámetros, ~0,26 % del modelo, y ~0,5 % del tiempo de kernels. Su ahorro quedaría por debajo del ruido de medición en cualquier dimensión.

### Candidato propuesto: RCAB del nivel profundo

Sustituir las ocho parejas conv1/conv2 de C=128. Se conservan LayerNorm, atención de canales y residual del RCAB, y todo el resto de MAXIM.

Estimación analítica con el módulo `1×1 C→d → GELU → depthwise 3×3 → 1×1 d→C`, contando bias:

| Por pareja sustituida | Parámetros | GFLOP a 256 |
|---|---:|---:|
| Original, dos 3×3 128→128 | 295.168 | ~2,42 |
| Módulo d=32 | 8.672 | ~0,07 |
| Módulo d=64 | 17.216 | ~0,14 |
| Banco S2 de cinco módulos d=32, residentes | 43.360 | ~0,07 activos |
| Banco S2 de cinco módulos d=64, residentes | 86.080 | ~0,14 activos |

Sobre las ocho parejas, S2 con d=32 reduce el modelo en ~2,01 M de parámetros, ~14 % del total, aun con los cinco expertos guardados. Los FLOPs activos bajan ~19 GFLOP a 256. S1 usa un único módulo del mismo tamaño activo y sirve de control.

Esta sustitución reduce tamaño y operaciones, pero **no se espera ahorro de latencia relevante**, porque esas convoluciones ocupan poco tiempo de GPU. El módulo depthwise además agrega kernels. La tesis debe declarar qué dimensión se reduce.

Cross-gating y los gMLP pesan en tamaño y en tiempo. Son la opción si más adelante se busca latencia, pero hacen la mezcla espacial global de MAXIM y su sustitución es más riesgosa. El notebook siguiente mide su peso total con variantes de MAXIM sin esos bloques, sólo como techo de costo.

## 8. Memoria

NVML registra ~4,75 GiB en inferencia a 256, ~16,8 GiB a 512 y ~16,9 GiB en entrenamiento para ambos modelos. Es memoria reservada del proceso, no pico de buffers vivos, y se arrastra entre configuraciones del mismo proceso.

`memory_analysis()` da temporales de ~11,8 GiB baseline y ~12,3 GiB shared en el paso de entrenamiento con batch 2. Los parámetros ocupan 57 MB y el estado de AdamW unas tres veces eso. **La memoria de entrenamiento es casi toda activaciones.** La rematerialización por bloque es la palanca directa sobre esa dimensión.

## 9. Calidad de referencia

En test, shared queda prácticamente igual al baseline: +0,009 dB macro y +0,061 dB ponderado de PSNR, con SSIM macro −0,001. Se usa sólo como diagnóstico. Las decisiones de esta fase se toman con validación.

## 10. Siguiente encargo

El detalle está en `handoff_optimization_maxim_moe_2026-10-08.md` y en el notebook `benchmark_maxim_v2_compresion.ipynb`:

1. Selección efectiva de la cabeza final de shared, con verificación de equivalencia.
2. Protocolo corregido: un solo proceso, variantes alternadas, caché persistente de compilación, FLOPs previos a la optimización, bytes y cantidad de kernels.
3. S1/S2 en las RCAB del nivel profundo, con pesos nuevos, sólo para costo.
4. Techos de costo de cross-gating y gMLP con variantes de MAXIM sin esos bloques.
5. Referencias bf16 en inferencia y rematerialización en entrenamiento.

La recuperación de calidad de S1/S2 queda para después de revisar esas mediciones.
