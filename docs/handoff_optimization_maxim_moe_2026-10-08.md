# Handoff — Diagnóstico y reducción del costo de MAXIM con expertos

Fecha: 8 de octubre de 2026. Actualizado el 14 de octubre de 2026. Destinatario: Codex que trabaja en el repositorio de la tesis.

## 0. Estado al 14 de octubre de 2026

Esta sección resume lo ocurrido desde el 8 de octubre y **prevalece sobre las secciones siguientes donde se contradigan**. El análisis completo está en `docs/analisis_benchmark_maxim_20261014.md`.

**Objetivo aclarado por el usuario.** La fase busca reducir tamaño y complejidad de MAXIM aprovechando los expertos por tarea, o al menos medir el impacto de los expertos sobre el costo. El test final ya ejecutado es un diagnóstico para orientar esa optimización. No reorientar la tesis hacia la falta de mejora de calidad de shared.

**Fase A completada.** El usuario ejecutó `benchmark_maxim_baseline_shared.ipynb` y entregó sus artefactos. Hallazgos que cambian el plan:

1. Shared ejecuta las cinco cabezas. Fijar la tarea no reduce trabajo.
2. El modelo está limitado por ancho de banda de memoria y por lanzamiento de kernels: ~8 FLOP por byte y ~2.300 kernels por inferencia. Reducir FLOPs no garantiza reducir latencia.
3. Baseline y shared se midieron en sesiones distintas y cada una eligió algoritmos cuDNN distintos para el mismo backbone. Las diferencias de ±5 % en latencia y las de FLOPs del compilador no son atribuibles a los expertos.
4. La traza baseline se puede atribuir por familia alineando kernels y programa compilado por posición. Tiempo y parámetros se reparten distinto.
5. Las convoluciones RCAB conv1/conv2 del nivel más profundo, C=128, concentran ~16,6 % de los parámetros con ~1,3 % del tiempo de kernels. Es el lugar indicado para reducir tamaño con expertos. El candidato anterior, dos parejas de C=32 a resolución completa, pesa ~0,26 % de los parámetros y queda descartado.
6. La memoria de entrenamiento es casi toda activaciones: ~12 GiB de temporales con batch 2 frente a 57 MB de parámetros.

**Decisiones.**

- S1/S2 sustituyen las ocho parejas conv1/conv2 de los RCAB del nivel 2: dos RCAB por bloque, en el encoder y el decoder más profundos de cada una de las dos etapas. Se conservan LayerNorm, atención de canales y residual.
- Módulo: `1×1 C→d → GELU → depthwise 3×3 → 1×1 d→C`, con d=32 y d=64. S1 usa un módulo común; S2 un banco de cinco con selección oracle.
- La selección efectiva se implementa reuniendo los pesos del experto de cada ejemplo y aplicándolos por ejemplo. Funciona con batch 1 y con batches de tareas mezcladas, sin cambiar el sampler.
- El ahorro esperado es de parámetros y operaciones, no de latencia. Se declara así en la tesis.
- Todas las variantes se miden en un mismo proceso, alternadas, con caché persistente de compilación.

**Código agregado al repositorio, pendiente de commit y push por el usuario.**

- `maxim/models/maxim.py`: opciones nuevas que por defecto reproducen exactamente el MAXIM original, con el mismo árbol de parámetros. `rcab_mixer`, `rcab_mixer_levels`, `rcab_mixer_width`, `rcab_mixer_num_experts` y `rcab_mixer_selection` controlan la sustitución. `remat_blocks` activa rematerialización por bloque. `task_id` se propaga hasta los RCAB.
- `maxim/models/compression.py`: `TaskBottleneckBank`, `SelectiveSharedResidualMaximMoE` con el mismo árbol de parámetros que shared, `transplant_params` con auditoría estricta y el catálogo `VARIANTS`.
- `maxim/models/compat.py`: parche acotado a `flax.core.lift` porque Flax 0.11.2 pasa a `jax.remat` una opción que JAX 0.10 eliminó.
- `tests/test_compression_models.py`: pruebas en CPU. Comparan el modelo por defecto contra `maxim.py` del commit `98984438`, la cabeza selectiva contra shared, el banco contra su referencia que calcula todos los expertos, conteos de parámetros, ausencia de convoluciones de expertos no seleccionados, gradientes sólo en los expertos seleccionados y equivalencia de la rematerialización.
- `notebooks/benchmark/benchmark_maxim_v2_compresion.ipynb`: benchmark de costo de B0, S0, S0 con selección efectiva, S1/S2 con d=32 y d=64, techos sin RCAB profundos, sin cross-gating y sin gMLP, bf16 en inferencia y rematerialización en entrenamiento. Se validó de punta a punta en CPU con un modelo reducido y checkpoints sintéticos.

**Próximo paso del usuario:** subir el código y ejecutar el benchmark v2 en Colab con L4.

**Reorganización del 15/10.** Los notebooks vigentes están en `notebooks/`, la documentación en `docs/`, los logs y resultados en `runs_logs/` y lo obsoleto en `archivo/`. El mapa completo está en `docs/ESTRUCTURA.md`. El resultado del benchmark v1 está en `runs_logs/benchmark_v1_20261013/`.

## 1. Decisión y objetivo

En la reunión con los profesores se decidió orientar la siguiente fase a reducir el costo computacional conservando calidad. La evidencia actual no permite asegurar una mejora robusta de calidad frente al baseline. Esto no equivale a haber realizado una prueba formal de significancia.

Usar **shared head residual** como punto de partida. Conservar el paradigma de especialistas por tarea y estudiar si permite comprimir partes internas de MAXIM. Comparar siempre contra un módulo pequeño común para separar el efecto de la compresión del efecto de la especialización.

Objetivo general: **reducir el costo computacional de MAXIM conservando la calidad de restauración**, manteniendo el paradigma de expertos y comparándolo con módulos pequeños comunes. El costo se caracteriza mediante FLOPs estimados, latencia, memoria máxima y parámetros totales, activos y entrenables; entrenamiento e inferencia se analizan por separado. **Todavía no se prioriza una sola medida ni se fija un porcentaje de reducción.**

Objetivo inmediato: preparar un diagnóstico reproducible en un notebook Colab. El usuario ejecutará el notebook y devolverá sus artefactos; con esos resultados se determinará qué dimensiones podemos mejorar, cuáles son más fáciles de optimizar con los recursos disponibles y qué compromisos de calidad/costo aparecen. No interpretar menor latencia, menos parámetros o menor memoria como equivalentes automáticos.

Hipótesis: un banco de expertos pequeños, del que se ejecuta sólo el experto correspondiente a la tarea, puede reemplazar un bloque compartido costoso y conservar mejor calidad que un único bloque pequeño común.

Para ahorrar operaciones o tiempo de inferencia, el costo del experto ejecutado + selección/dispatch + componentes comunes nuevos debe ser menor que el costo del bloque eliminado, bajo la misma medida. Tener expertos pequeños no basta si se ejecutan todos o si el bloque original sigue activo. El ahorro de memoria o parámetros debe comprobarse por separado: un banco de especialistas puede reducir operaciones activas y aumentar los pesos totales.

## 2. Contexto y referencias actuales

- Framework: JAX/Flax. Configuración principal: MAXIM **S-2**, con supervisión multietapa/multiescala. No reconstruir un modelo single-stage por error.
- Shared actual: cabeza común más cinco expertos residuales finales, con selección **oracle por tarea conocida**. No es token-choice ni routing aprendido.
- Token-choice K=8 queda como antecedente. Su implementación descrita en el plan calcula todos los expertos antes de aplicar top-2; no tomarla como evidencia de ahorro sparse.
- Baseline y shared actuales parten del baseline limpio entrenado con el nuevo split. Ambas continuaciones ejecutaron diez épocas nominales adicionales.
- Checkpoint baseline: `baseline_S-2_from_newsplit_ckpt_extra10/best_checkpoint/checkpoint_10`.
- Checkpoint shared: `moe_shared_residual_from_newsplit_baseline_extra10/best_checkpoint/checkpoint_8`.
- Carpeta baseline: https://drive.google.com/drive/folders/1PwaRhH1PnTcuuYSTgdV6vXgiQ8p4nPsy
- Carpeta shared: https://drive.google.com/drive/folders/19w59aFyr0brgfKePVJwLP_koZoBE6nsd
- Test final completado: https://drive.google.com/drive/folders/1d0yGUvOiZWafk3YeR5PdoDfDbGrfGy0T
- En test, según los resultados ya revisados en la conversación, shared obtuvo aproximadamente +0,0092 dB macro y +0,0611 dB ponderado frente al baseline. Son referencias de contexto, no nuevas mediciones realizadas para este handoff. Verificar el JSON fuente antes de incorporarlas a un reporte automatizado.
- El archivo `plan_correccion_moe.md` todavía describe el test como pendiente: esa sección quedó desactualizada. Esta nueva decisión reemplaza la prioridad anterior de ejecutar ese test. Conservar sus condiciones y antecedentes; no sobrescribir resultados históricos.
- Repositorio: https://github.com/Matiata/maxim
- El notebook de test se construyó revisando `Matiata/maxim` en el commit `98984438176afb21694dd1ec29b02fd9b40426c1`. No se conoce el commit exacto de todos los runtimes de entrenamiento. Inspeccionar la rama vigente y registrar el commit realmente utilizado.

## 3. Orden del trabajo

Estado al 14/10: los pasos 1 a 4 están completos. El paso 5 se reemplazó por el benchmark v2 de costo, ya generado. La adaptación de S1/S2 queda pendiente de sus resultados.

1. **Codex:** inspeccionar el repositorio, sus instrucciones y los constructores reales de baseline/shared sólo hasta el nivel necesario para generar el notebook de diagnóstico.
2. **Codex:** generar `benchmark_maxim_baseline_shared.ipynb`, listo para ejecutarse en Colab, y validarlo estáticamente o con pruebas sintéticas livianas que no requieran los checkpoints ni una GPU.
3. **Usuario:** ejecutar el notebook en Colab con los checkpoints fijados y conservar `benchmark_results.json`, `diagnostico_costo_maxim.md`, la salida de consola y las trazas que correspondan.
4. **Codex, después de recibir esos artefactos:** revisar los perfiles, seleccionar una sustitución interna acotada y estimar su efecto potencial sobre operaciones, tiempo, memoria y parámetros.
5. **Codex:** generar el notebook o los notebooks de las pruebas S1/S2, incluyendo verificación de ejecución selectiva y adaptación acotada, pero sin ejecutarlos.
6. **Usuario:** ejecutar en Colab las pruebas de costo y luego las corridas de adaptación autorizadas; devolver sus JSONs y logs.
7. **Codex y usuario:** analizar los resultados y decidir qué dimensión de costo conviene priorizar y cuál mejora es más viable. Sólo entonces definir extensiones, metas cuantitativas y evaluación final.

### Modalidad de trabajo y responsabilidades

Codex **genera y revisa el código de los notebooks; no ejecuta los benchmarks, el profiling, el entrenamiento ni la evaluación sobre checkpoints reales**. La ejecución con GPU y Drive corresponde al usuario en Colab. Las comprobaciones locales de Codex se limitan a estructura del notebook, sintaxis, imports que estén disponibles, funciones puras y casos sintéticos pequeños. Toda medición dependiente de GPU, checkpoints o datasets debe quedar marcada como pendiente hasta que el usuario ejecute el notebook.

El encargo inmediato para quien recibe este handoff se limita a generar la fase A. La fase de compresión está definida como dirección posterior, pero sus ubicaciones y tamaños dependen de evidencia aún no disponible. **No implementar ni generar todavía S1/S2, no entrenar modelos y no redactar conclusiones de costo antes de recibir `benchmark_results.json` y las trazas producidas por el usuario.**

## 4. Fase A — Diagnóstico reproducible

### A1. Auditoría del forward

- Identificar etapas, niveles del encoder/decoder, bloques MLP, proyecciones de canales, cross-gating y cabezas finales.
- Verificar cómo se calculan y seleccionan hoy los cinco expertos finales de shared. No asumir que seleccionar uno implica ejecutar uno.
- Verificar el uso de salidas auxiliares durante inferencia. Distinguir salidas descartables de operaciones necesarias para etapas posteriores.
- Cargar checkpoints con cobertura completa de keys/shapes/dtypes y sin fallback aleatorio. Registrar hash y configuración.
- Mantener dos referencias claras si se optimiza el forward sin alterar la arquitectura: implementación actual y forward de inferencia equivalente. Aplicar optimizaciones equivalentes a los modelos comparados y verificar la equivalencia numérica de la salida final.

### A2. Notebook y protocolo

Codex debe crear `benchmark_maxim_baseline_shared.ipynb`, reutilizando código de restauración/evaluación existente. El usuario lo ejecutará en Colab. La instrumentación vive fuera del bucle ordinario de entrenamiento; no agregar nuevos logs por paso a los notebooks actuales.

Configuraciones iniciales:

| Uso | Entrada | Batch | Observación |
|---|---|---:|---|
| Inferencia de referencia | RGB 256×256 | 1 | Coincide con el tamaño de la evaluación existente. |
| Escalabilidad | RGB 512×512 | 1 | Sólo costo; no mezclar sus métricas con la evaluación a 256. |
| Paso de entrenamiento | RGB 256×256 | 2 | Forward, pérdida, backward y actualización real del optimizador. |

- Preferir L4 para continuidad, pero usar la misma GPU concreta para todas las variantes de una comparación. Registrar GPU, JAX/jaxlib/Flax, CUDA disponible, dtype y precisión de matmul, configuración de compilación, modelo y shapes.
- Mantener inicialmente la precisión del código/checkpoints actuales. Cualquier cambio posterior de precisión es una ablación separada aplicada también a los controles.
- Compilar cada configuración antes del cronometraje, realizar al menos diez calentamientos y medir inicialmente cien ejecuciones por configuración. Repetir en tres bloques de medición para observar variabilidad de tiempo, alternando el orden de modelos cuando sea posible.
- Mantener entradas en GPU y sincronizar resultados con `jax.block_until_ready` o equivalente que cubra el árbol de salidas. No cronometrar solamente el dispatch asíncrono.
- Excluir carga de imágenes, acceso a Drive, transferencias CPU/GPU, PSNR/SSIM y compilación del tiempo principal. Informar compilación y tiempo de pipeline por separado si se recogen.
- Medir cada tarea para shared y cualquier variante condicional. Reportar por tarea y agregado equiponderado, junto al costo fijo del baseline. No esconder overhead de selección o agrupamiento necesario para inferencia.
- Medir el paso de entrenamiento en estado descartable; no escribir sobre checkpoints originales. Excluir validación y lectura de datos de esa medición y describir qué módulos son entrenables.

### A3. Medidas y salidas

| Campo | Definición |
|---|---|
| Latencia | Mediana y p95 de ms/imagen, por shape y tarea cuando corresponda. |
| FLOPs | Estimación del ejecutable compilado, con método y versión. No confundir MACs y FLOPs ni asumir cómo cuenta las ramas condicionales. |
| Memoria | Pico de memoria durante inferencia y entrenamiento, medidos por separado. Distinguir buffers vivos de reserva del allocator. |
| Parámetros | Totales, entrenables y usados en la ruta por tarea. Los pesos no ejecutados pueden seguir residentes en GPU. |
| Entrenamiento | ms/paso y presupuesto de pasos; incluir costo del profesor si después hay destilación. |
| Calidad | PSNR/SSIM por tarea, macro y ponderados, bajo el protocolo existente. |

Usar `jax.jit(...).lower(...).compile().cost_analysis()` cuando esté disponible. Su formato y disponibilidad dependen del backend/version; reportar ausencia, no estimar silenciosamente. Para condiciones dinámicas, el análisis del compilador puede no representar el costo de una sola rama: corroborar con ejecutables especializados por tarea y perfiles reales.

Para memoria, usar XProf/Memory Viewer o herramientas disponibles en el runtime. Un snapshot posterior no equivale al pico, y `nvidia-smi` puede mostrar principalmente memoria reservada por JAX. Si sólo se puede obtener reserva o una estimación estática, identificarla como tal. Ejecutar comparaciones de memoria en procesos limpios para evitar que modelos anteriores contaminen el resultado.

El notebook debe guardar un único `benchmark_results.json` con configuración, resúmenes y estado de cada medición, y generar un reporte breve `diagnostico_costo_maxim.md`. Debe conservar trazas de profiling separadas como respaldo; no generar imágenes restauradas masivas ni métricas nuevas. Codex sólo debe comprobar que estas salidas estén implementadas y que el notebook falle de forma explícita ante datos o checkpoints faltantes; los archivos con mediciones reales aparecerán cuando el usuario lo ejecute en Colab.

### A4. Localización del costo

- Capturar trazas de unas pocas inferencias y pasos de entrenamiento ya compilados. El profiling va separado de los tiempos sin instrumentación.
- Añadir nombres de scopes a módulos relevantes para atribuir operaciones a etapas/niveles/bloques.
- Examinar proyecciones de canales, MLP local/global, cross-gating, convoluciones a alta resolución y expertos finales como candidatos, sin declarar de antemano el bloque dominante.
- Distinguir tiempo de kernels, movimiento de memoria y overhead de ejecución. Los kernels pueden estar fusionados: no sumar microbenchmarks independientes como si reprodujeran el tiempo del modelo completo.
- Seleccionar uno o pocos bloques repetidos de una misma familia con costo agregado relevante. Empezar por una intervención acotada.

Criterio de salida de la **ejecución del usuario**: tabla baseline/shared con todas las dimensiones de costo disponibles, mapa de operaciones/tiempos y memoria, y datos suficientes para proponer un bloque concreto a sustituir, con shapes, función local/global y efecto estimado. Distinguir medidas obtenidas, estimaciones y pendientes. La propuesta concreta se redactará después de revisar esos resultados, no durante la generación inicial del notebook. El diagnóstico debe permitir estudiar varias formas de ahorro, sin elegir todavía la latencia como objetivo único. Ejemplo puramente ilustrativo sobre tiempo: reducir a la mitad una parte que consume 40 % del tiempo permitiría aproximadamente 20 % de ahorro total si todo lo demás permaneciera igual y no hubiera overhead nuevo.

## 5. Fase B — Compresión manteniendo expertos

### B1. Experimento controlado

| Variante | Arquitectura | Papel |
|---|---|---|
| B0 | Baseline actual | Referencia de calidad/costo original. |
| S0 | Shared actual | Referencia y profesor principal. |
| S1 | Shared con sustitución por un módulo pequeño común | Control del efecto de simplificar el bloque. |
| S2 | Shared con sustitución por cinco módulos pequeños, seleccionados por tarea | Medir si la especialización conserva más calidad a costo activo comparable. |

Conservar iguales en S1/S2 los módulos no intervenidos, incluyendo la cabeza compartida y expertos residuales finales. La diferencia principal es el módulo común versus el banco interno por tarea. No eliminar las cabezas finales al mismo tiempo: sería otro cambio.

S1 y S2 deben tener el mismo diseño/tamaño del módulo activo, ubicación y entrenamiento comparable. S2 tendrá más parámetros totales; registrar esa diferencia en vez de afirmar igualdad de tamaño o memoria. Comparar también el costo efectivo completo, no sólo el del experto.

### B2. Sustitución interna

- El bloque pesado original deja de ejecutarse en el estudiante; no calcular su salida y añadir después un experto pequeño.
- Mantener contrato de shapes, residual y conexiones necesarias para S-2 y supervisión profunda.
- Primer candidato de experto, sujeto al diagnóstico: `1×1 C→d → GELU → depthwise 3×3 → 1×1 d→C`, con `d<C`. Probar inicialmente dos tamaños acotados, por ejemplo `d≈C/4` y `d≈C/2`, adaptados a restricciones reales de shapes.
- **Ubicación decidida el 14/10:** las ocho parejas conv1/conv2 de RCAB del nivel 2, con C=128, d=32 y d=64. Cada pareja original tiene 295.168 parámetros. Un módulo d=32 tiene 8.672 y un banco de cinco, 43.360. Con S2 d=32, el modelo baja ~2,01 M de parámetros, ~14 %, y ~19 GFLOP activos a 256.
- Este módulo mezcla canales y aporta contexto local; no reproduce automáticamente la mezcla global de MAXIM. Si el bloque elegido depende fuertemente del contexto global, conservar una parte global común ligera o elegir otra sustitución. Contar todo su costo.
- Inicializar parámetros compatibles desde shared y documentar los nuevos. Un bloque sustituido no mantiene equivalencia exacta sólo por inicializar una corrección en cero: en ese caso se pierde la función del bloque original.
- Mantener inicialmente un experto por tarea y cinco tareas, sin router aprendido ni top-2. Describirlo como selección oracle de especialistas; no atribuirle descubrimiento automático de degradaciones.

### B3. Ejecución selectiva real

- En inferencia batch 1, usar una ruta compilada por tarea o un mecanismo condicional que ejecute únicamente el experto elegido. Todos los caminos deben cumplir el mismo contrato de salida.
- **Implementado el 14/10:** los pesos de los expertos se guardan apilados y se reúnen por `task_id` antes de aplicarlos, con `vmap` sobre el batch. No hay ramas condicionales. En batch 1 se ejecuta un único experto. En batches mezclados cada ejemplo usa sus pesos. Los expertos no seleccionados reciben gradiente exactamente cero. El modo `all_experts_reference` existe sólo para verificar la semántica y medir el costo de no seleccionar.
- Inicializar/restaurar los cinco expertos sin obligar al forward a calcularlos todos.
- No implementar selección mediante `stack` de cinco salidas calculadas, máscara, `where` o mezcla posterior y llamarlo ahorro sparse.
- `lax.switch`/`cond` requieren revisión de sus transformaciones: aplicar `vmap` sobre decisiones diferentes puede convertir el programa en selección y evaluar varias ramas. Comprobar el ejecutable y las trazas.
- Para entrenamiento batch 2 con tareas mezcladas, preservar el muestreo uniforme y resolver la ejecución por ejemplo o por grupos. Registrar overhead; no cambiar silenciosamente el sampler a batches homogéneos. Si se propone eso como optimización posterior, declararlo como cambio experimental separado.
- Separar gradientes y actualizaciones: comprobar que módulos congelados no cambian por weight decay/momentum. Definir y documentar también el tratamiento del optimizador de expertos no seleccionados.

Antes de entrenamiento largo, medir S1/S2 sin profesor en operaciones, tiempo, memoria y parámetros. Los pesos nuevos pueden servir para comprobar shapes y costo, pero no para sacar conclusiones de calidad. Si hay mejora en una dimensión y retroceso en otra, registrar el intercambio; no descartar una variante sólo porque no reduzca latencia. Si no aparece una reducción relevante en ninguna dimensión, revisar tamaño/ubicación antes de invertir en recuperación de calidad.

### B4. Recuperación de calidad

1. Usar el shared original congelado como profesor y los pares limpios como supervisión principal.
2. Comenzar entrenando el módulo o expertos sustitutos con el resto del estudiante congelado, sin cortar gradientes necesarios para alcanzar los módulos nuevos.
3. Combinar la pérdida de reconstrucción existente con una pérdida de destilación sobre salida final; hacer opcional una comparación de features en el punto sustituido cuando shapes y escalas sean compatibles.
4. Después, si hace falta, descongelar brevemente módulos cercanos con LR bajo. Aplicar la misma política a S1/S2.
5. Definir antes de las corridas comparativas un presupuesto acotado e igual de pasos, seeds, LR/schedule y pérdidas. El warm-up necesario se fija a partir de pruebas cortas en train/val, no del test.
6. Seleccionar checkpoints por el criterio de validación documentado. Conservar PSNR/SSIM por tarea, macro y ponderados y revisar especialmente enhance.

La destilación añade trabajo durante entrenamiento. Registrar profesor + estudiante y el preentrenamiento reutilizado como contexto; no concluir que entrenar desde cero es más barato por tener una inferencia más pequeña. Medir por separado la inferencia del estudiante aislado y el costo de su adaptación/entrenamiento; no reducir ambas dimensiones a una sola conclusión de ahorro.

## 6. Verificación y decisión

Comprobaciones relevantes:

- Carga correcta de las referencias y reproducción de salida final al optimizar sólo su implementación.
- Shapes finitas y compatibles en todas las etapas/tareas; un paso con gradientes útiles en los módulos nuevos.
- Equivalencia de la ruta selectiva con una implementación de referencia que calcula todos los expertos y selecciona la misma tarea. Usar esa referencia sólo para verificar semántica.
- Comprobación mediante perfil/ejecutable de que los expertos internos no seleccionados y el bloque eliminado no se ejecutan en la inferencia medida.
- Parámetros congelados conservados; nueva ejecución usa pesos de las rutas esperadas.
- Tabla de ahorro real respecto a **S0 y B0**, junto con diferencias de PSNR/SSIM por tarea y agregadas. Puede haber ahorro respecto de shared y no respecto del baseline.

Fórmulas: ahorro de tiempo (%) = `100*(t_referencia-t_variante)/t_referencia`; aceleración = `t_referencia/t_variante`. Aplicar equivalentes a FLOPs/memoria sólo con magnitudes comparables. PSNR se reporta en diferencias dB, no como porcentaje de mejora visual.

No usar el 15–20 % de reducción de latencia mencionado anteriormente como meta de esta fase. **Primero diagnóstico y pruebas; después elección de la dimensión a optimizar y de una meta realista.** Presentar una tabla de calidad frente a FLOPs, tiempo, memoria y parámetros para las variantes ejecutadas. Incluir dificultad de implementación, recursos necesarios y compromisos observados para decidir qué mejora conviene continuar. Fijar la tolerancia de calidad antes de la evaluación final y reportar cualquier intercambio entre tareas.

No diseñar ni seleccionar nuevas variantes con test. El test actual ya informó esta nueva fase, por lo que una reevaluación futura sobre él no será un nuevo test completamente ciego. Declararlo; si se necesita una afirmación confirmatoria independiente, acordar un conjunto final no utilizado. Usar validación para las decisiones de esta fase.

## 7. Entregables y límites del primer encargo

Entregar primero:

1. `benchmark_maxim_baseline_shared.ipynb`, autocontenido para Colab y con rutas/configuración visibles al inicio.
2. Funciones reutilizables dentro del repositorio sólo cuando eviten duplicación material y el notebook pueda importarlas de manera reproducible.
3. Validaciones estáticas y sintéticas realizadas por Codex, junto con una lista clara de las celdas y comprobaciones que requieren GPU, Drive, checkpoints o datasets.
4. Instrucciones breves de ejecución y la ubicación esperada de `benchmark_results.json`, `diagnostico_costo_maxim.md` y trazas.

El primer encargo **no incluye** ejecutar el notebook, producir mediciones reales, identificar costos dominantes, elegir el bloque a sustituir, implementar S1/S2 ni entrenar/adaptar modelos. Esos pasos requieren primero los artefactos de la ejecución en Colab.

Después de que el usuario ejecute el diagnóstico y entregue sus resultados, revisar el perfil y preparar un segundo encargo acotado: generar los notebooks de S1/S2 y su adaptación, conservando checkpoints, JSONs y notebooks históricos. El usuario volverá a ejecutarlos en Colab. Sólo después de revisar esas salidas se resumirá qué se logró optimizar y qué resulta más viable antes de planear corridas extensas. Mantener el plan general de tesis actualizado con esta nueva dirección y con el test ya completado cuando corresponda.

No iniciar una búsqueda amplia de arquitecturas, cambios de dataset, nuevos routers o muchas métricas simultáneamente. Los adapters/LoRA quedan como alternativas futuras; no son necesarios para el primer experimento de reemplazo interno.

## 8. Referencias técnicas

- MAXIM, arquitectura original: https://research.google/pubs/maxim-multi-axis-mlp-for-image-processing/
- Implementación oficial, útil como referencia pero distinta del fork de tesis: https://github.com/google-research/maxim/blob/main/maxim/models/maxim.py
- Benchmarking/profiling JAX: https://docs.jax.dev/en/latest/201/profiling.html
- Compilación y análisis de costo: https://docs.jax.dev/en/latest/aot.html
- Reserva de memoria GPU: https://docs.jax.dev/en/latest/gpu_memory_allocation.html
- Selección condicional: https://docs.jax.dev/en/latest/_autosummary/jax.lax.switch.html
- Evaluación de ramas en selección: https://docs.jax.dev/en/latest/_autosummary/jax.lax.select.html

## Instrucción para el próximo encargo

Pendiente de que el usuario entregue `benchmark_v2_results.json`, `diagnostico_v2_compresion.md` y la carpeta `hlo/` del benchmark v2. Con esos artefactos:

1. Verificar las comprobaciones del JSON: selección efectiva de la cabeza y del banco, equivalencias, pérdida conservada con rematerialización.
2. Tabular para B0, S0_sel, S1 y S2 los parámetros totales y por tarea, FLOPs previos a la optimización, latencia, temporales de memoria y costo del paso de entrenamiento. Comparar contra S0_sel y contra B0.
3. Elegir el ancho d y decidir si la sustitución se amplía a otros niveles, con el criterio de tamaño acordado.
4. Generar el notebook de adaptación de S1 y S2 descrito en B4: profesor shared congelado, pérdida de reconstrucción más destilación de salida final, primero sólo módulos nuevos entrenables y presupuesto de pasos igual para S1 y S2. Seleccionar por validación, nunca por test.
5. No ejecutar entrenamiento ni benchmarks: el usuario los corre en Colab.

## Instrucción inicial para Codex, encargo del 8 de octubre ya completado

Lee este handoff y las instrucciones del repositorio. Para este primer encargo, inspecciona la construcción S-2 y el forward shared real únicamente hasta donde sea necesario para **generar** `benchmark_maxim_baseline_shared.ipynb`. El notebook debe quedar listo para que el usuario lo ejecute en Colab con los checkpoints baseline/shared fijados y debe implementar: restauración estricta, parámetros totales/activos/entrenables, FLOPs estimados cuando el backend lo permita, latencia sincronizada, memoria máxima, inferencia y paso de entrenamiento separados, profiling separado del cronometraje y escritura reanudable de `benchmark_results.json` y `diagnostico_costo_maxim.md`.

No ejecutes benchmarks, profiling, entrenamiento ni evaluación sobre los checkpoints reales. No inventes mediciones ni conclusiones. Valida solamente la estructura del notebook, su sintaxis y funciones puras o casos sintéticos livianos que puedan ejecutarse sin GPU, Drive ni checkpoints, y deja explícitas las celdas pendientes de Colab. **No implementes todavía la sustitución interna, S1/S2 ni notebooks de entrenamiento:** espera a que el usuario ejecute el diagnóstico y comparta sus resultados para definir el siguiente encargo.
