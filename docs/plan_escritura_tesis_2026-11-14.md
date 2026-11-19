# Plan de escritura de la tesis

Creado el 14 de noviembre de 2026. Documento vivo: el estado, los pendientes y el registro de avances se actualizan acá a medida que se escribe. Las entradas nuevas del registro van arriba, con fecha.

## 0. Estado al 14/11

- **La etapa experimental está cerrada.** Los experimentos de compresión terminaron el 4/11 y los directores aprobaron los resultados: los expertos por tarea permiten achicar MAXIM sin perder calidad. No se agregan variantes nuevas. Lo único que queda por correr son mediciones de costo para la tesis (sección 5.2).
- **Fuente de la tesis:** `tesis/main.tex`, con la plantilla de FaMAF (`MastersDoctoralThesis.cls`), y la bibliografía en `tesis/list.bib`. Se compila en Overleaf. `tesis/ref.bib` es la bibliografía de la versión anterior y queda sólo como respaldo.
- **Qué hay escrito:**

| Parte | Estado |
|---|---|
| Resumen, agradecimientos, dedicatoria | Texto de la plantilla, sin escribir |
| Cap. 1, Introducción | **Desactualizada.** Describe el MoE con ruteador aprendido y sin etiqueta de tarea, que ya no es el resultado principal. Tiene bloques duplicados: líneas ~203–219, la ecuación de las líneas 228–235, el párrafo de las líneas 248–258 y un objetivo repetido en las líneas 274–279. Hay que reescribirla (sección 4). |
| Cap. 2, Fundamentos teóricos | Vacío |
| Cap. 3, Herramientas | Escrito: Python, NumPy, JAX, Flax, tf.data, Pillow, Colab, Drive y Matplotlib. Falta agregar XLA, Optax y Orbax si se mencionan en otros capítulos. |
| Resto | Sin crear |

- **Título actual:** "Restauración de imágenes mediante MAXIM y Mixture of Experts". Decisión abierta: si se mantiene o si pasa a reflejar la compresión (sección 7).

## 1. Historia del trabajo, insumo para la introducción

La introducción resume estas tres etapas sin detalle técnico. Los números de cada etapa salen de protocolos distintos y **no se comparan entre etapas**.

**Etapa 1, junio–julio: MoE con ruteador aprendido.** MAXIM S-2 con cinco expertos chicos y un ruteador que decide, para cada imagen, cuánto pesa cada experto a partir de la imagen de entrada. No se le indica la tarea; se espera que aprenda a reconocer la degradación. Se encontró y corrigió un error de alineación en el recorte de los pares de entrenamiento. Con la corrida corregida, el MoE converge pero no supera al MAXIM multitarea comparable: 23,4 frente a 24,8 dB en la validación de entonces. Además, el ruteador termina repartiendo el peso por igual entre los expertos, así que no hay especialización. Fuentes: `runs_logs/moe_outputs/`, `runs_logs/no_moe_outputs/` y `tesis/eval_metrics_output.txt`.

**Etapa 2, agosto–octubre: corrección del protocolo y expertos por tarea.** Detalle en `docs/plan_correccion_moe.md`.
- Se auditaron los datos, se amplió denoise con SIDD y se creó un split train/val/test agrupado: 20.523 / 2.303 / 4.730 pares.
- Se pasó a muestreo uniforme por tarea y se conservó la supervisión profunda de MAXIM.
- Se probaron varias formas de expertos: con la tarea conocida (*oracle*), latentes K=8 con top-2 y token-choice K=8.
- La versión elegida fue la cabeza compartida con expertos residuales por tarea (shared). En el test del 4/10, shared y el baseline dan la misma calidad: +0,009 dB de PSNR macro.

**Etapa 3, octubre–noviembre: los expertos como herramienta de compresión.** Detalle en `docs/resumen_optimizacion_maxim_expertos_2026-10-28.md` y en el handoff `docs/handoff_optimization_maxim_moe_2026-10-08.md`.
- Un diagnóstico de costo mostró que el modelo está limitado por memoria y eligió el bloque a reemplazar: las convoluciones RCAB del nivel más profundo, con el 16,7 % de los parámetros.
- Se reemplazaron por módulos chicos: uno común (S1) o cinco expertos por tarea (S2). Se adaptaron por destilación desde shared, con dos semillas.
- **Resultado principal:** S2_d32 baja 13,5 % los parámetros y 7,3 % los FLOPs frente al baseline, con la misma calidad en test que shared. S1, con el mismo costo de ejecución, pierde calidad (S2 − S1 = +0,066 dB, con un IC que excluye el cero).
- Llevar los expertos a más niveles es un compromiso: S2_d32_l12 baja 15,6 % los FLOPs pero pierde 0,26 dB en derain.
- En todas las versiones la latencia baja como mucho 4 %.

**Mensaje de la tesis.** Agregar expertos por tarea a MAXIM no mejoró la calidad. Pero esa especialización permite reemplazar una parte pesada de la red por módulos mucho más chicos sin perder calidad, cosa que un módulo común del mismo costo no logra. El ahorro es de tamaño y de operaciones, no de tiempo en GPU.

## 2. Pautas de escritura

- **Lector:** alguien que sabe computación pero no necesariamente redes neuronales. Cada concepto fundamental se explica la primera vez que aparece, en dos o tres oraciones y con una referencia para profundizar: neurona, capa, convolución, función de pérdida, descenso por gradiente, *backpropagation*, normalización, conexión residual, atención, MoE, destilación, PSNR/SSIM, bootstrap. El capítulo 2 los desarrolla y los demás capítulos remiten a él.
- **Términos en inglés:** en itálica la primera vez, con la traducción si existe ("mezcla de expertos, *Mixture of Experts* (MoE)"). Después se usa siempre el mismo término. Hay que fijar un glosario (sección 6) y respetarlo.
- **Formato del fuente:** en `main.tex`, cada párrafo va en una sola línea, sin cortes manuales, para que sea fácil corregirlo a mano.
- **Números:** coma decimal y punto de miles, como en los resúmenes. PSNR con dos o tres decimales e intervalos de confianza donde existan. Hay que decir siempre si un número es de validación o de test.
- **Honestidad metodológica:**
  - Separar lo decidido en validación de lo confirmado en test.
  - Decir qué no se probó, por ejemplo GELU contra LeakyReLU o la depthwise 5×5.
  - Contar el resultado negativo de la etapa 1 sin esconderlo.
- **Trazabilidad:** cada número de la tesis sale de un archivo en `runs_logs/` o de un resumen en `docs/`. Al escribir, anotar la fuente en un comentario LaTeX (`% fuente: runs_logs/...`).
- **Figuras:** cada figura se genera con un script en `tesis/` que lee de `runs_logs/`, como `make_figs.py`. El título va en el caption, no dentro de la figura.

## 3. Estructura propuesta

Basada en la tesis de FaMAF de Meriles (`tesis/referencias/Meriles,Emanuel. Predicción temprana de tendencias en redes sociales y contenido.pdf`, 48 páginas, mismo director). Su estructura es:
- Introducción de una página y media, sin lista de objetivos ni párrafo de organización.
- Fundamentos teóricos, que cierran con una sección de trabajos previos.
- Herramientas.
- Conjunto de datos.
- Un capítulo de experimentos con configuración y resultados por modelo.
- Conclusiones y trabajo futuro.

Seguimos el mismo esquema, con dos capítulos de experimentos porque hubo dos etapas.

| # | Capítulo | Contenido | Estado |
|---|---|---|---|
| 1 | Introducción | Una página como máximo: problema, enfoque y resultado principal, sin detalle técnico (sección 4) | A reescribir, después de los gráficos |
| 2 | Fundamentos teóricos | Restauración y degradaciones; redes neuronales y entrenamiento; convoluciones, incluidas 1×1 y depthwise; activaciones; normalización y residuales; U-Net; atención de canal y RCAB; MLP, gMLP y MAXIM; MoE; destilación; compresión de redes; PSNR/SSIM; bootstrap pareado. **Cierra con trabajos previos:** estado del arte y derivados de MAXIM (sección 5.3) | Vacío |
| 3 | Herramientas | Ya escrito | Revisar al final |
| 4 | Conjunto de datos | Datasets por tarea, split agrupado, muestreo, recorte, métricas macro y ponderada, uso de validación y test | Sin escribir |
| 5 | Expertos por tarea en MAXIM | Configuración y resultados. Etapa 1 resumida como antecedente; etapa 2: variantes, shared, test del 4/10 | Sin escribir |
| 6 | Compresión con expertos | Configuración: diagnóstico de costo, elección del bloque, módulo nuevo, S1/S2, adaptación por destilación, criterio fijado antes de correr. Resultados: calidad en val y test con semillas e intervalos; costo en parámetros, FLOPs, latencia y entrenamiento; versiones por niveles; gráficos de calidad frente a compresión | Sin escribir |
| 7 | Conclusiones y trabajo futuro | Por qué la especialización conserva calidad, por qué los FLOPs no se convierten en tiempo, límites; depthwise 5×5, clasificador de tarea, compresión de gMLP | Sin escribir |
| A | Apéndices, opcional | Hiperparámetros, conteos por tarea, protocolos y verificaciones | — |

El capítulo 2 es el más largo y el que más referencias necesita.

**Índice escrito el 17/11 en `tesis/main.tex`.** Desde el 18/11 el índice del documento lista sólo capítulos y secciones (`tocdepth` = 1). Las subsecciones siguen en el texto, con su `\label` y sus comentarios, pero no aparecen en el índice.

1. Introducción
2. Fundamentos teóricos
   - 2.1 Restauración de imágenes
   - 2.2 Redes neuronales
   - 2.3 Redes convolucionales
   - 2.4 MAXIM
   - 2.5 Mezcla de expertos
   - 2.6 Compresión de redes neuronales
   - 2.7 Evaluación
   - 2.8 Trabajos previos
3. Herramientas
   - 3.1 Lenguaje y cómputo numérico
   - 3.2 Aprendizaje profundo
   - 3.3 Carga y procesamiento de datos
   - 3.4 Ejecución y almacenamiento
   - 3.5 Visualización de resultados
4. Conjunto de datos
   - 4.1 Fuentes por tarea
   - 4.2 Auditoría y depuración de pares
   - 4.3 División en entrenamiento, validación y test
   - 4.4 Muestreo y preprocesamiento
   - 4.5 Protocolo de evaluación
5. Expertos por tarea en MAXIM
   - 5.1 Primera aproximación: expertos con ruteador
   - 5.2 Expertos por tarea con cabeza compartida
   - 5.3 Discusión
6. Compresión de MAXIM con expertos
   - 6.1 Diagnóstico de costo
   - 6.2 Reemplazo de los bloques RCAB
   - 6.3 Adaptación por destilación
   - 6.4 Resultados
   - 6.5 Expertos en más niveles
   - 6.6 Calidad frente a compresión
7. Conclusiones y trabajo futuro
   - 7.1 Conclusiones
   - 7.2 Trabajo futuro

**Para revisar al escribir Herramientas:** el párrafo de Matplotlib menciona mapas de calor del uso de los expertos y del ruteador, que ya no son los gráficos centrales.

## 4. Introducción: plan

Se escribe **después de los gráficos** (sección 5.1), por pedido del 14/11. **Largo máximo: una página**, unas 400 a 450 palabras, sin figuras, sin ecuaciones y sin listas. Se reescribe entera. Del texto actual sólo sirve, condensada, la motivación de las líneas 130–167.

**Esquema, cuatro párrafos:**
1. **Problema:** las imágenes reales llegan degradadas por desenfoque, neblina, ruido, lluvia o poca luz, y restaurarlas importa. Un único modelo para todas las degradaciones es atractivo, pero cada una pide transformaciones distintas.
2. **Enfoque:** MAXIM como modelo base, con la cita a Tu et al. 2022, más módulos especializados por tarea (expertos) dentro de una red compartida. Se explica con una idea intuitiva, sin la ecuación del ruteador.
3. **Qué se hizo y qué se encontró:** las tres etapas en una o dos oraciones cada una (sección 1). Los expertos no mejoraron la calidad, pero su especialización permite reemplazar una parte pesada de la red por módulos mucho más chicos.
4. **Resultado principal:** −13,5 % de parámetros y −7,3 % de FLOPs sin perder calidad; un módulo común del mismo costo sí pierde; el tiempo de ejecución casi no cambia. Puede cerrar con una oración sobre el orden de los capítulos, o sin ella, como en la tesis de Meriles.

**Hay que sacar:**
- la ecuación con el ruteador y la frase de que el modelo funciona "sin requerir una etiqueta que indique la tarea": el resultado principal usa la tarea conocida;
- la lista de objetivos;
- todos los bloques duplicados.

Si los directores piden objetivos explícitos, van en un párrafo corto al final del capítulo 2 o al principio del 5, no en la introducción. Conviene contrastarlos con el plan de tesis presentado (`../Plan de tesis.docx`).

## 5. Pendientes pedidos por los directores

### 5.1 Gráfico de calidad frente a compresión

**Recomendación: un gráfico de dispersión como figura principal y un heatmap como complemento.**

- **Por qué no un heatmap como figura principal:** un heatmap muestra bien una matriz de dos factores categóricos con un valor por celda. La relación entre calidad y compresión es entre dos magnitudes continuas, y en un heatmap se pierden la escala del eje de compresión y los intervalos de confianza.
- **Figura principal, dispersión:**
  - **eje x:** reducción de parámetros frente a B0, en %. Un segundo panel, o el tamaño del punto, puede mostrar los GFLOP.
  - **eje y:** Δ PSNR macro frente a shared, con barras del IC 95 %.
  - **puntos:** un punto por versión, con S1 y S2 de distinto color y los niveles 2, 1–2 y 0–2 unidos por una línea, para que se vea cómo crece la brecha S2 − S1.
  - **referencia:** una banda horizontal en −0,05 dB, el criterio fijado de antemano.
  - **resultado esperado:** S2_d32 queda arriba del criterio y las demás versiones caen por debajo a medida que se comprime más. Es el resumen visual de la etapa 3.
- **Complemento, heatmap por tarea:** versiones en filas, tareas en columnas y Δ PSNR por tarea frente a shared en cada celda, con escala divergente centrada en 0. Este sí es un buen uso del heatmap: muestra dónde se va la pérdida (el nivel 1 afecta a derain, el nivel 0 a dehaze) y que S2 la reparte mejor que S1.
- **Datos:** `runs_logs/compresion_validacion_niveles/`, `runs_logs/compresion_test_niveles/` (diferencias e IC) y `runs_logs/benchmark_maxim_niveles/` (parámetros y FLOPs).

**Hecho el 14/11.** `tesis/fig_calidad_vs_compresion.py` lee esos JSON sin necesidad de JAX ni Colab y genera tres figuras en `tesis/figs/`, cada una con validación y test:

| Archivo | Qué muestra |
|---|---|
| `fig_calidad_vs_flops.pdf` | Δ PSNR macro frente a shared contra la reducción de GFLOP. S1 y S2 de cada nivel quedan en la misma vertical porque cuestan lo mismo, así que la brecha entre ambos se ve directamente. **Candidata a figura principal.** |
| `fig_calidad_vs_parametros.pdf` | Lo mismo contra la reducción de parámetros. S2 queda a la izquierda de S1 porque guarda cinco expertos y los puntos se amontonan entre 13 y 20 %. Probablemente sobra. |
| `fig_calidad_por_tarea.pdf` | Heatmap de Δ PSNR por tarea frente a shared, en validación y en test. |

Convenciones de las tres figuras:
- los marcadores llenos son el promedio de dos semillas y los huecos, una sola semilla;
- las barras son el IC 95 % del bootstrap pareado;
- la banda gris marca el criterio de −0,05 dB;
- B0 y shared aparecen como referencia.

**Decidido el 17/11:**
- se descarta el gráfico de parámetros;
- quedan `fig_calidad_vs_flops.pdf` y `fig_calidad_por_tarea.pdf`, **sólo con test**, en un único panel cada uno, para la sección 6.6;
- el script conserva `--split val`, que genera las versiones de validación con sufijo `_val` por si hacen falta en un apéndice.

### 5.2 Tiempo de entrenamiento de los modelos comprimidos

La pregunta de los directores tiene dos partes que conviene separar.

**a) ¿El modelo comprimido entrenado entero es más rápido por paso?** Según lo ya medido, no. Benchmark v2 en L4, batch 2, 256×256: el paso completo de S2_d32 tarda 464,8 ms y el de B0 445,7 ms, un 4 % más. La razón es la misma que en inferencia: el tiempo lo dominan los accesos a memoria y los lanzamientos de kernels, no los FLOPs. Además, la memoria de entrenamiento es casi toda de activaciones (~12 GiB frente a 57 MB de pesos), y achicar los pesos casi no la cambia. Fuente: `runs_logs/benchmark_maxim_v2_compresion_results/`.

**b) ¿Cuánto cuesta obtener el modelo comprimido?** Acá sí hay ahorro.
- La adaptación entrena sólo los módulos nuevos: 227,5 ms por paso, −49 %, con 5,4 GiB frente a 12,0 GiB.
- Son 5 épocas de 2.000 pasos (10.000 pasos), contra 60.000 pasos del baseline desde cero y 20.000 más de su continuación.
- **Pero la adaptación necesita al profesor ya entrenado:** el ahorro es el costo marginal de comprimir un modelo existente, no un reemplazo del entrenamiento. Así hay que presentarlo.

**Tiempo real de las adaptaciones, medido en los logs el 14/11.** El dato contradice la estimación anterior.
- Los `history.json` de `runs_logs/compresion_adaptacion/` guardan los segundos de cada época, con la validación incluida. Restando la validación, que figura en `training.log` (unos 100 s), queda el entrenamiento.
- Todas las adaptaciones corrieron en L4 y tardaron entre 1,2 y 1,6 horas en total.
- Por paso de pared:

| Adaptación | ms por paso |
|---|---:|
| S2_d32 v1 / v2 | 432 / 432 |
| S2_d32_l12 v1 / v2 | 464 / 439 |
| S2_d32_l012 v1 | 443 |
| S1_d32 v1 / v2 | 479 / 351 |
| S1_d32_l12 v1 | 335 |
| S1_d32_l012 v1 | 367 |

- **Son unos 430 ms por paso, no 227.** El paso del benchmark v2 no incluía el forward del profesor, la destilación ni la carga de datos. Un paso real de adaptación cuesta casi lo mismo que un paso completo de B0 (446 ms en el benchmark). La ventaja real de la adaptación sería entonces la cantidad de pasos, 10.000 frente a 60.000, y la memoria, no el costo por paso.
- La variación entre corridas iguales (S1_d32: 479 frente a 351 ms) sugiere que la carga de datos desde Drive pesa en el tiempo de pared.
- El log del baseline no tiene tiempos por época, así que no hay tiempo de pared de B0 para comparar.

**Benchmark preparado el 14/11:** `notebooks/benchmark/benchmark_entrenamiento.ipynb`.
- Mide en un mismo proceso en L4, batch 2 y 256×256:
  - `B0:full`;
  - `S2_d32:full`, el modelo comprimido entrenado entero;
  - `S2_d32:new_only`, la referencia del benchmark v2;
  - `S2_d32:adapt` y `S2_d32_l12:adapt`, el paso real de adaptación con profesor y destilación, copiado de `adaptacion_compresion.ipynb`.
- El reporte proyecta horas de GPU para cada corrida completa: B0 desde cero, B0 +10, S2_d32 desde cero y las dos adaptaciones.
- Se probó de punta a punta en CPU con checkpoints sintéticos, incluida la reanudación, y tiene test estático.
- Escribe en `ckpts/benchmark_entrenamiento/`.
- Es una medición de costo, no una variante nueva, así que no reabre los experimentos.

**Resultados del benchmark, 17/11.** En `runs_logs/benchmark_entrenamiento/`, L4, commit 944f1ee, todas las verificaciones OK. Mediana por paso, batch 2, 256×256:

| Paso | ms | vs B0 | Memoria temporal | Entrenables | Horas de GPU de la corrida completa |
|---|---:|---:|---:|---:|---:|
| B0, completo | 463,4 | — | 11,8 GiB | 14,18 M | 7,72 (60.000 pasos) + 2,57 (+10 épocas) |
| S2_d32, completo, sin profesor | 465,0 | +0,3 % | 11,8 GiB | 12,26 M | 7,75 (hipotético, 60.000 pasos) |
| S2_d32, adaptación real | 329,7 | −28,9 % | 5,3 GiB | 0,35 M | 0,92 (10.000 pasos) |
| S2_d32_l12, adaptación real | 338,0 | −27,1 % | 5,7 GiB | 0,53 M | 0,94 (10.000 pasos) |
| S2_d32, sólo módulos nuevos, sin profesor | 229,0 | −50,6 % | 5,3 GiB | 0,35 M | referencia del benchmark v2 |

- **El modelo comprimido no es más rápido de entrenar entero.** Su paso cuesta lo mismo que el de B0: +0,3 %, con la misma memoria. Tener 13,5 % menos parámetros no cambia el tiempo, por la misma razón que en inferencia: dominan los accesos a memoria, y la memoria la ocupan las activaciones, no los pesos.
- **Obtenerlo por adaptación es barato.** El paso real cuesta 29 % menos que uno de B0, con menos de la mitad de memoria. Pero el ahorro grande está en la cantidad de pasos: 0,92 horas de GPU frente a 7,72 de entrenar B0 desde cero, el 12 %.
- **El profesor y la destilación agregan unos 100 ms por paso** (330 frente a 229). Por eso la estimación de "la mitad del tiempo" era optimista.
- **La carga de datos y el resto agregan otros ~100 ms.** El tiempo de pared real de las adaptaciones de S2_d32 fue 432 ms por paso, frente a 330 de GPU: un ~30 % que se va en carga de datos desde Drive, transferencias y registro. Cada adaptación tardó en total ~1,45 h de pared con la validación incluida.
- **Cómo presentarlo en la tesis:** la adaptación no reemplaza el entrenamiento, porque necesita al profesor ya entrenado (baseline 60.000 pasos + shared 20.000). Es el costo marginal de obtener la versión comprimida de un modelo existente: menos de una hora de GPU y menos de la mitad de memoria, que cabe en una GPU más chica.

**Entrenar S2_d32 desde cero no se corre.** Respondería si el modelo chico converge en menos pasos, pero es caro (~60.000 pasos) y por paso no es más rápido. Se explica en la tesis.

### 5.3 Estado del arte y derivados de MAXIM

Es una búsqueda pendiente. La lista siguiente son candidatos a verificar: datos bibliográficos, qué tareas cubren y si reportan resultados multitarea. No citar nada sin haber leído al menos el resumen.

- **Restauración general:** MPRNet (Zamir et al., 2021), HINet (Chen et al., 2021), SwinIR (Liang et al., 2021), Uformer (Wang et al., 2022), Restormer (Zamir et al., 2022), NAFNet (Chen et al., 2022) y MambaIR (Guo et al., 2024).
- **Un modelo para varias degradaciones (*all-in-one*):** AirNet (Li et al., 2022), IDR (Zhang et al., 2023), PromptIR (Potlapalli et al., 2023), DA-CLIP (Luo et al., 2024), InstructIR (Conde et al., 2024) y DiffUIR (Zheng et al., 2024).
- **MoE en restauración:** buscar trabajos recientes que usen expertos para restauración multitarea, por ejemplo MoCE-IR (Zamfir et al., 2025). Es lo más cercano a esta tesis.
- **Derivados de MAXIM:** MaxViT (Tu et al., 2022), de los mismos autores, con atención multieje. Buscar trabajos que extiendan o compriman MAXIM y los que lo usan como comparación.
- **Restauración eficiente y compresión:** destilación aplicada a restauración, arquitecturas livianas, poda y cuantización. Sirve para ubicar el aporte de la etapa 3.
- **Advertencia para el capítulo:** los números de esta tesis no son comparables con los de los papers. Los splits son propios, se evalúa con un recorte central de 256 y el entrenamiento es multitarea con presupuesto acotado. El capítulo compara enfoques, no tablas de PSNR.

### 5.4 Referencias

`tesis/list.bib` tiene hoy: MAXIM, Jacobs 1991, Shazeer 2017 y las herramientas (Python, NumPy, JAX, Flax, TensorFlow, Pillow, Colab y Matplotlib). La tabla lista las que harían falta por tema. Hay que verificar cada entrada (autores, año, venue, DOI o arXiv) antes de agregarla.

| Tema | Referencias candidatas |
|---|---|
| Libro general de aprendizaje profundo | Goodfellow, Bengio y Courville, *Deep Learning*, 2016 |
| Entrenamiento | Rumelhart et al. 1986 (backpropagation); Kingma y Ba 2015 (Adam); Loshchilov y Hutter 2019 (AdamW) |
| Convoluciones y arquitecturas | LeCun et al. 1998; Ronneberger et al. 2015 (U-Net); He et al. 2016 (ResNet); Chollet 2017 (separables); Howard et al. 2017 y Sandler et al. 2018 (MobileNet v1/v2); Liu et al. 2022 (ConvNeXt) |
| Activaciones y normalización | Maas et al. 2013 (LeakyReLU); Hendrycks y Gimpel 2016 (GELU); Ba et al. 2016 (LayerNorm) |
| Atención de canal y RCAB | Hu et al. 2018 (Squeeze-and-Excitation); Zhang et al. 2018 (RCAN, origen del RCAB) |
| MLP y MAXIM | Tolstikhin et al. 2021 (MLP-Mixer); Liu et al. 2021 (gMLP); Tu et al. 2022 (MAXIM, ya está) |
| MoE | Jacobs et al. 1991 (ya está); Jordan y Jacobs 1994; Shazeer et al. 2017 (ya está); Fedus et al. 2022 (Switch Transformer); Riquelme et al. 2021 (V-MoE) |
| Multitarea | Caruana 1997 |
| Destilación y compresión | Hinton et al. 2015; Romero et al. 2015 (FitNets, imitación de activaciones intermedias, como nuestra pérdida de bloques) |
| Métricas y estadística | Wang et al. 2004 (SSIM); Efron y Tibshirani 1993 (bootstrap) |
| Datasets | GoPro (Nah et al. 2017); RealBlur (Rim et al. 2020); RESIDE, que incluye SOTS y HSTS (Li et al. 2019); SIDD (Abdelhamed et al. 2018); lluvia: identificar el dataset de los archivos `rain_light`/`rain_heavy`; LOL (Wei et al. 2018); MIT-Adobe FiveK (Bychkovsky et al. 2011) |
| Herramientas que faltan | XLA, Optax, Orbax, si se mencionan |

## 6. Glosario

Fijar al escribir el capítulo 2. Propuesta inicial:

| Término en la tesis | Inglés / código | Nota |
|---|---|---|
| mezcla de expertos | Mixture of Experts, MoE | |
| experto | expert | |
| ruteador | router | Sólo en la etapa 1 |
| selección por tarea conocida | oracle routing | La tarea se pasa como entrada |
| cabeza compartida | shared head | S0, S0_sel |
| módulo común / banco de expertos | S1 / S2 | |
| profesor / estudiante | teacher / student | Destilación |
| baseline | B0 | Se deja en inglés |
| PSNR macro / ponderado | — | Promedio por tarea / por imagen |

## 7. Decisiones abiertas

1. Título de la tesis: si se mantiene o si incluye la compresión.
2. Cuánto espacio le da la tesis a la etapa 1, incluido el error del recorte: una sección del capítulo 5 o un antecedente breve.
3. ~~Qué gráficos de calidad frente a compresión quedan.~~ Decidido el 17/11: GFLOP y heatmap, sólo test (5.1).
4. ~~Si se corre la medición del paso real de adaptación.~~ Decidido el 14/11: se corre (5.2).
5. ~~Estado del arte como capítulo propio o dentro de Fundamentos.~~ Decidido el 14/11: dentro de Fundamentos, como trabajos previos, igual que Meriles.

## 8. Fuentes en el repositorio

| Para | Ver |
|---|---|
| Resumen de la etapa 3, con tablas y figuras | `docs/resumen_optimizacion_maxim_expertos_2026-10-28.md` y `.pdf` |
| Decisiones y registro de la etapa 3 | `docs/handoff_optimization_maxim_moe_2026-10-08.md` |
| Etapa 2 | `docs/plan_correccion_moe.md` |
| Diagnóstico de costo | `docs/analisis_benchmark_maxim_20261014.md` y `docs/analisis_benchmark_v2_20261019.md` |
| Resultados | `runs_logs/` (mapa en `docs/ESTRUCTURA.md`) |
| Splits y datasets | `datasets/splits_v2/split_manifest.json` y `scripts/create_grouped_splits.py` |
| Figuras de la etapa 1 | `tesis/make_figs.py`, `tesis/eval_metrics.py` y `tesis/fig_task_bars.py`, con salida en `tesis/figs/` |
| Código de los modelos | `maxim/models/maxim.py`, `moe.py`, `router.py` y `compression.py` |

## 9. Registro de avances

**19/11.**
- Se escribió la introducción en `main.tex`, unas 420 palabras en cinco párrafos: problema, enfoque, las tres etapas, resultado principal y un párrafo de una oración con la organización.
- Se reemplazó la introducción anterior entera, incluidos la ecuación del ruteador, los objetivos y los bloques duplicados.
- Falta comprobar en Overleaf que entre en una página. Si no entra, lo primero que se saca es el párrafo de organización.
- Las "casi ocho horas" son tiempo de GPU proyectado por el benchmark de entrenamiento (7,72 h), no tiempo de pared.
- Correcciones del usuario al párrafo de etapas:
  - Etapa 1: el ruteador decide a partir de la imagen y se espera que reconozca la degradación sin que se le diga la tarea. No decir que elegía "sin conocer la degradación".
  - Etapa 2: no presentar la separación train/val/test como novedad de esa etapa, porque es práctica estándar. En la tesis se cuenta como si los datos siempre hubieran estado divididos así.
  - Etapa 2 sí menciona la depuración de datos y las variantes probadas: más expertos, expertos compartidos y expertos especializados en características en lugar de tareas.
- Todos los párrafos de `main.tex` pasan a una línea cada uno (111 líneas unidas, sin cambios de texto).

**18/11.** El índice de la tesis lista sólo capítulos y secciones; las subsecciones quedan en el texto.

**17/11, tercera parte.**
- Gráficos definitivos: calidad frente a GFLOP y heatmap por tarea, sólo test; se descarta el de parámetros.
- Se escribió el índice de la tesis en `main.tex` (sección 3).
- Sigue la introducción.

**17/11, segunda parte.** Se ajustaron los gráficos de dispersión:
- el eje Y queda en [−0,75, +0,35], con más espacio en la parte positiva;
- S1_d32_l012 en validación (−1,12 dB) queda fuera de rango y se marca con una flecha y su valor;
- el eje X tiene menos margen vacío y la leyenda queda más cerca;
- el heatmap tiene menos espacio entre paneles.

Se descartaron las salidas de Colab del notebook de entrenamiento.

**17/11.** Se corrió `benchmark_entrenamiento.ipynb` en L4 (resultados en 5.2). Confirma que:
- el modelo comprimido no se entrena más rápido entero;
- el paso real de adaptación cuesta 29 % menos que el de B0, no 50 %;
- la adaptación completa son 0,92 horas de GPU, frente a 7,72 de entrenar B0 desde cero.

**14/11, segunda parte.**
- La introducción baja a una página como máximo y se escribe después de los gráficos.
- Se adopta la estructura de la tesis de Meriles.
- Se generaron los tres gráficos de calidad frente a compresión, con validación y test, para elegir.
- Los logs muestran que el paso real de adaptación cuesta unos 430 ms, no 227, así que el ahorro en entrenamiento hay que confirmarlo.
- Se preparó `benchmark_entrenamiento.ipynb`; falta correrlo en Colab.

**14/11.** Se crea este plan. La etapa experimental queda cerrada tras la reunión con los directores, que pidieron:
- un gráfico de calidad frente a compresión;
- comparar el tiempo de entrenamiento de los modelos comprimidos;
- investigar el estado del arte y los derivados de MAXIM;
- referencias en toda la tesis;
- un tono accesible para quien no conoce redes neuronales.

Se empieza por la introducción.
