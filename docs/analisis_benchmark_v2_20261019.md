# Análisis del benchmark v2: compresión de MAXIM con expertos

Fecha: 19 de octubre de 2026.

Fuente: `runs_logs/benchmark_maxim_v2_compresion_results/`. Protocolo `86d9c15980815dff46ca1621a844b0352ebbbad1299bd01f4937fda881510e6c`, commit `a4fce8b`, NVIDIA L4, JAX 0.10.2. Los FLOPs salen de `flops_previos_optimizacion_cpu.json`, explicado en la sección 2.

Sólo se mide costo. Los módulos nuevos de S1 y S2 tienen pesos aleatorios y no dicen nada sobre calidad.

## 1. Resumen

1. **Todas las verificaciones pasaron.** Selección efectiva de la cabeza y del banco, conteo de convoluciones, pérdida conservada con rematerialización y salidas finitas en todas las variantes.
2. **Seleccionar el experto funciona y no cuesta nada en inferencia.** S2 tiene la misma latencia y los mismos FLOPs que S1, que usa un único módulo común.
3. **El shared actual cuesta +5,6 % de latencia a 256 frente al baseline, medido en el mismo proceso.** Con selección efectiva baja a +1,5 %.
4. **S2 con d=32 reduce el modelo un 13,5 % en parámetros y un 7,3 % en FLOPs frente al baseline.** La latencia apenas baja, entre 0,5 % y 1,1 %. Es la reducción de tamaño que se buscaba, sin ahorro de tiempo, como se había previsto.
5. **d=64 no cuesta más que d=32 en FLOPs ni en latencia, pero agrega parámetros.** Con cinco expertos, S2 con d=64 tiene 342 mil parámetros más que con d=32.
6. **Adaptar sólo los módulos nuevos cuesta la mitad de un paso completo.** El paso tarda 49 % menos y usa 55 % menos memoria temporal.
7. **Las palancas grandes no son los expertos.** bf16 reduce la latencia entre 54 % y 64 %. La rematerialización baja 81 % la memoria de entrenamiento. Sacar los gMLP multi-eje baja 33 % a 40 % la latencia.

## 2. Corrección de la medición de FLOPs

El benchmark registró dos medidas de FLOPs y ninguna sirve para comparar:

- **El análisis del programa previo a la optimización devolvió vacío en GPU.**
- **El análisis del ejecutable compilado no cuenta las convoluciones de cuDNN.** Al quitar 16 convoluciones de S0_sel, sus FLOPs compilados no bajan. Esto también explica las inconsistencias de FLOPs del benchmark v1.

Para reemplazarlas, el análisis previo a la optimización se calculó en CPU con las mismas variantes de tamaño real. Usa sólo shapes abstractas: no inicializa ni ejecuta nada. Esa medida no depende del backend y es la que se usa en este documento. La sustitución baja 18,4 GFLOP a 256, contra 18,8 GFLOP de la estimación analítica.

Los bytes accedidos de ese mismo análisis corresponden al programa sin fusionar. No representan el tráfico real de memoria y no se usan.

## 3. Inferencia, batch 1

Latencias en mediana. Las variaciones son frente al baseline B0. La dispersión entre bloques fue de 1,8 % o menos a 256 y de 0,4 % o menos a 512.

| Variante | Parámetros | GFLOP a 256 | ms a 256 | Δ latencia a 256 | ms a 512 | Δ latencia a 512 |
|---|---:|---:|---:|---:|---:|---:|
| B0, baseline | 14,18 M | 216,3 | 41,58 | — | 254,0 | — |
| S0, shared actual | 14,27 M | 229,2 | 43,91 | +5,6 % | 263,2 | +3,6 % |
| S0_sel, shared con selección | 14,27 M | 218,9 | 42,22 | +1,5 % | 256,3 | +0,9 % |
| S1_d32 | 11,98 M | 200,5 | 41,14 | −1,0 % | 252,8 | −0,5 % |
| **S2_d32** | **12,26 M** | **200,5** | **41,11** | **−1,1 %** | **252,6** | **−0,5 %** |
| S2_d32, calculando los cinco | 12,26 M | 202,9 | 41,77 | +0,5 % | 258,8 | +1,9 % |
| S1_d64 | 12,05 M | 201,1 | 40,84 | −1,8 % | 253,0 | −0,4 % |
| S2_d64 | 12,60 M | 201,1 | 40,92 | −1,6 % | 252,9 | −0,4 % |
| Techo: sin las RCAB profundas | 11,91 M | 200,0 | 40,17 | −3,4 % | 252,0 | −0,8 % |
| Techo: sin cross-gating | 11,89 M | 185,7 | 33,68 | −19,0 % | 206,7 | −18,6 % |
| Techo: sin gMLP multi-eje | 10,25 M | 154,3 | 27,80 | −33,1 % | 153,0 | −39,8 % |
| B0 en bf16 | 14,18 M | 216,3 | 19,28 | −53,6 % | 91,4 | −64,0 % |
| S2_d32 en bf16 | 12,26 M | 200,5 | 18,86 | −54,6 % | 90,3 | −64,4 % |

Frente a S0_sel, que es el punto de partida de S1 y S2:

| Medida | S1_d32 | S2_d32 |
|---|---:|---:|
| Parámetros totales | −16,1 % | −14,1 % |
| Parámetros usados por tarea | −16,1 % | −16,1 % |
| FLOPs a 256 | −8,4 % | −8,4 % |
| Latencia a 256 | −2,6 % | −2,6 % |
| Latencia a 512 | −1,4 % | −1,4 % |

### Lectura

- **Una reducción de 7 % en FLOPs se traduce en 1 % de latencia.** Confirma que el modelo está limitado por memoria y lanzamiento de kernels, como mostró el v1. Ni siquiera quitar por completo esas convoluciones baja la latencia más de 3,4 %.
- **El costo de la especialización es sólo de parámetros guardados.** S2 tiene 277 mil parámetros más que S1, un 2,3 % del modelo, y en inferencia cuesta lo mismo. Calcular los cinco expertos en vez de seleccionar agrega 2,3 GFLOP y entre 1,6 % y 2,4 % de latencia.
- **El sobrecosto del shared actual era real.** El +6,5 % del v1 se confirma como +5,6 % en el mismo proceso. Casi todo viene de calcular cinco cabezas a resolución completa. La cabeza seleccionada que queda cuesta entre 0,9 % y 1,5 %.
- **bf16 es la palanca de latencia más grande.** El desvío numérico frente a float32 es de 0,0016 en promedio y hasta 0,011 en el peor píxel, sobre salidas en [0, 1]. No es una medida de calidad: hace falta evaluar PSNR/SSIM en validación.
- **Los gMLP multi-eje y cross-gating son las familias de costo dominante.** Sin ellos el modelo deja de ser MAXIM, así que sólo sirven como techo. Si en el futuro se busca latencia, son el lugar donde buscar.

## 4. Paso de entrenamiento, 256×256 con batch 2 mixto

Medido en secuencia, de la corrida que más memoria pide a la que menos, en el mismo proceso. La alternancia sólo se aplicó en inferencia, por un problema de fragmentación de memoria resuelto durante la corrida.

| Corrida | Entrenables | ms por paso | Δ vs B0 | Memoria temporal | Δ vs B0 | GFLOP |
|---|---:|---:|---:|---:|---:|---:|
| B0, completo | 14,18 M | 445,7 | — | 11,76 GiB | — | 1.271 |
| S0, completo | 14,27 M | 463,7 | +4,0 % | 12,15 GiB | +3,3 % | 1.348 |
| S0_sel, completo | 14,27 M | 454,1 | +1,9 % | 11,86 GiB | +0,8 % | 1.286 |
| S2_d32, completo | 12,26 M | 464,8 | +4,3 % | 11,82 GiB | +0,4 % | 1.176 |
| S1_d32, sólo módulos nuevos | 69 mil | 227,3 | −49,0 % | 5,28 GiB | −55,1 % | 750 |
| S2_d32, sólo módulos nuevos | 347 mil | 227,5 | −49,0 % | 5,27 GiB | −55,2 % | 750 |
| B0 con rematerialización | 14,18 M | 529,5 | +18,8 % | 2,22 GiB | −81,2 % | 1.645 |

- **La adaptación prevista es barata.** Entrenar sólo los módulos nuevos, con el resto congelado, cuesta la mitad de tiempo y memoria que un paso completo. S1 y S2 cuestan lo mismo.
- **Con batches de tareas mezcladas, entrenar S2 completo cuesta 2,3 % más que S0_sel, aun con 8,6 % menos FLOPs.** Es el costo de aplicar pesos distintos por ejemplo en el backward. Sólo importa si se entrena el modelo completo.
- **La rematerialización es una palanca de memoria muy fuerte.** Baja la memoria temporal de 11,8 a 2,2 GiB a cambio de 19 % más de tiempo por paso. Permitiría batches mucho mayores en la misma L4.
- **Las memorias son estáticas, de `memory_analysis()`.** No son picos medidos.

## 5. Recomendación

1. **Elegir d=32 para la adaptación.** Tiene los mismos FLOPs y latencia que d=64 y menos parámetros. Volver a d=64 sólo si la calidad de d=32 queda corta.
2. **Pasar a la recuperación de calidad de S1_d32 y S2_d32**, como describe B4 del handoff: profesor S0 congelado, reconstrucción más destilación de salida final y sólo los módulos nuevos entrenables al principio. Es la prueba central de la hipótesis: si con el mismo costo activo los especialistas conservan más calidad que el módulo común.
3. **Evaluar bf16 en validación como ablación separada.** Es barato porque es sólo inferencia, y es la mayor reducción de latencia medida. Se aplica igual a todas las variantes, así que no compite con los expertos.
4. **Declarar en la tesis qué reduce cada intervención.** Los expertos reducen tamaño y operaciones. bf16 reduce latencia. La rematerialización reduce memoria de entrenamiento.
5. **Dejar los gMLP y cross-gating como trabajo futuro** si se busca latencia con expertos.
