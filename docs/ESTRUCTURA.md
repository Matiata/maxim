# Estructura del repositorio

Fork de [google-research/maxim](https://github.com/google-research/maxim) usado para la tesis sobre MAXIM con expertos por tarea. Reorganizado el 15 de octubre de 2026.

| Carpeta o archivo | Contenido |
|---|---|
| `maxim/` | Paquete Python. Los notebooks de Colab lo instalan con `pip install -e .` e importan desde acá. |
| `maxim/models/` | `maxim.py` es MAXIM con opciones de compresión que vienen apagadas por defecto. `moe.py` y `router.py` contienen los MoE de la tesis. `compression.py` tiene el banco de expertos, la cabeza selectiva y el catálogo de variantes. `compat.py` es el parche de rematerialización. |
| `maxim/evaluation_metrics.py` | PSNR/SSIM por tarea, macro y ponderado, más el criterio de selección de checkpoints. |
| `maxim/run_eval.py`, `predict.py`, `trainer.py`, `test_maxim.py` | Scripts del repositorio original de Google. |
| `notebooks/entrenamiento/` | Entrenamiento en Colab: baseline S-2, baseline +10 épocas y MoE/shared. |
| `notebooks/evaluacion/` | Test final de baseline +10 contra shared +10. |
| `notebooks/benchmark/` | Diagnóstico de costo v1 y benchmark de compresión v2. |
| `tests/` | Pruebas estáticas de notebooks y pruebas en CPU de los modelos. Los tests que necesitan JAX se saltean si no está instalado. |
| `scripts/` | Creación y verificación de los splits train/val/test. |
| `runs_logs/` | Logs de entrenamiento y resultados descargados de Drive. `make_figs.py` lee de acá. |
| `docs/` | Plan de trabajo, handoff para Codex, análisis del benchmark y notas. |
| `tesis/` | Fuente LaTeX, bibliografía, figuras y los scripts que las generan. `tesis/referencias/` guarda PDFs de consulta. |
| `archivo/` | Scripts, notebooks y comandos obsoletos. Se conservan sólo como historial. |
| `datasets/` | Datos locales, ignorados por git salvo `splits_v2/split_manifest.json`. |

## Convenciones

- Los checkpoints y datasets completos viven en Drive, bajo `MyDrive/Facultad/tesis/`. No se versionan.
- Los zip descargados de Colab quedan ignorados por git. Los resultados que importan se extraen a `runs_logs/` como JSON o Markdown.
- Cada corrida nueva se guarda en `runs_logs/<nombre_de_la_corrida>/`, con el mismo nombre que su carpeta en Drive.
- Los tests apuntan a las rutas de `notebooks/`. Si se mueve un notebook, hay que actualizar su test.
