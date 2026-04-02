import re
import matplotlib.pyplot as plt
import numpy as np  # Importamos numpy para la interpolación

def extraer_datos_entrenamiento(ruta_archivo):
    datos_por_epoca = {}
    patron = re.compile(r"Epoch (\d+), Step \d+: loss = ([\d.]+), psnr = ([\d.]+) dB, bal = ([\d.]+)")
    
    with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
        for linea in archivo:
            coincidencia = patron.search(linea)
            if coincidencia:
                epoca = int(coincidencia.group(1))
                loss = float(coincidencia.group(2))
                psnr = float(coincidencia.group(3))
                
                datos_por_epoca[epoca] = {
                    'loss': loss,
                    'psnr': psnr
                }
    return datos_por_epoca

# 1. Extraer los datos del archivo
archivo_log = 'training_outs_2.txt'
datos_extraidos = extraer_datos_entrenamiento(archivo_log)

# 2. Separar los datos originales (incompletos)
epocas_reales = sorted(datos_extraidos.keys())
losses_reales = [datos_extraidos[e]['loss'] for e in epocas_reales]
psnrs_reales = [datos_extraidos[e]['psnr'] for e in epocas_reales]

# 3. INTERPOLACIÓN DE DATOS FALTANTES
# Generamos un array continuo desde la época mínima hasta la máxima
epocas_completas = np.arange(min(epocas_reales), max(epocas_reales) + 1)

# np.interp realiza una interpolación lineal de los valores
losses_interpolados = np.interp(epocas_completas, epocas_reales, losses_reales)
psnrs_interpolados = np.interp(epocas_completas, epocas_reales, psnrs_reales)

# 4. Graficar los resultados
plt.figure(figsize=(12, 5))

# Gráfico de Loss
plt.subplot(1, 2, 1)
# Dibujamos la línea continua con todos los datos (reales + interpolados)
plt.plot(epocas_completas, losses_interpolados, color='crimson', linestyle='-', alpha=0.6, label='Loss (Interpolado)')
# Superponemos los puntos reales
plt.scatter(epocas_reales, losses_reales, color='darkred', zorder=5, label='Datos Reales')
plt.title('Evolución del Loss (con Interpolación)')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Gráfico de PSNR
plt.subplot(1, 2, 2)
# Dibujamos la línea continua
plt.plot(epocas_completas, psnrs_interpolados, color='dodgerblue', linestyle='-', alpha=0.6, label='PSNR (Interpolado)')
# Superponemos los puntos reales
plt.scatter(epocas_reales, psnrs_reales, color='navy', zorder=5, label='Datos Reales')
plt.title('Evolución del PSNR (con Interpolación)')
plt.xlabel('Época')
plt.ylabel('PSNR (dB)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()