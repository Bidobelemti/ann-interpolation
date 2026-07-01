# RNA Interpolación

Modelo de red neuronal artificial (RNA) que aprende a estimar señales de amplitud
de onda en coordenadas `(x, y, z)` a partir de un conjunto reducido de mediciones
reales, con el fin de **interpolar** valores en posiciones donde no existen datos
medidos.

## Objetivo

A partir de 27 datos recolectados en posiciones conocidas del espacio, se busca
entrenar un modelo que aprenda la relación entre la coordenada `(x, y, z)` y la
señal asociada a esa posición, para poder **predecir la señal en coordenadas
intermedias** no muestreadas. Como referencia y como conjunto de entrenamiento
adicional, primero se generan datos sintéticos mediante **interpolación
trilineal** entre los 27 puntos originales.

## Estructura del proyecto

```text
├───data                # 27 datos recolectados (mediciones reales)
├───datainterpolation   # Datos generados mediante interpolación trilineal
├───datatrain           # Datos usados para entrenar el modelo
├───img                 # Imágenes de las muestras (1000 datos y 300 datos)
├───notebook            # Notebooks para pruebas de código
├───src                 # Lógica de implementación
├───main.py             # Entrenamiento y evaluación del modelo
└───requirements.txt
```

## Flujo de trabajo

1. **`data/`** contiene las 27 mediciones reales, indexadas por coordenada
   `(x, y, z)`.
2. **`interpolation.py`** toma esas 27 mediciones, arma cubos de 8 vértices
   conocidos y genera nuevas señales sintéticas en los puntos intermedios del
   cubo mediante **interpolación trilineal**. El resultado se guarda en
   **`datainterpolation/`**, un archivo `.txt` por punto
   (`DatReceptor_{x}_{y}_{z}.txt`), con columnas `idx` y `signal` separadas por
   tabulador.
3. Los datos usados para entrenar el modelo (reales y/o interpolados, según se
   organice) se colocan en **`datatrain/`**.
4. **`main.py`** carga los datos de `datatrain/`, normaliza las señales,
   entrena una red neuronal (`SignalNet`, definida en `src/model.py`) y
   reporta pérdida (MSE) y precisión con tolerancia en cada época.
5. **`notebook/`** contiene notebooks para experimentar con el pipeline de
   forma interactiva y visualizar los resultados (por ejemplo, comparar la
   señal interpolada contra la predicción del modelo).

## Modelo

`SignalNet` (en `src/model.py`) es un perceptrón multicapa simple:

- Entrada: coordenada `(x, y, z)` → 3 valores.
- Dos capas ocultas (64 y 128 neuronas) con activación `Tanh`.
- Salida: vector de longitud igual al número de muestras de la señal.

## Requisitos

Este proyecto usa **PyTorch**, cuya instalación depende del sistema operativo y
de si se cuenta con GPU (CUDA). Por eso **no** está incluido directamente en
`requirements.txt`; debe instalarse siguiendo la guía oficial:

https://pytorch.org/get-started/locally/

Una vez instalado PyTorch, instala el resto de dependencias con:

```bash
pip install -r requirements.txt
```

## Uso

**1. Generar datos por interpolación** (a partir de `data/`):

```bash
python interpolation.py
```

**2. Entrenar el modelo:**

```bash
python main.py
```

Durante el entrenamiento se imprime, por cada época, la pérdida (MSE) y la
precisión (porcentaje de valores predichos dentro de una tolerancia de
`0.05`) tanto en el conjunto de entrenamiento como en el de validación.

## Resultados

Las imágenes de muestra generadas durante las pruebas (con 1000 y 300 datos)
se encuentran en `img/`.
