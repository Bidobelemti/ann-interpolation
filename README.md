# RNA interpolación

## Estructura del proyecto

``` text
├───data                # 27 datos recolectados
├───datainterpolation   # Datos generados mediante interpolación
├───datatrain            # Datos de testeo para el modelo
├───img                 # Imagenes de las muestras, 1000 datos y 300 datos
├───notebook            # Notebooks para pruebas de código
├───src                 # lógica de implementación
└main.py
```

## Objetivo

Diseñar un modelo RNA (redes neuronales artificiales) para estimar valores desconocidos entre datos de muestra conocidos, interpolación.

## Bibliotecas usadas

Se usa PyTorch para la estructura de la red neuronal artificial, no se agrega directamente en el archivo _requirements.txt_ debido a que la instalación es personalizada, https://pytorch.org/get-started/locally/

Ahora para instalar las dependencias se ejecuta el siguiente comando:
``` bash
pip install -r requirements.txt
```

## Funcionamiento