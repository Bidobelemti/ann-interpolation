import pandas as pd
import numpy as np
from load_data import load_data

def trilinear_interpolation(u, v, w, values):
    """
    Obtiene el valor interpolado en un punto (u, v, w) dentro de un cubo definido por 8 valores en sus esquinas.
    input:
        u: float
            Coordenada normalizada en el eje x (0 <= u <= 1).
        v: float
            Coordenada normalizada en el eje y (0 <= v <= 1).
        w: float
            Coordenada normalizada en el eje z (0 <= w <= 1).
        values: list of float
            Lista de 8 valores correspondientes a las esquinas del cubo.
    output:
        float
            Valor interpolado en el punto (u, v, w).
    values:
    [V000, V100, V010, V110,
     V001, V101, V011, V111]
    """

    V000, V100, V010, V110, \
    V001, V101, V011, V111 = values

    return (
        V000*(1-u)*(1-v)*(1-w)
        + V100*u*(1-v)*(1-w)
        + V010*(1-u)*v*(1-w)
        + V110*u*v*(1-w)
        + V001*(1-u)*(1-v)*w
        + V101*u*(1-v)*w
        + V011*(1-u)*v*w
        + V111*u*v*w
    )

STEP = 10

data = load_data('data')

# Coordenadas de los vértices del cubo unitario en el espacio 3D
cube_origins = [
    (-1,-1,-1), (-1,-1,0),
    (-1,0,-1),  (-1,0,0),
    (0,-1,-1),  (0,-1,0),
    (0,0,-1),   (0,0,0)
]
length_signal = len(next(iter(data.values())))

# Creamos una grilla de puntos dentro del cubo unitario para interpolar
grid = np.linspace(0,1, STEP)

true_coord = list(data.keys()) # obtengo mis coordenadas reales donde no se debe interpolar
signals = list(data.values())

# Creamos un diccionario para almacenar los datos interpolados
for (x_0, y_0, z_0) in cube_origins:
    x_1, y_1, z_1 = x_0 + 1, y_0 + 1, z_0 + 1
    # Verificamos si los 8 vértices del cubo están presentes en los datos
    cube_vertex = [
        (x_0,y_0,z_0), (x_1,y_0,z_0), (x_0,y_1,z_0), (x_1,y_1,z_0),
        (x_0,y_0,z_1), (x_1,y_0,z_1), (x_0,y_1,z_1), (x_1,y_1,z_1)
    ]
    # Si alguno de los vértices no está presente, continuamos con el siguiente cubo
    if not all(v in data for v in cube_vertex):
        continue
    # Obtenemos los valores de las señales en los vértices del cubo
    v000 = data[(x_0,y_0,z_0)]
    v100 = data[(x_1,y_0,z_0)]
    v010 = data[(x_0,y_1,z_0)]
    v110 = data[(x_1,y_1,z_0)]
    v001 = data[(x_0,y_0,z_1)]
    v101 = data[(x_1,y_0,z_1)]
    v011 = data[(x_0,y_1,z_1)]
    v111 = data[(x_1,y_1,z_1)]
    # Creamos un diccionario para almacenar los datos interpolados
    for x_d in grid:
        for y_d in grid:
            for z_d in grid:
                # Calculamos las coordenadas reales del punto a interpolar
                x = x_0 + x_d
                y = y_0 + y_d
                z = z_0 + z_d
                # Si el punto ya existe en los datos, no lo interpolamos
                if (x,y,z) in data:
                    continue
                # Interpolamos la señal en el punto (x_d, y_d, z_d) usando la interpolación trilineal
                interpolation_signal = np.zeros(length_signal)
                # Iteramos sobre cada punto de la señal y aplicamos la interpolación trilineal
                for i in range(length_signal):
                    interpolation_signal[i] = trilinear_interpolation(
                        x_d, y_d, z_d,
                        [v000[i], v100[i], v010[i], v110[i],
                        v001[i], v101[i], v011[i], v111[i]]
                    )
                
                df = pd.DataFrame(
                    {
                        'idx' : np.arange(length_signal),
                        'signal' : interpolation_signal
                    }
                )
                fn = f'DatReceptor_{round(x,3)}_{round(y,3)}_{round(z,3)}.txt'
                df.to_csv('datainterpolation/'+fn,
                    sep='\t',
                    header=False,
                    index=False)

print('Interpolación completada')