import pandas as pd
import numpy as np
from load_data import load_data

def trilinear_interpolation(u, v, w, values):
    """
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

cube_origins = [
    (-1,-1,-1), (-1,-1,0),
    (-1,0,-1),  (-1,0,0),
    (0,-1,-1),  (0,-1,0),
    (0,0,-1),   (0,0,0)
]
length_signal = len(next(iter(data.values())))

grid = np.linspace(0,1, STEP)

true_coord = list(data.keys()) # obtengo mis coordenadas reales donde no se debe interpolar
signals = list(data.values())

for (x_0, y_0, z_0) in cube_origins:
    x_1, y_1, z_1 = x_0 + 1, y_0 + 1, z_0 + 1
    cube_vertex = [
        (x_0,y_0,z_0), (x_1,y_0,z_0), (x_0,y_1,z_0), (x_1,y_1,z_0),
        (x_0,y_0,z_1), (x_1,y_0,z_1), (x_0,y_1,z_1), (x_1,y_1,z_1)
    ]
    if not all(v in data for v in cube_vertex):
        continue
    v000 = data[(x_0,y_0,z_0)]
    v100 = data[(x_1,y_0,z_0)]
    v010 = data[(x_0,y_1,z_0)]
    v110 = data[(x_1,y_1,z_0)]
    v001 = data[(x_0,y_0,z_1)]
    v101 = data[(x_1,y_0,z_1)]
    v011 = data[(x_0,y_1,z_1)]
    v111 = data[(x_1,y_1,z_1)]
    for x_d in grid:
        for y_d in grid:
            for z_d in grid:
                x = x_0 + x_d
                y = y_0 + y_d
                z = z_0 + z_d
                if (x,y,z) in data:
                    continue
                interpolation_signal = np.zeros(length_signal)
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