import pandas as pd
import numpy as np
import os


def load_data(path : str) -> dict:
    '''
    Cargamos los datos de una ruta en especifico
    
    ## Parameters:
    path (str): La ruta que contiene los archivos .txt (27 datos recolectados)
    
    ## Returns:
    dict: un diccionario {(x,y,z) : [valor de amplitud de tamaño 1000]}
    '''
    data = dict()
    # iteramos entre archivos
    for fn in os.listdir(path):
        # tenemos presente si un archivo finaliza en .txt
        if fn.endswith('.txt'):
            # eliminamos temrinación de .txt para obtener DataReceptor y coordenadas
            name = fn.replace('.txt', '')
            _,x,y,z = name.split('_')
            # como key son las coordenadas y valor 500 datos de cada .txt, la segunda columna que se considera en si la amplitud de la onda
            key = (float(x), float(y), float(z))
            df = pd.read_csv(os.path.join(path, fn), sep='\t', header=None, names = ['_', 'amplitud'])
            data[key] = df['amplitud'].to_numpy()[:300] # 500 datos debido a que 
    return data