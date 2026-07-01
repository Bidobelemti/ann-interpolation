import torch
import torch.nn as nn

class SignalNet(nn.Module):
    '''
    Red neuronal para predecir la señal de salida a partir de los parámetros de entrada.
    La red tiene 3 capas lineales con funciones de activación Tanh.
    '''
    def __init__(self, signal_len):
        '''
        Inicializa la red neuronal.
        input:
            signal_len: int
                Longitud de la señal de salida.
        '''
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),

            nn.Linear(64, 128),
            nn.Tanh(),

            nn.Linear(128, signal_len)
        )

    def forward(self, x):
        '''
        Propaga la entrada a través de la red neuronal y devuelve la señal de salida.
        input:
            x: torch.Tensor
                Tensor de entrada con las coordenadas (x, y, z).
        output:
            torch.Tensor
                Tensor de salida con la señal predicha.
        '''
        return self.net(x)