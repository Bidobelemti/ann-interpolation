import os
import torch

import pandas as pd
import numpy as np
import torch.nn as nn

from src.load_data import load_data
from src.model import SignalNet
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

PATH = 'datatrain/'
EPOCHS = 50
BATCH_SIZE = 64
TOLERANCIA = 0.05 

def make_tensor_values(data: dict, device: str = 'cpu') -> tuple:
    '''
        Convierte los datos en tensores de PyTorch.
        input:
            data: dict
                Diccionario con coordenadas y señales.
            device: str
                Dispositivo donde se almacenarán los tensores ('cpu' o 'cuda').
        output:
            X: torch.Tensor
                Tensor con las coordenadas.
            Y: torch.Tensor
                Tensor con las señales.
    '''
    # 1. Extraer las llaves (coordenadas) y valores (señales) directamente
    coords = list(data.keys())      # (x,y,z)
    signals = list(data.values())   # array de amplitud de onda

    # 2. Se obtiene lso nparray
    coord_vect = np.array(coords)
    signal_vect = np.array(signals)

    # 3. Normalización de señales
    abs_signals = np.abs(signal_vect)
    global_max = np.max(abs_signals)
    global_min = np.min(abs_signals)

    signal_vect = (signal_vect - global_min) / (global_max - global_min)

    # 4. Convertir a Tensores
    X = torch.tensor(coord_vect, dtype=torch.float32, device=device)
    Y = torch.tensor(signal_vect, dtype=torch.float32, device=device)
    return X, Y

def main():
    
    data = load_data(PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = make_tensor_values(data, device) # X coordenadas (x,y,z) - Y lista de amplitud de onda

    print(X.shape)
    print(Y.shape)

    # 1. Dividimos el dataset en train y test con train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(len(X_train), len(X_test))
    # 1.1. Obtenemos los dataloader de cada uno
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 2. Entrenamiento

    model = SignalNet(250).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    train_loss_history = list() 
    val_loss_history = list()
    train_acc_history = list()
    test_acc_history = list()
    for epoch in range(EPOCHS):
        #----entrenamiento-----
        model.train()
        loss = 0.0
        correct = 0.0
        total_elementos = 0

        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            mse = criterion(outputs, batch_Y)   # Loss
            mse.backward()
            optimizer.step()

            loss += mse.item() * batch_X.size(0)
            # Accuracy con tolerancia
            diff = torch.abs(outputs - batch_Y)
            acciertos = (diff < TOLERANCIA).sum().item()

            correct += acciertos
            total_elementos += batch_Y.numel()
        
        epoch_loss = loss / len(train_loader.dataset)
        train_acc = correct / total_elementos
    
        train_loss_history.append(epoch_loss)
        train_acc_history.append(train_acc)
        # --- FASE DE VALIDACIÓN / TEST ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total_elements_test = 0
        
        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                
                val_loss += loss.item() * batch_X.size(0)
                
                # CÁLCULO DE ACCURACY CON TOLERANCIA
                diferencia = torch.abs(outputs - batch_Y)
                aciertos = (diferencia < TOLERANCIA).sum().item()
                
                correct += aciertos
                total_elements_test += batch_Y.numel()

        epoch_val_loss = val_loss / len(test_loader.dataset)
        test_acc = correct / total_elements_test
        
        val_loss_history.append(epoch_val_loss)
        test_acc_history.append(test_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {epoch_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()