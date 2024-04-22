
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pip
import requests
import seaborn as sns
import torch
import torchmetrics
import torchvision
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
#
# from torch.utils.tensorboard import SummaryWriter
from torchmetrics import ConfusionMatrix
from tqdm.auto import tqdm


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Treina um modelo PyTorch para uma única época.

     Transforma um modelo PyTorch de destino no modo de treinamento e, em seguida,
     percorre todas as etapas de treinamento necessárias (avançar
     passar, cálculo de perda, etapa do otimizador).

     Argumentos:
     model: um modelo PyTorch a ser treinado.
     dataloader: Uma instância do DataLoader para o modelo a ser treinado.
     loss_fn: uma função de perda do PyTorch para minimizar.
     otimizador: Um otimizador PyTorch para ajudar a minimizar a função de perda.
     dispositivo: um dispositivo de destino para calcular (por exemplo, "cuda" ou "cpu").

     Retorna:
     Uma tupla de métricas de perda de treinamento e precisão de treinamento.
     No formulário (train_loss, train_accuracy). Por exemplo:

     (0,1112, 0,8743)
     """
    # Put model in train mode
    model.to(device)
    model.train().double()
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (inputs, target) in enumerate(dataloader):
        # movendo os dados para o dispositivo de processamento
        inputs, target = inputs.to(device), target.to(device)
        inputs = inputs.unsqueeze(2)

        # fazendo as previsões
        target_pred = model(inputs.double())

        # calculando a perda
        loss = loss_fn(target_pred, target.long())

        train_loss += loss.data.item()

        # retropropagando os gradientes e atualizando os pesos
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(target_pred, dim=1), dim=1)
        train_acc += (y_pred_class == target).sum().data.item() / \
            len(target_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Testa um modelo PyTorch para uma única época.

     Transforma um modelo PyTorch de destino no modo "eval" e, em seguida, executa
     uma passagem direta em um conjunto de dados de teste.

     Argumentos:
     model: Um modelo PyTorch a ser testado.
     dataloader: Uma instância do DataLoader para o modelo a ser testado.
     loss_fn: uma função de perda do PyTorch para calcular a perda nos dados de teste.
     dispositivo: um dispositivo de destino para calcular (por exemplo, "cuda" ou "cpu").

     Retorna:
     Uma tupla de perda de teste e métricas de precisão de teste.
     No formulário (test_loss, test_accuracy). Por exemplo:

     (0,0223, 0,8985)
     """
    # Coloca o modelo no modo eval
    model.eval().double()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (data, target) in enumerate(dataloader):
            # Send data to target device
            data, target = data.to(device), target.to(device)
            data = data.unsqueeze(2)

            # 1. Forward pass
            test_pred_logits = model(data)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, target.long())
            test_loss += loss.data.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels ==
                         target).sum().data.item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc




def Making_Predictions(data_loader: torch.utils.data.DataLoader,
                       model: torch.nn.Module,
                       device: torch.device):
    """
 Faz previsões usando um modelo treinado no carregador de dados fornecido.

 Argumentos:
     data_loader (torch.utils.data.DataLoader): DataLoader contendo os dados de entrada.
     modelo (torch.nn.Module): modelo treinado para ser usado para fazer previsões.

 Retorna:
     tocha.Tensor: Tensor contendo os rótulos previstos.

 """

    # 1. Make predictions with trained model
    y_preds = []
    model.eval()
    with torch.inference_mode():
        for X, y in (data_loader):
            # Send data and targets to target device
            X, y = X.to(device), y.to(device)

            X = X.unsqueeze(2)
            # Do the forward pass
            y_logit = model(X)
            # Turn predictions from logits -> prediction probabilities -> predictions labels
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
            # Put predictions on CPU for evaluation
            y_preds.append(y_pred.cpu())
        # Concatenate list of predictions into a tensor
        y_pred_tensor = torch.cat(y_preds)

    return y_pred_tensor


def train(model: torch.nn.Module,
          data_loader_train: torch.utils.data.DataLoader,
          data_loader_test: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
#          writer: torch.utils.tensorboard.writer.SummaryWriter,
          columns: int) -> Dict[str, List]:
    """Treina e testa um modelo PyTorch.

     Passa um modelo PyTorch de destino por meio de train_step() e test_step()
     funções para um número de épocas, treinando e testando o modelo
     no mesmo loop de época.

     Calcula, imprime e armazena métricas de avaliação.

     Argumentos:
     model: um modelo PyTorch a ser treinado e testado.
     train_dataloader: Uma instância do DataLoader para o modelo a ser treinado.
     test_dataloader: Uma instância do DataLoader para o modelo a ser testado.
     otimizador: Um otimizador PyTorch para ajudar a minimizar a função de perda.
     loss_fn: uma função de perda do PyTorch para calcular a perda em ambos os conjuntos de dados.
     epochs: Um número inteiro indicando para quantas épocas treinar.
     dispositivo: um dispositivo de destino para calcular (por exemplo, "cuda" ou "cpu").

     Retorna:
     Um dicionário de perda de treinamento e teste, bem como treinamento e
     testar métricas de precisão. Cada métrica tem um valor em uma lista para
     cada época.
     Na forma: {train_loss: [...],
               train_acc: [...],
               teste_perda: [...],
               test_acc: [...]}
     Por exemplo, se o treinamento for epochs=2:
              {train_loss: [2.0616, 1.0537],
               train_acc: [0,3945, 0,3945],
               perda_teste: [1.2641, 1.5706],
               test_acc: [0,3400, 0,2973]}
     """
    # Cria um dicionário de resultados vazio
    results = {
        "test_loss": [],
        "test_acc": []
    }

    # Make sure model on target device
    model.to(device).double()

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=data_loader_train,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=data_loader_test,
                                        loss_fn=loss_fn,
                                        device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)


    # Return the filled results at the end of the epochs
    return results


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time