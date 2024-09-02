from utils import *
from func import *
import logging
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm

def train_model(input_positions, delta_input, delta_output, model_class, epochs=15, batch_size=8, learning_rate=0.03):
    n_samples = input_positions.shape[0]
    x_size = input_positions.shape[1]

    # Prepare training tensors with the correct shape for LSTM/GRU (3D: batch_size, sequence_length, input_size)
    if model_class in [LSTMModel, GRUModel]:
        X_train = torch.zeros((n_samples, x_size, 2)).to(device)  # 3D input for LSTM/GRU
    else:
        X_train = torch.zeros((n_samples, 8, x_size, 2)).to(device)  # 4D input for CNN
    y_train = torch.zeros((n_samples, 2)).to(device)

    for sample_idx in range(n_samples):
        input_pos = input_positions[sample_idx]
        if model_class in [LSTMModel, GRUModel]:
            X_train[sample_idx, :, :] = torch.tensor(input_pos, dtype=torch.float32)  # No channel expansion for LSTM/GRU
        else:
            X_train[sample_idx, :, :, :] = torch.tensor(input_pos, dtype=torch.float32).expand(8, x_size, 2)  # Expand to match in_channels for CNN
        
        y_train[sample_idx, 0] = input_pos[-1, 0] + delta_output
        y_train[sample_idx, 1] = input_pos[-1, 1] + delta_output

    X_train = normalize_data(X_train)
    
    # Split data into training and validation sets
    train_ratio = 0.8
    n_train = int(len(X_train) * train_ratio)
    n_val = len(X_train) - n_train

    # Training and validation datasets
    train_data = X_train[:n_train]
    val_data = X_train[n_train:]
    train_labels = y_train[:n_train]
    val_labels = y_train[n_train:]

    # PyTorch datasets
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)

    # PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model based on the provided class
    if model_class in [LSTMModel, GRUModel]:
        model = model_class(input_size=2, hidden_size=64, num_layers=2, output_size=2).to(device)
    else:
        model = model_class(x_size).to(device)  # For CNNModel

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    criterion = nn.SmoothL1Loss()

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels_batch in train_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = validate(model, validation_loader, criterion)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return model


# Configuração de logging
logging.basicConfig(level=logging.INFO)

core_path = r'C:\Python\BeamPrediction_DeepLearning\BeamPrediction_DeepLearning\data'

scen_idx = 36
training_path ='deepsense_challenge2023_trainset.csv'
testing_path = 'deepsense_challenge2023_testset_example'
csv_train = os.path.join(core_path, training_path)
csv_dict_path = os.path.join(core_path, 'scenario36.p')

X_SIZE = 5
N_GPS = 2
N_GPS_COORD = 2
N_ARR = 4
N_BEAMS = 64
IDX_COL1 = 'unique_index'
IDX_COL2 = 'abs_index'

# Carregar dados
with open(csv_dict_path, 'rb') as fp:
    csv_dict = pickle.load(fp)

df_train = pd.read_csv(csv_train)

if True:  # TREINAMENTO
    # Preparar os dados
    samples_of_scen = np.where(df_train['scenario'] == scen_idx)[0]
    n_samples = len(samples_of_scen)
    train_positions = np.zeros((n_samples, X_SIZE, N_GPS, N_GPS_COORD))
    y_pos1 = np.zeros((n_samples, N_GPS_COORD))
    y_pos2 = np.zeros((n_samples, N_GPS_COORD))
    y_pwrs = np.zeros((n_samples, N_ARR, N_BEAMS))
    logging.info('Carregando Dataset de Treinamento')

    for sample_idx in tqdm(range(n_samples), desc='Loading data'):
        train_sample = samples_of_scen[sample_idx]
        for x_idx in range(X_SIZE):
            abs_idx_relative_index = (csv_dict[IDX_COL2] == df_train[f'x{x_idx+1}_'+IDX_COL1][train_sample])
            train_positions[sample_idx, x_idx, 0, :] = csv_dict['unit1_gps1'][abs_idx_relative_index]
            train_positions[sample_idx, x_idx, 1, :] = csv_dict['unit2_gps1'][abs_idx_relative_index]

        y_idx = (csv_dict[IDX_COL2] == df_train['y1_'+IDX_COL1][train_sample])
        y_pos1[sample_idx] = csv_dict['unit1_gps1'][y_idx]
        y_pos2[sample_idx] = csv_dict['unit2_gps1'][y_idx]

        for arr_idx in range(N_ARR):
            y_pwrs[sample_idx, arr_idx] = csv_dict[f'unit1_pwr{arr_idx+1}'][y_idx]

    # Preparar os dados de treinamento
    y_true_beams = df_train['y1_unit1_overall-beam'].values[samples_of_scen]
    y_pwrs_reshaped = y_pwrs.reshape((n_samples, -1))
    all_true_beams = np.flip(np.argsort(y_pwrs_reshaped, axis=1), axis=1)
    true_best_pwr = y_pwrs_reshaped[np.arange(n_samples), all_true_beams[:, 0]]
    train_unit_01 = train_positions[:, :, 0]
    train_unit_02 = train_positions[:, :, 1]

    # Ajuste do reshape
    input_positions_unit_01 = train_unit_01.reshape(-1, X_SIZE, N_GPS_COORD)
    input_positions_unit_02 = train_unit_02.reshape(-1, X_SIZE, N_GPS_COORD)

    logging.info(f'Shape train unit 01: {input_positions_unit_01.shape}')
    logging.info(f'Shape train unit 02: {input_positions_unit_02.shape}')

    # Treinamento dos modelos usando LSTM
    model_unit_01 = train_model(input_positions_unit_01, delta_input=0.2, delta_output=0.5, model_class=GRUModel, epochs=20, batch_size=8, learning_rate=0.0001)
    model_unit_02 = train_model(input_positions_unit_02, delta_input=0.2, delta_output=0.5, model_class=GRUModel, epochs=20, batch_size=8, learning_rate=0.0001)

    # Definir o caminho do arquivo CSV de teste
    csv_test = csv_train  
    df_test = pd.read_csv(csv_test)[0:3478]

    # Definir o número de amostras no conjunto de teste
    n_samples_test = df_test.shape[0]
    folder = '/'.join(csv_test.split('/')[:-1])
    test_positions = np.zeros((n_samples_test, X_SIZE, N_GPS, N_GPS_COORD))

    # Carregar dados de teste
    for sample_idx in tqdm(range(n_samples_test), desc='Loading test data'):
        for x_idx in range(X_SIZE):
            gps_file_path = os.path.join('C:/Python/BeamPrediction_DeepLearning/BeamPrediction_DeepLearning/data/', folder, df_test[f'x{x_idx+1}_unit1_gps1'][sample_idx])
            try:
                test_positions[sample_idx, x_idx, 0, :] = np.loadtxt(gps_file_path)
            except FileNotFoundError:
                print(f"File not found: {gps_file_path}")

            gps_file_path = os.path.join('C:/Python/BeamPrediction_DeepLearning/BeamPrediction_DeepLearning/data/', folder, df_test[f'x{x_idx+1}_unit2_gps1'][sample_idx])
            try:
                test_positions[sample_idx, x_idx, 1, :] = np.loadtxt(gps_file_path)
            except FileNotFoundError:
                print(f"File not found: {gps_file_path}")

    samples_of_scen_test = np.where(df_test['scenario'] == scen_idx)[0]
    n_samples_test = len(samples_of_scen_test)

    # Inicializar arrays para as posições e potências do teste
    y_pos1_test = np.zeros((n_samples_test, N_GPS_COORD))
    y_pos2_test = np.zeros((n_samples_test, N_GPS_COORD))
    y_pwrs_test = np.zeros((n_samples_test, N_ARR, N_BEAMS))

    # Carregar as posições de teste
    for sample_idx in tqdm(range(n_samples_test), desc='Loading test power data'):
        test_sample = samples_of_scen_test[sample_idx]
        y_idx = (csv_dict[IDX_COL2] == df_test.loc[test_sample, 'y1_'+IDX_COL1])
        y_pos1_test[sample_idx] = csv_dict['unit1_gps1'][y_idx].squeeze()
        y_pos2_test[sample_idx] = csv_dict['unit2_gps1'][y_idx].squeeze()
        for arr_idx in range(N_ARR):
            y_pwrs_test[sample_idx, arr_idx] = csv_dict[f'unit1_pwr{arr_idx+1}'][y_idx].squeeze()

    y_true_beams_test = df_test['y1_unit1_overall-beam'].values[samples_of_scen_test]
    y_pwrs_reshaped_test = y_pwrs_test.reshape((n_samples_test, -1))
    all_true_beams_test = np.flip(np.argsort(y_pwrs_reshaped_test, axis=1), axis=1)
    test_positions_unit_01 = test_positions[:, :, 0]
    test_positions_unit_02 = test_positions[:, :, 1]

    input_positions_unit_01 = test_positions_unit_01.reshape(-1, X_SIZE, N_GPS_COORD)
    input_positions_unit_02 = test_positions_unit_02.reshape(-1, X_SIZE, N_GPS_COORD)

    logging.info(f'Shape test_positions unit 01: {input_positions_unit_01.shape}')
    logging.info(f'Shape test_positions unit 02: {input_positions_unit_02.shape}')

    # Realizar predições usando o modelo LSTM
    gps1_est_pos = Making_Predictions(input_positions_unit_01, 0.2, 0.5, model_unit_01, model_type='lstm')
    gps2_est_pos = Making_Predictions(input_positions_unit_02, 0.2, 0.5, model_unit_02, model_type='lstm')
    
    # (Restante do código para avaliação, predição e cálculo de métricas permanece inalterado)
# Verificar se a estimativa AOA está correlacionada o suficiente para uma previsão precisa
    lat_deltas = gps1_est_pos[:, 0] - test_positions[:, -1, 0, 0]
    lon_deltas = gps1_est_pos[:, 1] - test_positions[:, -1, 0, 1]
    heading = compute_ori_from_pos_delta(lat_deltas, lon_deltas)
    lat_deltas = gps1_est_pos[:, 0] - gps2_est_pos[:, 0]
    lon_deltas = gps1_est_pos[:, 1] - gps2_est_pos[:, 1]
    ori_rel = compute_ori_from_pos_delta(lat_deltas, lon_deltas)
    aoa_estimation = norm_2pi(-1 * (ori_rel - heading - np.pi / 4))

    # Estimar feixes
    beam_pred_all = predict_beam_uniformly_from_aoa(aoa_estimation)
    
    # Verificar tamanho das predições
    if beam_pred_all.shape[0] != y_true_beams_test.shape[0]:
        raise Exception(f"Mismatch in number of predicted beams ({beam_pred_all.shape[0]}) and true labels ({y_true_beams_test.shape[0]}).")

    best_beam_pred = np.copy(beam_pred_all[:, 0])

    # Ajustar previsões de feixe
    pred_diff = np.array([circular_distance(a, b, sign=True) for a, b in zip(best_beam_pred, y_true_beams_test)])
    if np.isnan(pred_diff).any():
        print("Existem valores NaN em pred_diff. Tentando remover...")

    filtered_pred_diff = pred_diff[~np.isnan(pred_diff) & (abs(pred_diff) < 5)]
    shift = -round(np.mean(filtered_pred_diff)) if len(filtered_pred_diff) > 0 else 0
    print(f'estimated_shift = {shift}')

    beam_pred_all += shift
    best_beam_pred += shift

    beam_pred_all[beam_pred_all > 255] -= 255
    beam_pred_all[beam_pred_all < 0] += 255
    best_beam_pred[best_beam_pred > 255] -= 255
    best_beam_pred[best_beam_pred < 0] += 255

    # Calcular pontuações
    pred_diff_abs = np.array([circular_distance(a, b) for a, b in zip(best_beam_pred, all_true_beams_test[:,0])])
    average_beam_index_diff = np.mean(pred_diff_abs)
    print(f'Average Beam Index Distance = {average_beam_index_diff:.2f}')

    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(np.arange(len(pred_diff_abs)), pred_diff_abs, s=2)
    plt.hlines(y=average_beam_index_diff, xmin=0, xmax=len(pred_diff_abs), color='r', alpha=.5)

    # Calcular top-k accuracy
    top_k = compute_acc(all_true_beams_test, best_beam_pred, top_k=[1, 3, 5])
    print(f'Top-k = {top_k}')

    # Calcular APL (Average Power Loss)
    est_best_pwr = y_pwrs_reshaped_test[np.arange(n_samples_test), best_beam_pred]
    true_best_pwr = y_pwrs_reshaped_test[np.arange(n_samples_test), all_true_beams_test[:, 0]]
    apl = APL(true_best_pwr, est_best_pwr)
    print(f'Average Power Loss = {apl:.2f} dB')
