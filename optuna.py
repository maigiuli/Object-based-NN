import h5py
import torch 
import optuna
print("imported optuna")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
print("imported torch")
import numpy as np
#from new_scripts_with_root_output.train_MET_prepo_latest_sumpT_Ruben_NORM import ATLASH5HighLevelDataset, ATLASEventNN

#SE SI BLOCCA FARE rm -rf new_scripts_with_root_output/__pycache__


# Debug: Controllo file H5
print("🔍 Caricamento file H5...")
input_file = h5py.File("/storage_tmp/atlas/gmaineri/file_NN_h5/train_Wmunu_METTST_softfJvt_SumpTMiss_pd.h5", 'r')

num_objects = 20
num_inputs = 4 * num_objects + 1
use_passOR = True
nevents = int(input_file["ph_pt"].shape[0])
print(f" File caricato! Numero di eventi: {nevents}")


#-------------definizione classi


def clean_and_pad_array(array, target_length=10):
    new_array = []

    for row in array:  # Itera su ogni evento (riga)
        filtered_row = row[row != 0]  # Rimuove gli zeri
        filtered_row = filtered_row[~np.isnan(filtered_row)]  # Rimuove i NaN

        if len(filtered_row) < target_length:
            # Se ci sono meno di 10 elementi, aggiunge zeri
            padded_row = np.pad(filtered_row, (0, target_length - len(filtered_row)), 'constant', constant_values=0)
        else:
            # Se ci sono più di 10 elementi, prende solo i primi 10
            padded_row = filtered_row[:target_length]

        new_array.append(padded_row)

    return np.array(new_array)

def pad_array(array_list, max_length):
    return np.array([np.pad(arr, (0, max(0, max_length - len(arr))), 'constant', constant_values=0) for arr in array_list])


class ATLASH5HighLevelDataset(torch.utils.data.Dataset):
    def transform(self, data, transform=True):
        if transform:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return data - mean  # evita divisione per zero
            return (data - mean) / std
        else:
            return data
    def transform_padded(self, data, transform=True, pad_value=0.0):
        if not transform:
            return data

        mask = (data != pad_value)
        values = data[mask]

        mean = np.mean(values)
        std = np.std(values)
        std = std if std > 0 else 1.0

        norm = (data - mean) / std
        norm[~mask] = pad_value  # lascia i padding invariati
        return norm

    def __init__(self, file_path, use_passOR, num_objects=num_objects, nevents=nevents, transform=True):
        super(ATLASH5HighLevelDataset, self).__init__()

        h5_file = h5py.File(file_path, 'r')
        self.data=torch.tensor([])
        self.nevents =nevents
        self.num_objects = num_objects  

        #TARGET AND MASK (+tight che non dovrebbe servire)
        self.truth_px = h5_file['TruthpxMiss'][:self.nevents]
        self.truth_py = h5_file['TruthpyMiss'][:self.nevents]
        self.truth_SumpT = h5_file['TruthSumpT'][:self.nevents]
        nonzero_mask =  (np.sqrt(self.truth_px**2+self.truth_py**2)>-1)
        self.truth_px = h5_file['TruthpxMiss'][:self.nevents][nonzero_mask]
        self.truth_py = h5_file['TruthpyMiss'][:self.nevents][nonzero_mask]
        self.truth_SumpT = h5_file['TruthSumpT'][:self.nevents][nonzero_mask]
        self.jet_passOR = h5_file['jet_passOR'][:self.nevents][nonzero_mask]

        self.Tight_TST_pxMiss = self.transform(h5_file['Tight_TST_pxMiss'][:self.nevents][nonzero_mask], transform=transform)
        self.Tight_TST_pyMiss = self.transform(h5_file['Tight_TST_pyMiss'][:self.nevents][nonzero_mask], transform=transform)
        
        if use_passOR:
            # Se la flag `use_passOR` è True, applichiamo la selezione passOR == 1
            self.jet_px = [px[passOR == 1] for px, passOR in zip(
                h5_file['jet_pt'][:self.nevents][nonzero_mask] * np.cos(h5_file['jet_phi'][:self.nevents][nonzero_mask]), 
                self.jet_passOR
            )]

            self.jet_py = [py[passOR == 1] for py, passOR in zip(
                h5_file['jet_pt'][:self.nevents][nonzero_mask] * np.sin(h5_file['jet_phi'][:self.nevents][nonzero_mask]), 
                self.jet_passOR
            )]

            self.jet_Jvt = [jvt[passOR == 1] for jvt, passOR in zip(
                h5_file['jet_Jvt'][:self.nevents][nonzero_mask], 
                self.jet_passOR
            )]


            self.jet_px = pad_array(self.jet_px, num_objects)
 
            self.jet_py = pad_array(self.jet_py, num_objects)
            self.jet_Jvt = pad_array(self.jet_Jvt, num_objects)
        else:
            # Se `use_passOR` è False, usiamo i dati senza filtraggio
            self.jet_px = h5_file['jet_pt'][:self.nevents][nonzero_mask] * np.cos(h5_file['jet_phi'][:self.nevents][nonzero_mask])
            self.jet_py = h5_file['jet_pt'][:self.nevents][nonzero_mask] * np.sin(h5_file['jet_phi'][:self.nevents][nonzero_mask])
            self.jet_Jvt = h5_file['jet_Jvt'][:self.nevents][nonzero_mask]


        self.averageInteractionsPerCrossing = self.transform(h5_file['averageInteractionsPerCrossing'][:self.nevents], transform=transform)
        
        self.el_px = h5_file['el_pt'][:self.nevents][nonzero_mask] * np.cos(h5_file['el_phi'][:self.nevents][nonzero_mask])
        self.muon_px = h5_file['muon_pt'][:self.nevents][nonzero_mask] * np.cos(h5_file['muon_phi'][:self.nevents][nonzero_mask])
        self.tau_px = h5_file['tau_pt'][:self.nevents][nonzero_mask] * np.cos(h5_file['tau_phi'][:self.nevents][nonzero_mask])
        self.ph_px = h5_file['ph_pt'][:self.nevents][nonzero_mask] * np.cos(h5_file['ph_phi'][:self.nevents][nonzero_mask])
        self.soft_px = h5_file['Tight_TST_pxSoft'][:self.nevents][nonzero_mask] 
        self.soft_px = np.expand_dims(self.soft_px, axis=1)
  

        all_objects_px = np.concatenate([ self.ph_px,self.muon_px, self.tau_px, self.el_px, self.soft_px, self.jet_px],axis=1)
        self.all_objects_px = clean_and_pad_array(all_objects_px,self.num_objects)
        self.all_objects_px = self.transform_padded(self.all_objects_px, transform=transform)
       

        jet_labels = [np.where(jet!=0,1,0) for jet in self.jet_px]
        el_labels = [np.where(el!=0, 2,0) for el in self.el_px]
        muon_labels =[np.where(muon!=0, 3,0) for muon in self.muon_px]
        tau_labels = [np.where(tau!=0, 4,0) for tau in self.tau_px]
        ph_labels = [np.where(ph!=0, 5,0) for ph in self.ph_px]
        soft_labels = [np.where(soft!=0, 6,0) for soft in self.soft_px]

        all_labels_px = np.concatenate([ph_labels,muon_labels, tau_labels, el_labels, soft_labels,jet_labels],axis=1)
        self.all_labels_px = clean_and_pad_array(all_labels_px,self.num_objects)


        el_jvt =[np.where(el==0,1,1) for el in self.el_px]
        muon_jvt = [np.where(muon==0,1,1) for muon in self.muon_px]
        tau_jvt = [np.where(tau==0,1,1) for tau in self.tau_px]
        ph_jvt = [np.where(ph==0,1,1) for ph in self.ph_px]
        soft_jvt = [np.where(soft==0,1,1) for soft in self.soft_px]
        jet_jvt = [jet for jet in self.jet_Jvt]
        all_jvt = np.concatenate([ph_jvt,muon_jvt, tau_jvt, el_jvt, soft_jvt, jet_jvt],axis=1)
        self.all_jvt = clean_and_pad_array(all_jvt, self.num_objects)

  
        self.el_py = h5_file['el_pt'][:self.nevents][nonzero_mask] * np.sin(h5_file['el_phi'][:self.nevents][nonzero_mask])
        self.muon_py = h5_file['muon_pt'][:self.nevents][nonzero_mask] * np.sin(h5_file['muon_phi'][:self.nevents][nonzero_mask]) 
        self.tau_py = h5_file['tau_pt'][:self.nevents][nonzero_mask] * np.sin(h5_file['tau_phi'][:self.nevents][nonzero_mask]) 
        self.ph_py = h5_file['ph_pt'][:self.nevents][nonzero_mask] * np.sin(h5_file['ph_phi'][:self.nevents][nonzero_mask]) 
        self.soft_py = h5_file['Tight_TST_pySoft'][:self.nevents][nonzero_mask] 
        self.soft_py = np.expand_dims(self.soft_py, axis=1)
 
        all_objects_py = np.concatenate([self.ph_py,self.muon_py, self.tau_py, self.el_py, self.soft_py, self.jet_py],axis=1)
        self.all_objects_py = clean_and_pad_array(all_objects_py,self.num_objects)
        self.all_objects_py = self.transform_padded(self.all_objects_py, transform=transform)
        
    def __getitem__(self, index):
        
        jet_px_event = self.jet_px[index]  
        el_px_event = self.el_px[index]    
        muon_px_event = self.muon_px[index]   
        tau_px_event = self.tau_px[index]  
        ph_px_event = self.ph_px[index] 
        
        jet_py_event = self.jet_py[index]  
        el_py_event = self.el_py[index]    
        muon_py_event = self.muon_py[index]   
        tau_py_event = self.tau_py[index]  
        ph_py_event = self.ph_py[index] 

        jet_Jvt_event = self.all_jvt[index] 
        averageInteractionsPerCrossing_event = self.averageInteractionsPerCrossing[index]
        averageInteractionsPerCrossing_event = np.array([averageInteractionsPerCrossing_event])

        all_objects_px_event = self.all_objects_px[index] 
        all_objects_py_event = self.all_objects_py[index] 
        all_labels_px_event = self.all_labels_px[index] 

    
        data = torch.tensor(np.concatenate([all_objects_px_event, all_objects_py_event, all_labels_px_event,jet_Jvt_event, averageInteractionsPerCrossing_event], axis=0), dtype=torch.float32)

        target = torch.tensor([self.truth_px[index],self.truth_py[index], self.truth_SumpT[index]], dtype=torch.float32)

        weight =  torch.tensor(1.0, dtype=torch.float32)

        return data, target, weight

    def __len__(self):
        return min(len(self.truth_px), len(self.truth_py))  # Assicura che non ci siano problemi di mismatch

class ATLASEventNN(nn.Module):
    def __init__(self, num_objects, input_dim=num_inputs, first_embed=128, second_embed=512, first_regr=128, second_regr=64, third_regr=32, hidden_dim=128, num_heads=8, num_layers=4, output_dim=3):
        super(ATLASEventNN, self).__init__()
        self.num_objects = num_objects

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, first_embed),
            nn.ReLU(),
            nn.Linear(first_embed, second_embed),
            nn.ReLU(),
            nn.Linear(second_embed, hidden_dim),
            nn.ReLU(),
        )

        self.norm1=torch.nn.LayerNorm(hidden_dim)

        # Encoder Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP finale per la regressione
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, first_regr),#64
            nn.ReLU(),
            nn.Linear(first_regr, second_regr), #64, 64),
            nn.ReLU(),
            nn.Linear(second_regr, third_regr), #64, 64),
            nn.ReLU(),
            nn.Linear(third_regr, 3),#64, 3)
        )

    def forward(self, x):
        x = self.norm1(self.embedding(x))
        x = self.transformer(x)
        x = self.regressor(x)
        return x


 

def objective(trial):
    # Iperparametri da ottimizzare
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    first_embed = trial.suggest_int("first_embed", 64, 256)
    second_embed= trial.suggest_int("second_embed", 64, 256)
    hidden_size = trial.suggest_int("hidden_size", 64, 256)
    first_regr = trial.suggest_int("first_regr", 64, 256)
    second_regr = trial.suggest_int("second_regr", 64, 256)
    third_regr = trial.suggest_int("third_regr", 64, 256)
    num_heads = trial.suggest_int("num_heads", 4, 16)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-0)
    if hidden_size % num_heads != 0:
        raise optuna.TrialPruned()
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.1, 0.5)
    batch_size = trial.suggest_int("batch_size", 64, 512)
    num_epochs = trial.suggest_int("num_epochs", 20, 200)
    
    print(f"🎯 Ottimizzazione con: lr={lr}, first_embed={first_embed}, second_embed={second_embed}, hidden_size={hidden_size}, num_heads={num_heads}, num_layers={num_layers}, dropout={dropout_rate}, batch_size={batch_size}, epochs={num_epochs}")

    # Debug: Caricamento dataset
    print("🔍 Caricamento dataset...")
    dataset = ATLASH5HighLevelDataset("/storage_tmp/atlas/gmaineri/file_NN_h5/train_Wmunu_METTST_softfJvt_SumpTMiss_pd.h5", use_passOR=True, num_objects=num_objects, nevents=nevents)

    print(f" Dataset caricato! Shape dataset: {dataset.all_objects_px.shape}")

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(" DataLoader creato!")

    # Debug: Creazione modello
    print("🔍 Creazione modello Transformer...")
    model = ATLASEventNN(
        num_objects=dataset.num_objects,
        input_dim=num_inputs,
        first_embed = first_embed, 
        second_embed = second_embed, 
        first_regr = first_regr, 
        second_regr= second_regr, 
        third_regr = third_regr, 
        hidden_dim=hidden_size, 
        num_heads=num_heads,
        num_layers=num_layers
    )
    print(" Modello creato!")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Debug: Training
    print("🔍 Inizio training...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_X, batch_Y, _ in data_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f" Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    return loss.item()

# Debug: Lancio Optuna
print("🔍 Inizio ottimizzazione con Optuna...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500)

print(" Ottimizzazione completata!")
print(" Migliori iperparametri trovati:", study.best_params)
