#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import matplotlib.pyplot as plt
import h5py
import pandas as pd
import uproot
#import ROOT
import builtins
import time
import random
import numpy as np 
import argparse

import torch
from torch import nn
import torch.nn.init as init
from torch.utils.data import DataLoader, ConcatDataset, Dataset


torch.use_deterministic_algorithms(True)
print("GPU disponibile:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nome GPU:", torch.cuda.get_device_name(0))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_seed(42)


# In[2]:


# In[3]:


 #sample= "Wmunu_new_20_3lay_bestsofar_sumpT_noMETabs_transformer_Ruben_NORM_multi32targets8020_limited_many_full300_ttbar_nonallhad+Wmunu"
 #train_files = [
   #  "/eos/user/g/gimainer/SWAN_projects/prova.py/train_ttbar_nonallhad_METTST_softfJvt_SumpTMiss_pd_nobjects_80_many.h5" ,
    # "/eos/user/g/gimainer/SWAN_projects/prova.py/train_Wmunu_METTST_softfJvt_SumpTMiss_pd_nobjects_80_many.h5"  
   # "/eos/user/g/gimainer/SWAN_projects/prova.py/train_ttbar_allhad_METTST_softfJvt_SumpTMiss_pd_nobjects_80_many.h5", 
  #    ]
 #test_files = [
    
  #   "/eos/user/g/gimainer/SWAN_projects/prova.py/test_ttbar_nonallhad_METTST_softfJvt_SumpTMiss_pd_nobjects_20_many.h5", 
  # "/eos/user/g/gimainer/SWAN_projects/prova.py/test_Wmunu_METTST_softfJvt_SumpTMiss_pd_nobjects_20_many.h5" 
  #  "/eos/user/g/gimainer/SWAN_projects/prova.py/test_ttbar_allhad_METTST_softfJvt_SumpTMiss_pd_nobjects_20_many.h5", 
  #   ]


# In[4]:
parser = argparse.ArgumentParser(description="Train/test file handler")
parser.add_argument("--sample", type=str, required=True, help="Sample name")
parser.add_argument("--train_files", nargs='+', required=True, help="List of training files")
parser.add_argument("--test_files", nargs='+', required=True, help="List of test files")
parser.add_argument("--beta", type=float, required=True, help='Valore del parametro beta della loss function SmoothL1Loss')
args = parser.parse_args()


print("Sample:", args.sample)
print("Train files:", args.train_files)
print("Test files:", args.test_files)
print("Valore di beta: ", args.beta)

train_files = args.train_files
test_files = args.test_files 
sample = args.sample
beta = args.beta

def calculate_lin(pred, truth, bin_edges):
    y_values = []
    for i in range(len(bin_edges)-1):
        bin_mask = (truth >= bin_edges[i]) & (truth < bin_edges[i+1])
        
        if np.sum(bin_mask) > 0:
            y = (pred[bin_mask] - truth[bin_mask])/truth[bin_mask]
            y_values.append(np.nanmean(y))
        else:
            y_values.append(np.nan) # Assegna NaN se il bin è vuoto
   
    return np.array(y_values)

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


# In[11]:


def pad_array(array_list, max_length):
    return np.array([np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=0) for arr in array_list])



# In[6]:

nevents_train = 0
nevents_test = 0
for train_file in train_files:
    h5_file = h5py.File(train_file, 'r')
    nevents_train += int(h5_file["ph_pt"].shape[0])
for test_file in test_files:
    print(test_file)
    h5_file = h5py.File(test_file, 'r')
    nevents_test += int(h5_file["ph_pt"].shape[0])
h5_file = h5py.File(train_file, 'r')
num_objects=3
num_jets=20
num_inputs = 4*(num_jets+num_objects)+1
use_passOR=False
nevents_train = [int(2818564), int(3338732), int(2542241), int(2831215), int(1418376), int(161666), int(2744596) ] #ttbar non all hadr, ttbar dilepton, ttbar single lepton, Wmunu, WZ, Zmm, ZGamma, NO Zee, Ztautau int(894608), int(538096)
nevents_test = [int(704641), int(834683), int(635561), int(724962), int(354595), int(40417), int(686149)]#int(704641) #nevents_test #300000 NO Zee Ztautau , int(223653), int(134524)
print(nevents_train)
print(nevents_test)
epochs = 200
model_file = f"/eos/user/g/gimainer/model_{sample}.pth"


# In[7]:


class ATLASH5HighLevelDataset(torch.utils.data.Dataset):


    def __init__(self, file_path, nevents, use_passOR=use_passOR, num_objects=num_objects, num_jets = num_jets, transform=False):
        super().__init__()
        h5_file = h5py.File(file_path, 'r')
        self.nevents = nevents
        self.num_objects = num_objects  
        self.transform_flag = transform

        # --- Truth (target)
        self.truth_px = h5_file['TruthpxMiss'][:nevents]
        self.truth_py = h5_file['TruthpyMiss'][:nevents]
        self.truth_SumpT = h5_file['TruthSumpT'][:nevents]
        mask = (np.sqrt(self.truth_px**2 + self.truth_py**2) > -1)

        # --- Maschera
        self.truth_px = self.truth_px[mask]
        self.truth_py = self.truth_py[mask]
        self.truth_SumpT = self.truth_SumpT[mask]

        # --- Interazioni ambientali
        self.averageInteractionsPerCrossing = self.transform(h5_file['averageInteractionsPerCrossing'][:nevents][mask], transform)
        self.NPV = self.transform(h5_file['NPV'][:nevents][mask], transform)
        #self.nel = self.transform(h5_file['nel'][:nevents][mask], transform)
        #self.nmuon = self.transform(h5_file['nmuon'][:nevents][mask], transform)
        #self.ntau = self.transform(h5_file['ntau'][:nevents][mask], transform)
        #self.nph = self.transform(h5_file['nph'][:nevents][mask], transform)
        #self.njet = self.transform(h5_file['njet'][:nevents][mask], transform)

        # --- Particelle
        self.objects = {}

        def get_px_py(pt, phi):
            return pt * np.cos(phi), pt * np.sin(phi)

        # Jets con passOR
        jet_passOR = h5_file['jet_passOR'][:nevents][mask]
        jet_pt = h5_file['jet_pt'][:nevents][mask]
        jet_phi = h5_file['jet_phi'][:nevents][mask]
        jet_Jvt = h5_file['jet_Jvt'][:nevents][mask]
        #jet_eta = h5_file['jet_eta'][:nevents][mask]

        if use_passOR:
            self.objects["jet_px"] = [px[po == 1] for px, po in zip(jet_pt * np.cos(jet_phi), jet_passOR)]
            self.objects["jet_py"] = [py[po == 1] for py, po in zip(jet_pt * np.sin(jet_phi), jet_passOR)]
            self.objects["jet_Jvt"] = [jvt[po == 1] for jvt, po in zip(jet_Jvt, jet_passOR)]
            
        else:
            self.objects["jet_px"], self.objects["jet_py"] = get_px_py(jet_pt, jet_phi)
            self.objects["jet_Jvt"] = jet_Jvt
            #self.objects["jet_eta"] = jet_eta

        # Elettroni
        el_pt = h5_file['el_pt'][:nevents][mask]
        el_phi = h5_file['el_phi'][:nevents][mask]
        self.objects["el_px"], self.objects["el_py"] = get_px_py(el_pt, el_phi)
        #self.objects["el_eta"] = h5_file['el_eta'][:nevents][mask]

        # Muoni
        mu_pt = h5_file['muon_pt'][:nevents][mask]
        mu_phi = h5_file['muon_phi'][:nevents][mask]
        self.objects["muon_px"], self.objects["muon_py"] = get_px_py(mu_pt, mu_phi)
        #self.objects["muon_eta"] = h5_file['muon_eta'][:nevents][mask]

        # Tau
        tau_pt = h5_file['tau_pt'][:nevents][mask]
        tau_phi = h5_file['tau_phi'][:nevents][mask]
        self.objects["tau_px"], self.objects["tau_py"] = get_px_py(tau_pt, tau_phi)
        #self.objects["tau_eta"] = h5_file['tau_eta'][:nevents][mask]

        # Fotoni
        ph_pt = h5_file['ph_pt'][:nevents][mask]
        ph_phi = h5_file['ph_phi'][:nevents][mask]
        self.objects["ph_px"], self.objects["ph_py"] = get_px_py(ph_pt, ph_phi)
        #self.objects["ph_eta"] = h5_file['ph_eta'][:nevents][mask]


        # TIght MET 
        if "ttbar1l" not in file_path and "Zee" not in file_path and "Ztautau" not in file_path:
            self.tight_px = h5_file['Tight_TST_pxMiss'][:nevents][mask]
            self.tight_py = h5_file['Tight_TST_pyMiss'][:nevents][mask]
        else: 
            self.tight_px = h5_file['met_pxMiss_Tight_TST'][:nevents][mask]
            self.tight_py = h5_file['met_pyMiss_Tight_TST'][:nevents][mask]

        # Soft term
        if "ttbar1l" not in file_path and "Zee" not in file_path and "Ztautau" not in file_path:
            self.objects["soft_px"] = h5_file['Tight_TST_pxSoft'][:nevents][mask]
            self.objects["soft_py"] = h5_file['Tight_TST_pySoft'][:nevents][mask]
            self.objects["soft_px_CST"] = h5_file['Tight_CST_pxSoft'][:nevents][mask]
            self.objects["soft_py_CST"] = h5_file['Tight_CST_pySoft'][:nevents][mask]
        else: 
            self.objects["soft_px"] = h5_file['met_pxSoft_Tight_TST'][:nevents][mask]
            self.objects["soft_py"] = h5_file['met_pySoft_Tight_TST'][:nevents][mask]
            self.objects["soft_px_CST"] = h5_file['met_pxSoft_Tight_CST'][:nevents][mask]
            self.objects["soft_py_CST"] = h5_file['met_pySoft_Tight_CST'][:nevents][mask]

        #self.objects["jet_passOR"] = jet_passOR

        self.met = np.sqrt(self.truth_px**2 + self.truth_py**2)
        self.event_weights = np.ones_like(self.met)
        self.event_weights[(self.met > 100) & (self.met <= 300)] = 2.0
        self.event_weights[(self.met > 300) & (self.met <= 600)] = 3.0
        self.event_weights[self.met > 600] = 4.0       

    def transform(self, data, apply):
        if not apply:
            return data
        mean, std = np.mean(data), np.std(data)
        std = std if std > 0 else 1.0
        return (data - mean) / std

    def __len__(self):
        return len(self.truth_px)

    def __getitem__(self, index):
        data = {}

        for key in self.objects:
            val = self.objects[key][index]

            # --- TAGLIA I TENSORI ---
            if isinstance(val, np.ndarray):
                if "jet" in key:
                    val = val[:num_jets] 
                elif "px" in key or "py" in key : 
                    val = val[:num_objects]
                val = torch.tensor(val, dtype=torch.float32)
            else:
                val = torch.tensor(np.array([val]), dtype=torch.float32)

            data[key] = val

        # target e peso
        target = torch.tensor([self.truth_px[index], self.truth_py[index]], dtype=torch.float32)
        weight = torch.tensor(1.0, dtype=torch.float32)

        # variabile ambientale
        for var in ['averageInteractionsPerCrossing', 'NPV']: #,'nel', 'nmuon', 'ntau', 'nph', 'njet'
            data[var] = torch.tensor(
                np.array([getattr(self, var)[index]], dtype=np.float32), dtype=torch.float32
            )


        return data, target, weight
    


# In[8]:


# In[9]:


from torch.nn.utils.rnn import pad_sequence

def atlas_collate_fn(batch):
    all_keys = batch[0][0].keys()
    batch_data = {k: [] for k in all_keys}
    batch_targets = []
    batch_weights = []

    for data, target, weight in batch:
        for k in all_keys:
            batch_data[k].append(data[k])
        batch_targets.append(target)
        batch_weights.append(weight)

    for k in batch_data:
        shapes = [x.shape for x in batch_data[k]]
        # Se i tensori hanno tutti la stessa shape (tipicamente scalari o vettori fissi), faccio stack
        if all(s == shapes[0] for s in shapes):
            batch_data[k] = torch.stack(batch_data[k])
        # Altrimenti, assumo che siano sequenze variabili da paddare
        elif all(len(s) == 1 for s in shapes):  # es. torch.Size([4]), torch.Size([1]), etc.
            batch_data[k] = pad_sequence(batch_data[k], batch_first=True)
        else:
            raise ValueError(f"Forma inattesa nella chiave '{k}': {shapes}")

    batch_targets = torch.stack(batch_targets)
    batch_weights = torch.stack(batch_weights)

    return batch_data, batch_targets, batch_weights


# In[10]:


def get_input_dims_from_batch(dataloader):
    for batch in dataloader:
        X, _, _ = batch  # batch_data, target, weight
        return {k: v.shape[1] if v.ndim > 1 else 1 for k, v in X.items()}


# In[11]:




def get_ATLAS_inputs(train_files, test_files, batch_size, nevents_train=nevents_train, nevents_test=nevents_test):
    # Ensure inputs are lists (in caso vengano passati singoli file per errore)
    if isinstance(train_files, str):
        train_files = [train_files]
    if isinstance(test_files, str):
        test_files = [test_files]
    
    # Create datasets
    training_DSlist = [ATLASH5HighLevelDataset(f, nevents=nevents_train) for f, nevents_train in zip(train_files, nevents_train)]
    testing_DSlist = [ATLASH5HighLevelDataset(f, nevents=nevents_test) for f, nevents_test in zip(test_files, nevents_test)]
    # Concatenate datasets
    training_data = ConcatDataset(training_DSlist)
    testing_data = ConcatDataset(testing_DSlist)
    # Create data loaders
    g = torch.Generator()
    g.manual_seed(42)

    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        collate_fn=atlas_collate_fn,
        num_workers=2
    )

    test_dataloader = DataLoader(
        testing_data,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        collate_fn=atlas_collate_fn,
        num_workers=2
    )

    # Determine input dimensions
    input_dims = get_input_dims_from_batch(train_dataloader)
    print("Input dims:", input_dims)

    return train_dataloader, test_dataloader, input_dims


# In[12]:


class ATLASEventNN_MultiInput(nn.Module):
    def __init__(self, input_dims, hidden_dim=128, num_heads=4, num_layers=2, output_dim=2):
        super().__init__()

        self.embeddings = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, hidden_dim),
                nn.ReLU(),
            )
            for name, input_dim in input_dims.items()
        })

        self.norms = nn.ModuleDict({
            name: nn.LayerNorm(hidden_dim)
            for name in input_dims
        })

        self.transformers = nn.ModuleDict({
            name: nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 2,
                    dropout=0.2,
                    batch_first=True
                ),
                num_layers=num_layers
            )
            for name in input_dims
        })


        # Output regression MLP
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * len(input_dims), 128), #256),  # 2x perché mean + max
            nn.ReLU(),
            nn.Linear(128, 64), #256, 256),
            nn.ReLU(),
            #nn.Linear(256, 128),
            #nn.ReLU(),
            #nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, inputs):
        #print("Chiavi input ricevute:", inputs.keys())
        features = []

        for name, x in inputs.items():
            if x.ndim == 2:
                x = x.unsqueeze(1)  # [batch, 1, input_dim]
            
            x = self.embeddings[name](x)
            x = self.norms[name](x)
            x = self.transformers[name](x)
            x = x.mean(dim=1)
            #x_max = x.max(dim=1).values
            #x = torch.cat([x_mean, x_max], dim=-1)
            features.append(x)


        combined = torch.cat(features, dim=-1)
        out = self.regressor(combined)
        return out


# In[13]:


def train(dataloader, model, loss_fn, optimizer, loss_train, device):
    model.train()

    size = len(dataloader.dataset)
    for batch, (X, y, w) in enumerate(dataloader):

        X = {k: v.to(device) for k, v in X.items()}
        y, w = y.to(device), w.to(device)
        optimizer.zero_grad()

        pred = model(X)

        # VARIANZA FISSA
        var = torch.ones_like(pred) * 0.1

        # ⬇️ GaussianNLLLoss richiede: (mean, target, var)
        loss = loss_fn(pred, y, var)

        # Applico i pesi
        w = w.unsqueeze(1)
        loss = (loss * w / w.sum()).sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if batch % 100 == 0:
            print(f"Training loss: {loss.item():>7f}  [{(batch+1)*len(X):>5d}/{size:>5d}]")

    loss_train.append(loss.item())



# In[83]:

def test(dataloader, model, loss_fn, loss_test, acc_test):
    model.eval()
    num_samples = len(dataloader.dataset)
    avgloss = 0

    with torch.no_grad():
        for X, y, w in dataloader:
            X = {k: v.to(device) for k, v in X.items()}
            y, w = y.to(device), w.to(device)

            pred = model(X)

            var = torch.ones_like(pred) * 0.1

            # GaussianNLLLoss richiede var
            loss = loss_fn(pred, y, var)
            loss = (loss * w / w.sum()).sum()

            avgloss += loss.sum()

    avgloss /= num_samples
    loss_test.append(loss.item())

    print(f"Testing Error: Avg loss: {avgloss:>8f}")




# In[14]:


starttime=time.time()
train_dataloader, test_dataloader, input_dims = get_ATLAS_inputs(train_files, test_files, batch_size=256)


print(input_dims)
model = ATLASEventNN_MultiInput(input_dims=input_dims)


# In[15]:


for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: mean={param.mean().item()}, std={param.std().item()}")


# In[16]:


def init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.MultiheadAttention):
    # I pesi interni sono Linear: già gestiti, ma se servisse accedervi esplicitamente:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    elif isinstance(m, nn.TransformerEncoderLayer):
        # Inizializza i Linear interni (q, k, v, output)
        for name, param in m.named_parameters():
            if 'weight' in name and param.dim() > 1:
                init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
            elif 'bias' in name:
                init.constant_(param, 0.0)


# In[ ]:


import gc
gc.collect()
torch.cuda.empty_cache()

#model.apply(init_kaiming)
device = ( "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
print(f"\nUsing device \"{device}\"\n")

class total_loss(nn.Module):
    def __init__(self, beta=0.5, lambda_pyth=0.5, lambda_sign=0.3):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)
        self.lambda_pyth = lambda_pyth
        self.lambda_sign = lambda_sign

    def forward(self, pred, target):
        pred_metx, pred_mety, pred_met = pred[:,0], pred[:,1], pred[:,2]
        true_metx, true_mety, true_met = target[:,0], target[:,1], target[:,2]

        base_loss = self.smooth_l1(pred, target)

        # pitagor
        met_from_xy = torch.sqrt(pred_metx**2 + pred_mety**2 + 1e-8)
        pitagoras_loss = torch.mean((pred_met - met_from_xy) ** 2)

        #segni
        sign_loss_x = torch.mean((torch.sign(pred_metx) - torch.sign(true_metx))**2)
        sign_loss_y = torch.mean((torch.sign(pred_mety) - torch.sign(true_mety))**2)
        sign_loss = (sign_loss_x + sign_loss_y) / 2

        return base_loss + self.lambda_pyth * pitagoras_loss + self.lambda_sign* sign_loss

loss_fn = torch.nn.GaussianNLLLoss()
#MSELoss(reduction='none') #quando calcoli la loss se non hai la reduction ti da la media, ti  last term for each of the batch 
loss_fn = loss_fn.to(device)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
loss_test=[]
loss_train=[]
acc_test=[]

best_loss=1e6
patience_counter=0
patience=14

for t in range(epochs):
    
    print(f"\nEpoch {t+1}\n-------------------------------")
    starttime=time.time()
    train(train_dataloader, model, loss_fn, optimizer, loss_train, device)
    test(test_dataloader, model, loss_fn, loss_test, acc_test)
    print(loss_train)
    print(loss_test)
    print("Took %.2f minutes to run"%((time.time()-starttime)/60))

    #Early stopping criteria. If the new loss the best save it. Otherwise, after some patience threshold stop the loop
    #if I don't improve after 5 packs give up 
    if loss_test[-1]<best_loss:
        best_loss=loss_test[-1]
        patience_counter=0
        print(f"Saving model since best loss")
        torch.save(model.state_dict(), model_file)
    else:
        patience_counter+=1
        if patience_counter>=patience:
            print("Early stoping")
            break
print("Done!")


# In[ ]:


loss_test = [t for t in loss_test]
with uproot.recreate(f"/eos/user/g/gimainer/training_results_{sample}.root") as f:
    f["loss_tree"] = {"final_loss_train": np.array([loss_train[-1]], dtype=np.float32)}
    f["loss_tree"] = {"final_loss_test": np.array([loss_test[-1]], dtype=np.float32)}
print("Final loss test") 
print(np.array([loss_test[-1]], dtype=np.float32))
print("Final loss train") 
print(np.array([loss_train[-1]], dtype=np.float32))


# In[ ]:


figure = plt.figure()
plt.xlabel("Epoch")
plt.ylabel('Loss')
#loss_test = [t.cpu().item() for t in loss_test]
plt.plot(loss_train, color="orange", label="train")
plt.plot(loss_test, color="blue", label="test")
plt.legend(loc='lower right')
plt.savefig(f"/eos/user/g/gimainer/SWAN_projects/prova.py/plots/transformer/loss_{sample}.png")


# In[ ]:

met_tst_px_list = []
met_tst_py_list = []
truth_et_px_list = []
truth_et_py_list = []
mu_list = []
for test_file in test_files:
    with h5py.File(test_file, 'r') as h5_file:
        if "ttbar1l" not in test_file and "Zee" not in test_file and "Ztautau" not in test_file:
            met_tst_px_list.append(h5_file['Tight_TST_pxMiss'][:])
            met_tst_py_list.append(h5_file['Tight_TST_pyMiss'][:])
        else: 
            met_tst_px_list.append(h5_file['met_pxMiss_Tight_TST'][:])
            met_tst_py_list.append(['met_pyMiss_Tight_TST'][:])
        met_tst_px_list.append(h5_file["Tight_TST_pxMiss"][:])
        met_tst_py_list.append(h5_file["Tight_TST_pyMiss"][:])
        truth_et_px_list.append(h5_file["TruthpxMiss"][:])
        truth_et_py_list.append(h5_file["TruthpyMiss"][:])
        mu_list.append(h5_file["averageInteractionsPerCrossing"][:])

met_tst_et_px = np.concatenate(met_tst_px_list, axis=0)
met_tst_et_py = np.concatenate(met_tst_py_list, axis=0)
truth_et_px = np.concatenate(truth_et_px_list, axis=0)
truth_et_py = np.concatenate(truth_et_py_list, axis=0)
mu = np.concatenate(mu_list, axis=0)
met_tst_et_px = met_tst_et_px.flatten()
met_tst_et_py = met_tst_et_py.flatten()


model.eval()
truth_list = []
predicted_list = []
average_interactions_list = [] 

with torch.no_grad():  
    for X, y, w in test_dataloader:
        X = {k: v.to(device) for k, v in X.items()}
        y, w = y.to(device), w.to(device) 

        pred = model(X)  

        y = y.cpu().numpy()
        pred = pred.cpu().numpy()

        truth_list.append(y)
        predicted_list.append(pred)
        average_interactions = X["averageInteractionsPerCrossing"]  
        average_interactions_list.append(average_interactions.cpu().numpy())


truth_array = np.concatenate(truth_list, axis=0)  # Shape: (num_samples, 2)
predicted_array = np.concatenate(predicted_list, axis=0)  # Shape: (num_samples, 2)
averageinteractions_array = np.concatenate(average_interactions_list, axis=0)  # Shape: (num_samples, 2)


# In[ ]:




# In[ ]:


fig, axs = plt.subplots(1, 1, figsize=(10, 10))
met_tst_et = (met_tst_et_px**2 + met_tst_et_py**2)**0.5
truth_et = (truth_et_px**2+truth_et_py**2)**0.5
 #truth =(truth_array[:, 2])
 #pred = predicted_array[:, 2]
truth_x = (truth_array[:, 0])
truth_y= (truth_array[:,1])
truth =  (truth_x**2+truth_y**2)**0.5
pred_x = (predicted_array[:, 0])
pred_y= (predicted_array[:,1])
pred = (pred_x**2+pred_y**2)**0.5
#sumpT_truth = (truth_array[:, 2]) #3])
#sumpT_pred = (predicted_array[:,2]) #3])
max = len(truth)
#met_tst_et = met_tst_et[:max]

mean1, std1 = np.mean(met_tst_et), np.std(met_tst_et)
mean2, std2 = np.mean(truth), np.std(truth)
mean3, std3 = np.mean(pred), np.std(pred)

#axs.hist(met_tst_et, bins=800, histtype="step", density=True, color="green", label=f"TST MET(m={mean1:.2f}, s={std1:.2f})", linewidth=2)
axs.hist(truth, bins=800,  histtype="step", density=True, color="orange", label=f"Truth MET(m={mean2:.2f}, s={std2:.2f})", linewidth=2)
axs.hist(pred, bins=800,  histtype="step",  density=True, color="blue", label=f"Predicted MET(m={mean3:.2f}, s={std3:.2f})", linewidth=2)
axs.set_title("Comparison of MET distributions")
axs.set_xlabel("MET values")
axs.set_yscale("log")
axs.set_xlim(-1, 500)
axs.legend(loc="upper right")
axs.grid(True)

plt.tight_layout()
plt.savefig(f"/eos/user/g/gimainer/SWAN_projects/prova.py/plots/transformer/comparison_MET_vecpred_{sample}.png")
plt.show()


# In[ ]:


fig, axs = plt.subplots(1, 1, figsize=(10, 10))

mean1, std1 = np.mean(met_tst_et), np.std(met_tst_et)
mean2, std2 = np.mean(truth), np.std(truth)
mean3, std3 = np.mean(pred), np.std(pred)

# Istogrammi di confronto MET_
#counts1, bins1, _ = axs.hist(met_tst_et, bins=800, histtype="step", color="green", label=f"TST MET(m={mean1:.2f}, s={std1:.2f})", linewidth=2)
counts2, bins2, _ = axs.hist(truth, bins=800, histtype="step", color="orange", label=f"Truth MET(m={mean2:.2f}, s={std2:.2f})", linewidth=2)
counts3, bins3, _ = axs.hist(pred, bins=800, histtype="step", color="blue", label=f"Predicted MET(m={mean3:.2f}, s={std3:.2f})", linewidth=2)
axs.set_title("Comparison of MET distributions")
axs.set_xlabel("MET values")
axs.set_yscale("log")
axs.set_xlim(-1, 1100)
axs.legend(loc="upper right")
axs.grid(True)

plt.tight_layout()
plt.savefig(f"/eos/user/g/gimainer/SWAN_projects/prova.py/plots/transformer/comparison_MET_vecpred_{sample}_notnorm.png")
plt.show()
integral1 = counts1.sum()
integral2 = counts2.sum()
integral3 = counts3.sum()

print(f"Integral of TST MET: {integral1:.2f}")
print(f"Integral of Truth MET: {integral2:.2f}")
print(f"Integral of Predicted MET: {integral3:.2f}")






# In[ ]:





# In[ ]:



# Calcola la differenza (residui)
diff = pred - truth
diff_TST = met_tst_et - truth_et
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

axs[0].scatter(averageinteractions_array, diff, alpha=0.5, s=10, color="blue", label="Pred - Truth")
axs[0].axhline(0, color='red', linestyle='dashed', linewidth=2)
axs[0].set_xlabel("Mu")
axs[0].set_ylabel("Difference")
axs[0].set_title("Predicted MET - Truth MET vs Mu")
axs[0].legend()
axs[0].grid(True)

# Plot (MET - Truth) vs Mu
axs[1].scatter(mu, diff_TST, alpha=0.5, s=10, color="green", label="MET - Truth")
axs[1].axhline(0, color='red', linestyle='dashed', linewidth=2)
axs[1].set_xlabel("Mu")
axs[1].set_title("Measured MET - Truth MET vs Mu")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig(f"/eos/user/g/gimainer/SWAN_projects/prova.py/plots/transformer/residuals_vs_mu_{sample}.png")
plt.show()


bins = np.linspace(-200, 200, 50)  # Definisci i bin tra -200 e 200 GeV

plt.figure(figsize=(10, 6))
plt.hist(diff, bins=bins, alpha=0.7, label="NN Prediction", color="blue", histtype='step', linewidth=2)
plt.hist(diff_TST, bins=bins, alpha=0.7, label="Tight $E_T^{miss}$", color="green", histtype='step', linewidth=2)
plt.yscale('log')  
plt.xlabel("pTMiss - pTMiss Truth [GeV]")
plt.ylabel("Number of Events")
plt.title(f"Residuals distribution for sample: {sample}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"/eos/user/g/gimainer/SWAN_projects/prova.py/plots/transformer/residuals_hist_{sample}.png")
plt.show()


# In[ ]:


bins = np.linspace(0, 400, 30)  # 30 bins da 0 a 700 GeV
bin_centers = 0.5 * (bins[1:] + bins[:-1])
mean_diff = []
std_diff = []

for i in range(len(bins) - 1):
    mask = (truth >= bins[i]) & (truth < bins[i+1])
    residuals_in_bin = diff[mask]
    if len(residuals_in_bin) > 0:
        mean_diff.append(np.mean(residuals_in_bin))
        std_diff.append(np.std(residuals_in_bin))
    else:
        mean_diff.append(np.nan)
        std_diff.append(np.nan)
mean_diff_TST = []
std_diff_TST = []

for i in range(len(bins) - 1):
    mask = (truth_et >= bins[i]) & (truth_et < bins[i+1])
    residuals_in_bin = diff_TST[mask]
    if len(residuals_in_bin) > 0:
        mean_diff_TST.append(np.mean(residuals_in_bin))
        std_diff_TST.append(np.std(residuals_in_bin))
    else:
        mean_diff_TST.append(np.nan)
        std_diff_TST.append(np.nan)
plt.figure(figsize=(10, 6))
plt.fill_between(bin_centers, 
                 np.array(mean_diff) - np.array(std_diff), 
                 np.array(mean_diff) + np.array(std_diff), 
                 color='blue', alpha=0.3, label='GNMET Residuals Spread')
plt.plot(bin_centers, mean_diff, color='blue', linewidth=2)
plt.fill_between(bin_centers, 
                 np.array(mean_diff_TST) - np.array(std_diff_TST), 
                 np.array(mean_diff_TST) + np.array(std_diff_TST), 
                 color='green', alpha=0.3, label='TST Residuals Spread')
plt.plot(bin_centers, mean_diff_TST, color='green', linewidth=2)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel(r"True $E_T^{miss}$ [GeV]")
plt.ylabel("Average Residuals: Prediction - Truth [GeV]")
plt.title(r"Residuals vs $E_T^{{miss}}$ Truth")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.ylim(-100, 100)
plt.savefig(f"/eos/user/g/gimainer/SWAN_projects/prova.py/plots/transformer/residuals_vs_truth_profile_{sample}.png")
plt.show()


# In[ ]:


fig, axs = plt.subplots(2, 2, figsize=(12, 12))

axs[0,0].scatter(truth_x, pred_x, alpha=0.5, label="Predicted")
axs[0,0].set_xlabel("True MET_x")
axs[0,0].set_ylabel("Predicted MET_x")
axs[0,0].set_title("Predicted vs True MET_x")
axs[0,0].legend(loc="upper right")
axs[0,0].grid(True)
axs[0,0].set_xlim(-800, 800)
axs[0,0].set_ylim(-800, 800)

axs[1,0].scatter(truth_y, pred_y, alpha=0.5,label="Predicted", color="orange")
axs[1,0].set_xlabel("True MET_y")
axs[1,0].set_ylabel("Predicted MET_y")
axs[1,0].set_title("Predicted vs True MET_y")
axs[1,0].legend(loc="upper right")
axs[1,0].grid(True)
axs[1,0].set_xlim(-800, 800)
axs[1,0].set_ylim(-800, 800)


axs[0,1].scatter(truth_et_px, met_tst_et_px, alpha=0.5, label="MET_tst_px", color="green")
axs[0,1].set_xlabel("True MET_x")
axs[0,1].set_ylabel("TST MET_x")
axs[0,1].set_title("TST vs True MET_x")
axs[0,1].legend(loc="upper right")
axs[0,1].grid(True)
axs[0,1].set_xlim(-800, 800)
axs[0,1].set_ylim(-800, 800)


axs[1,1].scatter(truth_et_py, met_tst_et_py, alpha=0.5, label="MET_tst_py", color="green")
axs[1,1].set_xlabel("True MET_y")
axs[1,1].set_ylabel("TST MET_y")
axs[1,1].set_title("TST vs True MET_y")
axs[1,1].legend(loc="upper right")
axs[1,1].grid(True)
axs[1,1].set_xlim(-800, 800)
axs[1,1].set_ylim(-800, 800)

plt.tight_layout()
plt.savefig(f"/eos/user/g/gimainer/SWAN_projects/prova.py/plots/transformer/pre_vs_truth_{sample}.png")
plt.show()



# In[ ]:


bins = np.linspace(0, 700, 25)
labels = (bins[:-1] + bins[1:]) / 2
df = pd.DataFrame({
    'met_truth': truth,
    'res_nn': diff
})
df_TST = pd.DataFrame({
    'met_truth_TST': truth_et, 
    'res_tst': diff_TST
})
df['bin'] = pd.cut(df['met_truth'], bins=bins, labels=labels)
df_TST['bin_TST'] = pd.cut(df_TST['met_truth_TST'], bins=bins, labels=labels)
means_nn = df.groupby('bin')['res_nn'].mean()
means_tst = df_TST.groupby('bin_TST')['res_tst'].mean()
plt.figure(figsize=(10, 6))
plt.plot(labels, means_nn, label="NN Mean Residual", color="blue", marker='o')
plt.plot(labels, means_tst, label="TST Mean Residual", color="green", marker='s')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("MET Truth [GeV]")
plt.ylabel("Mean Residual [GeV]")
plt.title(f"Mean Residuals vs MET Truth - Sample: {sample}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"/eos/user/g/gimainer/SWAN_projects/prova.py/plots/transformer/mean_residuals_vs_truth_{sample}.png")
plt.show()




# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(20, 8))


mean1, std1 = np.mean(met_tst_et_px), np.std(met_tst_et_px)
mean2, std2 = np.mean(truth_x), np.std(truth_x)
mean3, std3 = np.mean(pred_x), np.std(pred_x)

# Istogrammi di confronto MET_x
axs[0].hist(met_tst_et_px, bins=800, histtype="step",  color="green", label=f"TST MET_x(m={mean1:.2f}, s={std1:.2f})", linewidth=2, density=True)
axs[0].hist(truth_x, bins=800,  histtype="step", color="orange", label=f"Truth MET_x(m={mean2:.2f}, s={std2:.2f})", linewidth=2, density=True)
axs[0].hist(pred_x, bins=800,  histtype="step",  color="blue", label=f"Predicted MET_x(m={mean3:.2f}, s={std3:.2f})", linewidth=2, density=True)
axs[0].set_title("Comparison of MET_x distributions")
axs[0].set_xlabel("MET_x values")
axs[0].set_yscale("log")
axs[0].set_xlim(-800, 800)
axs[0].legend()
axs[0].grid(True)

mean1, std1 = np.mean(met_tst_et_py), np.std(met_tst_et_py)
mean2, std2 = np.mean(truth_y), np.std(truth_y)
mean3, std3 = np.mean(pred_y), np.std(pred_y)

# Istogrammi di confronto MET_y
axs[1].hist(met_tst_et_py, bins=800, histtype="step", color="green", label=f"TST MET_y(m={mean1:.2f}, s={std1:.2f})", linewidth=2, density=True)
axs[1].hist(truth_y, bins=800, histtype="step",color="orange", label=f"Truth MET_y(m={mean2:.2f}, s={std2:.2f})", linewidth=2, density=True)
axs[1].hist(pred_y, bins=800, histtype="step", color="blue", label=f"Predicted MET_y(m={mean3:.2f}, s={std3:.2f})", linewidth=2, density=True)
axs[1].set_title("Comparison of MET_y distributions")
axs[1].set_xlabel("MET_y values")
axs[1].set_yscale("log")
axs[1].set_xlim(-800, 800)
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig(f"/eos/user/g/gimainer/SWAN_projects/prova.py/plots/transformer/comparison_MET_{sample}.png")
plt.show()



# In[ ]:





# In[ ]:


nbins=40
bin_edges = np.linspace(0,200, nbins+1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.figure(figsize=(8, 6))
plt.plot(bin_centers, calculate_lin(met_tst_et_px, truth_et_px, bin_edges),  marker='o',color='blue', label="TST")
plt.plot(bin_centers, calculate_lin(pred_x,truth_x, bin_edges), marker='o',color='orange',  label="pred")
plt.axhline(0, color='r', linestyle='--', label="Linearity Reference (y = 0)")
plt.title("Linearity: MET_x - truth_x vs truth")
plt.xlabel("Truth")
plt.ylabel("MET_x - truth_x")
plt.legend(loc="upper right")
plt.show()
plt.savefig(f"/eos/user/g/gimainer/SWAN_projects/prova.py/plots/transformer/linearity_METpred_{sample}_x.png")

plt.figure(figsize=(8, 6))
plt.plot(bin_centers, calculate_lin(met_tst_et_py, truth_et_py, bin_edges),  marker='o',color='blue', label="TST")
plt.plot(bin_centers, calculate_lin(pred_y,truth_y, bin_edges), marker='o',color='orange',  label="pred")
plt.axhline(0, color='r', linestyle='--', label="Linearity Reference (y = 0)")
plt.title("Linearity: MET_y - truth_y vs truth")
plt.xlabel("Truth")
plt.ylabel("MET_y - truth_y")
plt.legend(loc="upper right")
plt.show()
plt.savefig(f"/eos/user/g/gimainer/SWAN_projects/prova.py/plots/transformer/linearity_METpred_{sample}_y.png")

plt.figure(figsize=(8, 6))
plt.plot(bin_centers, calculate_lin(met_tst_et, truth_et, bin_edges), marker='o',color='blue', label="TST")
plt.plot(bin_centers, calculate_lin(pred, truth, bin_edges), marker='o',color='orange',  label="pred")
plt.axhline(0, color='r', linestyle='--')
plt.title("Linearity: MET_abs - truth_abs vs truth")
plt.xlabel("Truth")
plt.ylabel("MET_abs - truth_abs")
plt.legend(loc="upper right")
plt.show()
plt.savefig(f"/eos/user/g/gimainer/SWAN_projects/prova.py/plots/transformer/linearity_METpred_{sample}_abs.png")


# In[ ]:


fig, axs = plt.subplots(1, 1, figsize=(10, 10))

mean2, std2 = np.mean(sumpT_truth), np.std(sumpT_truth)
mean3, std3 = np.mean(sumpT_pred), np.std(sumpT_pred)

axs.hist(sumpT_truth, bins=800,  histtype="step", density="true", color="orange", label=f"Truth MET(m={mean2:.2f}, s={std2:.2f})", linewidth=2)
axs.hist(sumpT_pred, bins=800,  histtype="step", density="true",color="blue", label=f"Predicted MET(m={mean3:.2f}, s={std3:.2f})", linewidth=2)
axs.set_title("Comparison of sumpT distributions")
axs.set_xlabel("sumpT values")
axs.set_yscale("log")
axs.set_xlim(-1,1100)
axs.legend(loc="upper right")
axs.grid(True)

plt.tight_layout()
plt.savefig(f"/eos/user/g/gimainer/SWAN_projects/prova.py/plots/transformer/comparison_SumpT_vecpred_{sample}_notnorm.png")
plt.show()


fig, axs = plt.subplots(1, 1, figsize=(10, 10))

mean2, std2 = np.mean(sumpT_truth), np.std(sumpT_truth)
mean3, std3 = np.mean(sumpT_pred), np.std(sumpT_pred)

axs.hist(sumpT_truth, bins=800,  histtype="step", color="orange", label=f"Truth MET(m={mean2:.2f}, s={std2:.2f})", linewidth=2)
axs.hist(sumpT_pred, bins=800,  histtype="step", color="blue", label=f"Predicted MET(m={mean3:.2f}, s={std3:.2f})", linewidth=2)
axs.set_title("Comparison of sumpT distributions")
axs.set_xlabel("sumpT values")
axs.set_yscale("log")
axs.set_xlim(-1,1100)
axs.legend(loc="upper right")
axs.grid(True)

plt.tight_layout()
plt.savefig(f"/eos/user/g/gimainer/SWAN_projects/prova.py/plots/transformer/comparison_SumpT_vecpred_{sample}_notnorm.png")
plt.show()


# In[611]:


# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(20, 8))
nbins=25
bin_edges = np.linspace(np.min(mu), np.max(mu), nbins+1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

plt.figure(figsize=(10, 6))
plt.plot(bin_centers, calculate_rms(mu, met_tst_et_px, truth_et_px, nbins), marker='o', color='blue', label="RMS (TST - truth)")
plt.plot(bin_centers, calculate_rms(averageinteractions_array, pred_x, truth_x, nbins), marker='o', color='orange', label="RMS (pred - truth)")
plt.xlabel("mu")
plt.ylabel("RMS")
plt.title("RMS MET_x-MET_x_truth vs AverageInteractionsPerCrossing")
plt.grid(True)
plt.legend()
plt.show()
plt.savefig(f"/eos/user/g/gimainer/SWAN_projects/prova.py/plots/transformer/resolution_METpred_{sample}_x.png")

plt.figure(figsize=(10, 6))
plt.plot(bin_centers, calculate_rms(mu, met_tst_et_py, truth_et_py, nbins), marker='o', color='blue', label="RMS (TST - truth)")
plt.plot(bin_centers, calculate_rms(averageinteractions_array, pred_y, truth_y, nbins), marker='o', color='orange', label="RMS (pred - truth)")
plt.xlabel("mu")
plt.ylabel("RMS")
plt.title("RMS MET_y-MET_y_truth vs AverageInteractionsPerCrossing")
plt.grid(True)
plt.legend()
plt.show()
plt.savefig(f"/eos/user/g/gimainer/SWAN_projects/prova.py/plots/transformer/resolution_METpred_{sample}_y.png")

plt.figure(figsize=(10, 6))
plt.plot(bin_centers, calculate_rms(mu, met_tst_et, truth_et, nbins), marker='o', color='blue', label="RMS (TST - truth)")
plt.plot(bin_centers, calculate_rms(averageinteractions_array, pred, truth, nbins), marker='o', color='orange', label="RMS (pred - truth)")
plt.xlabel("mu")
plt.ylabel("RMS")
plt.title("RMS MET-MET_truth vs AverageInteractionsPerCrossing")
plt.grid(True)
plt.legend()
plt.show()
plt.savefig(f"/eos/user/g/gimainer/SWAN_projects/prova.py/plots/transformer/resolution_METpred_{sample}_abs.png")






# In[ ]:




