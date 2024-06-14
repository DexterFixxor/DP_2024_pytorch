from torch.utils.data import Dataset
from typing import List
import torch

class MyDataset(Dataset):
    
    def __init__(self, paths_to_csv : List[str]) -> None:
        super().__init__()
        
        # ucitati sve CSV fajlove u jednu promenljivu
                            # frames
        # [ID_klase, fajl, [x, y, z]]
        
        self.broj_klasa = len(paths_to_csv)
        self.broj_fajlova_po_klasi = 8
        
        self.dummy_data = torch.randn((2, self.broj_fajlova_po_klasi, 300, 3))
        
    def __getitem__(self, index):
        
        klasa_id = index // self.broj_fajlova_po_klasi
        fajl = index % self.broj_fajlova_po_klasi
        input_data = self.dummy_data[klasa_id][fajl]
        # prebacujemo sve u relativne pomeraje u odnosu na pocetni trenutak
        input_data = input_data[klasa_id][fajl][:] - input_data[klasa_id][fajl][0]
        y = torch.zeros(2) # broj klasa OVO TREBA IZMENITI. self.broj_klasa
        y[klasa_id] = 1.0
        return input_data, y
    def __len__(self):
        # treba da vrati MAX vrednost indeksa koji moze biti prosledjen u __getitem__
        # broj_klasa * broj_fajlova_po_klasi (npr 2 * 8)
        return 16
        
