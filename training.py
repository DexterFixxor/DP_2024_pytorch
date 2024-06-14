from model import LSTMagija
from dataset import MyDataset
import dask.dataframe as dd
import torch
from torch import optim
from torch.utils.data import DataLoader

model = LSTMagija()
criterion = torch.nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.001)
my_dataset = MyDataset([])
my_dataloader = DataLoader(my_dataset, batch_size=4, shuffle=True)

num_epochs = 100
for i in range(1, num_epochs + 1):
    print("-"*20)
    print(f"Epoch [{i}/{num_epochs+1}]:")
    for x, y in my_dataloader:
        out = model(x)
        loss = criterion(y, out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
        
    












