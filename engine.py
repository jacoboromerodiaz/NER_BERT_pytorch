import torch
from tqdm import tqdm 

def train_fn(dataloader, model, optimizer, device, lr_policy):
  model.train()
  train_loss = 0

  for i, data in enumerate(tqdm(dataloader, total=len(dataloader))):
    temp_data = {}
    for k, v in data.items():
        temp_data[k] = v.to(device, dtype= torch.long)
    optimizer.zero_grad()
    output, loss = model(**temp_data)
    loss.backward()
    optimizer.step()
    lr_policy.step()
    train_loss += loss.item()
  return train_loss / len(dataloader)

def eval_fn(dataloader, model, device):
  model.eval()
  val_loss = 0
  val_targets = torch.empty(0)
  val_outputs = torch.empty(0)
  for i, data in enumerate(tqdm(dataloader, total=len(dataloader))):
    temp_data = {}
    for k, v in data.items():
        temp_data[k] = v.to(device, dtype= torch.long)
    output, loss = model(**temp_data)
    val_loss += loss.item()
    val_targets = torch.cat((val_targets, data["target_tags"].view(-1)), dim=0)
    val_outputs = torch.cat((val_outputs, output.argmax(2).cpu().detach().view(-1)), dim=0)
  return val_loss / len(dataloader), val_targets, val_outputs
