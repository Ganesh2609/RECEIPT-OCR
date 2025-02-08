import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_batch(model:torch.nn.Module,
                batch:tuple,
                loss_fn:torch.nn.Module,
                optimizer:torch.optim.Optimizer,
                scaler:torch.cuda.amp.GradScaler,
                device:torch.device):
    
    x, y = batch['Input'], batch['Target']
    x, y = x.to(device), y.to(device)
    
    model.train()
    
    with torch.amp.autocast(enabled=True, device_type='cuda'):
        reconstructed = model(x)
        loss = loss_fn(reconstructed, y)
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    del x, y, reconstructed
    return loss.item()



def train_models(model:torch.nn.Module,
                dataloader:torch.utils.data.DataLoader,
                loss_fn:torch.nn.Module,
                optimizer:torch.optim.Optimizer,
                scaler:torch.cuda.amp.GradScaler,
                device:torch.device,
                NUM_EPOCHS:int,
                model_path:str=None,
                result_path:str=None
                ):
    
    for epoch in range(8, NUM_EPOCHS+8):
        
        loss = 0
        
        with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
            
            for i, batch in t:    
                
                batch_loss = train_batch(model=model,
                                         batch=batch,
                                         loss_fn=loss_fn,
                                         optimizer=optimizer,
                                         scaler=scaler,
                                         device=device)
                loss += batch_loss
                
                t.set_description(f'Epoch [{epoch}/{NUM_EPOCHS+7}]')
                t.set_postfix({
                    'Batch Loss' : batch_loss,
                    'Train Loss' : loss/(i+1)
                })
                
                if i % 100 == 0 and model_path:
                    torch.save(obj=model.state_dict(), f=model_path)
            
                if i % 100 == 0 and result_path:
                    
                    RESULT_SAVE_NAME = result_path + f'/Epoch_{epoch}'
                    
                    x, y = batch['Input'].to(device), batch['Target'].to(device)
                    model.eval()
                    reconstructed = model(x)
                    plot_inp = torch.cat(tensors=(x[:5], y[:5], reconstructed[:5]), dim=0).cpu()
                    
                    grid = torchvision.utils.make_grid(tensor=plot_inp, nrow=5, normalize=True, padding=16, pad_value=1)
                    fig = plt.figure(figsize=(10, 6))
                    plt.imshow(grid.permute(1, 2, 0))
                    plt.title(f'Epoch_{epoch}')
                    plt.axis(False);
                    plt.savefig(RESULT_SAVE_NAME)
                    plt.close(fig)