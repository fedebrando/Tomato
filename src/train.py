import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from metrics import acc_metric
from model import UNET
from dataset import TomatoDataset
from params import VAL_PCT

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, writer, epochs=1, early_stopping_patience=200):
    start = time.time()
    model.cuda()

    train_loss, valid_loss = [], []
    best_val_acc = 0.0
    epochs_no_improve = 0

    MODEL_DIR = '../models'
    os.makedirs(MODEL_DIR, exist_ok=True)

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            model.train(phase == 'train')
            dataloader = train_dl if phase == 'train' else valid_dl

            running_loss = 0.0
            running_acc = 0.0

            # ✅ Aggiunta di tqdm
            for step, (x, y) in enumerate(tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {epoch}")):
                x = x.cuda()
                y = y.cuda()

                if phase == 'train':
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())

                acc = acc_fn(outputs, y)
                running_loss += loss.item()
                running_acc += acc.item()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            writer.add_scalar(f'{phase}/loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase}/accuracy', epoch_acc, epoch)

            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                valid_loss.append(epoch_loss)

                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    epochs_no_improve = 0
                    print(f"✅ Saving BEST model at epoch {epoch} with val_acc: {epoch_acc:.4f}")
                    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pth'))
                else:
                    epochs_no_improve += 1
                    print(f"⚠️ No improvement in validation accuracy for {epochs_no_improve} epoch(s)")

        if epochs_no_improve > early_stopping_patience:
            print(f"⏹️ Early stopping triggered after {epoch + 1} epochs (no improvement in {early_stopping_patience} epochs).")
            break

    print("💾 Saving LAST model after final epoch")
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'last_model.pth'))

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return train_loss, valid_loss

def main():
    unet = UNET(3, 6)
    loss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=0.01)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = TomatoDataset(os.path.join('..', 'images'), transform=transform, target_transform=transform, mode='train')
    train_data_len = len(train_data)

    val_len = int(train_data_len * VAL_PCT)
    train_len = train_data_len - val_len

    train_dataset, val_dataset = random_split(train_data, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    writer = SummaryWriter(log_dir='../runs/tomato_segmentation_experiment')

    train_loss, valid_loss = train(unet, train_loader, val_loader, loss, opt, acc_metric, writer, epochs=100)

    writer.close()


if __name__ == '__main__':
    main()
