import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from metrics import acc_metric
from model import UNET
from dataset import TomatoDataset
from params import VAL_PCT
from arg_parser import get_train_args

SEED_FOR_VAL_SET_REPRODUCIBILITY = 42
AUGM = True # TODO convert into input param

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, writer, args, model_name, epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    start = time.time()
    model.to(device)
    train_loss, valid_loss = [], []
    best_val_acc = 0.0
    epochs_no_improve = 0

    MODEL_DIR = '../models/' + model_name
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
                x = x.to(device)
                y = y.to(device)

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

            if (epoch + 1) % args.print_every == 0:
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

        if epochs_no_improve > args.early_stop:
            print(f"⏹️ Early stopping triggered after {epoch + 1} epochs (no improvement in {args.early_stop} epochs).")
            break

    print("💾 Saving LAST model after final epoch")
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'last_model.pth'))

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))

    return train_loss, valid_loss, time_elapsed

def main(args):
    unet = UNET(3, 6)
    loss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=args.lr) if args.opt == 'Adam' else torch.optim.SGD(unet.parameters(), lr=args.lr)

    train_data = TomatoDataset(os.path.join('images'), flip_prob=args.aug_flip, rotate_prob=args.aug_rotate, jitter_prob=args.aug_jitter, crop_prob=args.aug_crop, mode='train')
    train_data_len = len(train_data)

    val_len = int(train_data_len * VAL_PCT)
    train_len = train_data_len - val_len

    train_indices, val_indices = random_split(
      range(train_data_len),
      [train_len, val_len],
      generator = torch.Generator().manual_seed(SEED_FOR_VAL_SET_REPRODUCIBILITY)
    )
    
    # Training set
    train_dataset = Subset(train_data, train_indices)
    
    # Validation set
    train_data_no_augm = TomatoDataset(os.path.join('images'), mode='train')
    val_dataset = Subset(train_data_no_augm, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
    
    if args.refine_model:
      unet.load_state_dict(torch.load(args.refine_model, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=True))
    
    model_name = f"{args.opt}_lr{args.lr:.0e}_bs{args.bs}_es{args.early_stop}" \
                  + (f'_flip{args.aug_flip}' if args.aug_flip else '') \
                  + (f'_rotate{args.aug_rotate}' if args.aug_rotate else '') \
                  + (f'_jitter{args.aug_jitter}' if args.aug_jitter else '') \
                  + (f'_crop{args.aug_crop}' if args.aug_crop else '') \
                  + (f'_refined' if args.refine_model else '')
                  

    writer = SummaryWriter(log_dir='../runs/' + model_name + "/train/")

    args_table = "| Nome parametro | Valore |\n|---|---|\n"
    for k, v in vars(args).items():
        args_table += f"| {k} | {v if bool(v) else 'None'} |\n"
    writer.add_text("Hyperparameters", args_table, 0)

    _, _, time_elapsed = train(unet, train_loader, val_loader, loss, opt, acc_metric, writer, args, model_name, epochs=args.epochs)
    args_table = "| Training time |\n|---|\n"
    args_table += f"| {'{:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60)} |\n"
    writer.add_text("Training time", args_table, 0)

    writer.close()


if __name__ == '__main__':
    args = get_train_args()
    main(args)