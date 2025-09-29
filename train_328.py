import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from datasetsss_328 import MyDataset
from syncnet_328 import SyncNet_color
from unet_328 import Model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser(
        description='Train (328 variant)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--use_syncnet', action='store_true',
                        help='Enable SyncNet-based audio-visual loss.')
    parser.add_argument('--syncnet_checkpoint', type=str, default='',
                        help='Path to the pre-trained SyncNet checkpoint.')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Root directory of the dataset.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to store trained checkpoints.')
    parser.add_argument('--see_res', action='store_true',
                        help='Dump a visual sample each epoch for inspection.')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--asr', type=str, default='hubert',
                        choices=['hubert', 'wenet', 'ave'])
    parser.add_argument('--weighted_sampling', action='store_true',
                        help='Enable sampling with per-frame weights if available.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading.')
    parser.add_argument('--perceptual', type=str, default='mobilenetv3',
                        choices=['vgg', 'resnet', 'mobilenetv3', 'none'])
    parser.add_argument('--perc_backbone', type=str, default='mbv3_small',
                        choices=['resnet18', 'resnet34', 'resnet50',
                                 'vgg19', 'vgg16',
                                 'mbv3_large', 'mbv3_small'])
    parser.add_argument('--perc_layer', type=str, default='mv3_12',
                        choices=['l2', 'l3', 'l4', 'c32',
                                 'mv3_12', 'mv3_16', 'mv3_last'])
    parser.add_argument('--pixel_weight', type=float, default=1.0)
    parser.add_argument('--perc_weight', type=float, default=0.01)
    parser.add_argument('--sync_w', type=float, default=10.0)
    parser.add_argument('--use_scheduler', action='store_true',
                        help='Use cosine learning-rate scheduler.')
    parser.add_argument('--early_stop', type=int, default=60,
                        help='Stop training if the loss plateaus for N epochs.')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm.')
    return parser.parse_args()


args = get_args()


class Perceptual(nn.Module):
    def __init__(self, kind='mobilenetv3', backbone='mbv3_small', layer='mv3_12'):
        super().__init__()
        self.kind = kind
        self.layer = layer

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.backbone = None
        self.crit = nn.L1Loss()

        if kind == 'none':
            return

        if kind == 'mobilenetv3':
            if backbone == 'mbv3_small':
                weights = tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                mb = tv_models.mobilenet_v3_small(weights=weights)
            else:
                weights = tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
                mb = tv_models.mobilenet_v3_large(weights=weights)
            feats = list(mb.features.children())
            if layer == 'mv3_12':
                cut = 12
            elif layer == 'mv3_16':
                cut = 16
            else:
                cut = len(feats)
            self.backbone = nn.Sequential(*feats[:cut]).eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
            return

        if kind == 'vgg':
            use_19 = backbone == 'vgg19'
            weights = (tv_models.VGG19_Weights.IMAGENET1K_V1 if use_19
                       else tv_models.VGG16_Weights.IMAGENET1K_V1)
            vgg = (tv_models.vgg19(weights=weights) if use_19
                   else tv_models.vgg16(weights=weights)).features
            layer_map = {'l2': 9, 'l3': 16, 'l4': 23, 'c32': 30}
            cut = layer_map.get(layer, layer_map['l3'])
            self.backbone = nn.Sequential(*list(vgg.children())[:cut]).eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
            return

        if kind == 'resnet':
            resnet_map = {
                'resnet18': (tv_models.resnet18, tv_models.ResNet18_Weights.IMAGENET1K_V1),
                'resnet34': (tv_models.resnet34, tv_models.ResNet34_Weights.IMAGENET1K_V1),
                'resnet50': (tv_models.resnet50, tv_models.ResNet50_Weights.IMAGENET1K_V2),
            }
            ctor, weights_enum = resnet_map.get(
                backbone,
                (tv_models.resnet18, tv_models.ResNet18_Weights.IMAGENET1K_V1),
            )
            res = ctor(weights=weights_enum)
            modules = [
                res.conv1,
                res.bn1,
                res.relu,
                res.maxpool,
                res.layer1,
                res.layer2,
                res.layer3,
                res.layer4,
            ]
            layer_map = {'l2': 6, 'l3': 7, 'l4': 8, 'c32': 7}
            cut = layer_map.get(layer, 7)
            self.backbone = nn.Sequential(*modules[:cut]).eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
            return

        # Fallback to disabled perceptual loss if configuration is unsupported.
        self.kind = 'none'

    def _imnet_norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, x_fake: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
        if self.kind == 'none' or self.backbone is None:
            return torch.zeros((), device=x_fake.device)

        xf = self._imnet_norm(x_fake)
        xr = self._imnet_norm(x_real).detach()
        return self.crit(self.backbone(xf), self.backbone(xr))


logloss = nn.BCELoss()


def cosine_loss(a: torch.Tensor, v: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    d = nn.functional.cosine_similarity(a, v)
    return logloss(d.unsqueeze(1), y)


def make_dataloader(dataset_dir: str, batch_size: int):
    dataset = MyDataset(dataset_dir, args.asr)
    sampler = None

    if args.weighted_sampling:
        wpath_hard = os.path.join(dataset_dir, 'weights_hard.npy')
        wpath = wpath_hard if os.path.isfile(wpath_hard) else os.path.join(dataset_dir, 'weights.npy')
        if os.path.isfile(wpath):
            weights = np.load(wpath).astype(np.float32)
            if len(weights) < len(dataset):
                rep = int(np.ceil(len(dataset) / max(1, len(weights))))
                weights = np.tile(weights, rep)[:len(dataset)]
            else:
                weights = weights[:len(dataset)]
            sampler = WeightedRandomSampler(
                torch.from_numpy(weights),
                num_samples=len(dataset),
                replacement=True,
            )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )
    return dataset, loader


def train(net: nn.Module, epoch: int, batch_size: int, lr: float) -> None:
    content_loss = Perceptual(
        kind=args.perceptual,
        backbone=args.perc_backbone,
        layer=args.perc_layer,
    ).to(device)
    content_loss.eval()

    syncnet = None
    if args.use_syncnet:
        if args.syncnet_checkpoint == '':
            raise ValueError("Using syncnet requires '--syncnet_checkpoint'.")
        syncnet = SyncNet_color(args.asr).eval().to(device)
        try:
            state = torch.load(args.syncnet_checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(args.syncnet_checkpoint, map_location=device)
        syncnet.load_state_dict(state)

    os.makedirs(args.save_dir, exist_ok=True)

    dataset_dir_list = [d.strip() for d in args.dataset_dir.split(',') if d.strip()]
    if not dataset_dir_list:
        raise ValueError('No valid dataset directory provided.')

    dataset_list = []
    dataloader_list = []
    for dataset_dir in dataset_dir_list:
        dataset, loader = make_dataloader(dataset_dir, batch_size)
        dataset_list.append(dataset)
        dataloader_list.append(loader)

    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.L1Loss()

    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epoch,
            eta_min=lr * 0.01,
        )

    best_loss = float('inf')
    patience_counter = 0

    for e in range(epoch):
        net.train()
        random_i = random.randint(0, len(dataset_dir_list) - 1)
        dataset = dataset_list[random_i]
        train_loader = dataloader_list[random_i]

        epoch_loss = 0.0
        num_batches = 0

        with tqdm(total=len(dataset), desc=f'Epoch {e + 1}/{epoch}', unit='img') as progress:
            for imgs, labels, audio_feat in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                audio_feat = audio_feat.to(device, non_blocking=True)

                preds = net(imgs, audio_feat)

                sync_loss = torch.zeros((), device=device)
                if args.use_syncnet and syncnet is not None:
                    audio_feat_sync = audio_feat
                    if args.asr == 'hubert':
                        channels = audio_feat.shape[1]
                        if channels == 16:
                            audio_feat_sync = torch.cat([audio_feat, audio_feat], dim=1)
                        elif 32 % channels == 0:
                            audio_feat_sync = audio_feat.repeat(1, 32 // channels, 1, 1)
                        elif channels > 32:
                            audio_feat_sync = audio_feat[:, :32, :, :]
                        else:
                            need = 32 - channels
                            audio_feat_sync = torch.cat([audio_feat, audio_feat[:, :need, :, :]], dim=1)
                    y = torch.ones((preds.shape[0], 1), device=device, dtype=torch.float32)
                    a, v = syncnet(preds, audio_feat_sync)
                    sync_loss = cosine_loss(a, v, y)

                loss_perc = content_loss(preds, labels) if args.perc_weight > 0 else torch.zeros((), device=device)
                loss_pixel = criterion(preds, labels)

                loss = args.pixel_weight * loss_pixel + args.perc_weight * loss_perc
                if args.use_syncnet:
                    loss = loss + args.sync_w * sync_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

                optimizer.step()

                batch_loss = loss.detach()
                epoch_loss += batch_loss.item()
                num_batches += 1

                progress.set_postfix(loss=float(batch_loss))
                progress.update(imgs.size(0))

        avg_loss = epoch_loss / num_batches if num_batches else float('inf')
        print(f'Epoch {e + 1} Average Loss: {avg_loss:.6f}')

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f'Learning Rate: {current_lr:.6f}')

        if e % 5 == 0:
            torch.save(net.state_dict(), os.path.join(args.save_dir, f'epoch_{e}.pth'))

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(net.state_dict(), os.path.join(args.save_dir, 'best.pth'))
            print(f'Best model updated: loss={best_loss:.6f}')
        else:
            patience_counter += 1
            if args.early_stop > 0 and patience_counter >= args.early_stop:
                print(f'Early stopping at epoch {e + 1}')
                break

        if args.see_res:
            net.eval()
            sample_idx = random.randint(0, len(dataset) - 1)
            img_concat, img_real, audio_feat = dataset.__getitem__(sample_idx)
            img_concat = img_concat[None].to(device)
            audio_feat = audio_feat[None].to(device)
            with torch.no_grad():
                pred = net(img_concat, audio_feat)[0]
            pred = (pred.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img_real = (img_real.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            os.makedirs('./train_tmp_img', exist_ok=True)
            cv2.imwrite(f'./train_tmp_img/epoch_{e}.jpg', pred)
            cv2.imwrite(f'./train_tmp_img/epoch_{e}_real.jpg', img_real)

    torch.save(net.state_dict(), os.path.join(args.save_dir, 'final.pth'))
    print(f'Training completed. Best loss: {best_loss:.6f}')


if __name__ == '__main__':
    model = Model(6, mode=args.asr).to(device)
    train(model, args.epochs, args.batchsize, args.lr)
