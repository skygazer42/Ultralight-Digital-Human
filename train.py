import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasetsss import MyDataset
from syncnet import SyncNet_color
from unet_plus import Model
from unet import Model
import random
import torchvision.models as tv_models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--use_syncnet', action='store_true')
    parser.add_argument('--syncnet_checkpoint', type=str, default="")
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--see_res', action='store_true')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--asr', type=str, default="hubert", choices=["hubert", "wenet"])
    parser.add_argument('--weighted_sampling', action='store_true')
    # 感知损失与SyncNet
    parser.add_argument('--perceptual', type=str, default='mobilenetv3',
                        choices=['vgg', 'resnet', 'mobilenetv3', 'lpips', 'none'],
                        help=' vgg / resnet / mobilenetv3 / lpips / none')
    parser.add_argument('--perc_backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50',
                                 'vgg19', 'vgg16',
                                 'mbv3_large', 'mbv3_small'],
                        help='perceptual=resnet/vgg/mobilenetv3 backbone')
    parser.add_argument('--perc_layer', type=str, default='l4',
                        choices=['l2', 'l3', 'l4', 'c32',  # resnet/vgg
                                 'mv3_12', 'mv3_16', 'mv3_last'],  # mobilenetv3
                        help='resnet: l2/l3/l4；vgg: c32；mobilenetv3: mv3_12/mv3_16/mv3_last')
    parser.add_argument('--sync_w', type=float, default=10.0,
                        help='SyncNet loss weights se_syncnet')

    # unet_plus
    parser.add_argument('--width_mult', type=float, default=1.0, help='channel scale：1.0 / 0.75 / 0.6')
    parser.add_argument('--upsample', type=str, default='nearest', choices=['nearest', 'bilinear'])
    parser.add_argument('--cross_attn', action='store_true', help='Tiny Cross-Attn')

    return parser.parse_args()
args = get_args()
use_syncnet = args.use_syncnet

class Perceptual(nn.Module):
    def __init__(self, kind='resnet', backbone='resnet18', layer='l4'):
        super().__init__()
        self.kind = kind
        self.layer = layer

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if kind == 'vgg':
            if backbone == 'vgg16':
                weights = tv_models.VGG16_Weights.IMAGENET1K_V1
                vgg = tv_models.vgg16(weights=weights).features
            else:
                weights = tv_models.VGG19_Weights.IMAGENET1K_V1
                vgg = tv_models.vgg19(weights=weights).features
            cut = 16 if layer == 'c32' else 16
            self.backbone = nn.Sequential(*[vgg[i] for i in range(cut)]).eval()
            for p in self.backbone.parameters(): p.requires_grad = False
            self.crit = nn.MSELoss()

        elif kind == 'resnet':
            if backbone == 'resnet18':
                weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
                resnet = tv_models.resnet18(weights=weights)
            elif backbone == 'resnet34':
                weights = tv_models.ResNet34_Weights.IMAGENET1K_V1
                resnet = tv_models.resnet34(weights=weights)
            else:
                weights = tv_models.ResNet50_Weights.IMAGENET1K_V1
                resnet = tv_models.resnet50(weights=weights)
            blocks = [
                resnet.conv1, resnet.bn1, nn.ReLU(inplace=True), resnet.maxpool,
                resnet.layer1,  # l1
                resnet.layer2,  # l2
                resnet.layer3,  # l3
                resnet.layer4,  # l4
            ]
            cut_idx = 6 if layer == 'l2' else (7 if layer == 'l3' else 8)
            self.backbone = nn.Sequential(*blocks[:cut_idx]).eval()
            for p in self.backbone.parameters(): p.requires_grad = False
            self.crit = nn.MSELoss()

        elif kind == 'mobilenetv3':
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
            else:  # 'mv3_last'
                cut = len(feats)
            self.backbone = nn.Sequential(*feats[:cut]).eval()
            for p in self.backbone.parameters(): p.requires_grad = False
            self.crit = nn.MSELoss()

        elif kind == 'lpips':
            import lpips
            self.backbone = lpips.LPIPS(net='alex').eval()
            for p in self.backbone.parameters(): p.requires_grad = False
        else:
            self.backbone = None

    def _imnet_norm(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, x_fake, x_real):
        if self.backbone is None:
            return torch.zeros((), device=x_fake.device)

        if self.kind in ['vgg', 'resnet', 'mobilenetv3']:
            xf = self._imnet_norm(x_fake)
            xr = self._imnet_norm(x_real).detach()
            bb = self.backbone.to(xf.device)
            return self.crit(bb(xf), bb(xr))
        else:  # lpips 期望 [-1,1]
            xf = x_fake * 2 - 1
            xr = (x_real * 2 - 1).detach()
            return self.backbone.to(xf.device)(xf, xr).mean()


logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    return logloss(d.unsqueeze(1), y)

def make_dataloader(dataset_dir, batch_size):
    dataset = MyDataset(dataset_dir, args.asr)
    sampler = None
    if args.weighted_sampling:
        wpath_hard = os.path.join(dataset_dir, "weights_hard.npy")
        wpath = wpath_hard if os.path.isfile(wpath_hard) else os.path.join(dataset_dir, "weights.npy")
        if os.path.isfile(wpath):
            w = np.load(wpath)
            if len(w) < len(dataset):
                rep = int(np.ceil(len(dataset) / max(1, len(w))))
                w = np.tile(w, rep)[:len(dataset)]
            else:
                w = w[:len(dataset)]
            w = torch.from_numpy(w.astype(np.float32))
            sampler = WeightedRandomSampler(w, num_samples=len(dataset), replacement=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True
    )
    return dataset, loader

def train(net, epoch, batch_size, lr):
    content_loss = Perceptual(kind=args.perceptual,
                              backbone=args.perc_backbone,
                              layer=args.perc_layer).to(device)
    if use_syncnet:
        if args.syncnet_checkpoint == "":
            raise ValueError("Using syncnet, you need to set 'syncnet_checkpoint'. Please check README")
        syncnet = SyncNet_color(args.asr).eval().to(device)
        try:
            state = torch.load(args.syncnet_checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(args.syncnet_checkpoint, map_location=device)
        syncnet.load_state_dict(state)

    os.makedirs(args.save_dir, exist_ok=True)

    dataset_dir_list = [args.dataset_dir]
    dataset_list, dataloader_list = [], []
    for dataset_dir in dataset_dir_list:
        dataset, loader = make_dataloader(dataset_dir, batch_size)
        dataset_list.append(dataset)
        dataloader_list.append(loader)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.L1Loss()
    for e in range(epoch):
        net.train()
        random_i = random.randint(0, len(dataset_dir_list) - 1)
        dataset = dataset_list[random_i]
        train_dataloader = dataloader_list[random_i]

        with tqdm(total=len(dataset), desc=f'Epoch {e + 1}/{epoch}', unit='img') as p:
            for batch in train_dataloader:
                imgs, labels, audio_feat = batch
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                audio_feat = audio_feat.to(device, non_blocking=True)

                preds = net(imgs, audio_feat)
                # SyncNet
                if use_syncnet:
                    if args.asr == "hubert":
                        if audio_feat.shape[1] == 16:
                            audio_feat_sync = torch.cat([audio_feat, audio_feat], dim=1)
                        else:
                            C = audio_feat.shape[1]
                            if 32 % C == 0:
                                audio_feat_sync = audio_feat.repeat(1, 32 // C, 1, 1)
                            elif C > 32:
                                audio_feat_sync = audio_feat[:, :32, :, :]
                            else:
                                need = 32 - C
                                audio_feat_sync = torch.cat([audio_feat, audio_feat[:, :need, :, :]], dim=1)
                    else:
                        audio_feat_sync = audio_feat
                    y = torch.ones([preds.shape[0], 1], device=device, dtype=torch.float32)
                    a, v = syncnet(preds, audio_feat_sync)
                    sync_loss = cosine_loss(a, v, y)

                # 损失
                loss_perc = content_loss(preds, labels)
                loss_pixel = criterion(preds, labels)
                loss = loss_pixel + 0.01 * loss_perc + (args.sync_w * sync_loss if use_syncnet else 0.0)

                p.set_postfix(**{'loss (batch)': float(loss.item())})
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                p.update(imgs.shape[0])

        if e % 5 == 0:
            torch.save(net.state_dict(), os.path.join(args.save_dir, f'{e}.pth'))

        if args.see_res:
            net.eval()
            idx = random.randint(0, len(dataset) - 1)
            img_concat_T, img_real_T, audio_feat = dataset.__getitem__(idx)
            img_concat_T = img_concat_T[None].to(device)
            audio_feat = audio_feat[None].to(device)
            with torch.no_grad():
                pred = net(img_concat_T, audio_feat)[0]
            pred = (pred.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img_real = (img_real_T.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            os.makedirs("./train_tmp_img", exist_ok=True)
            cv2.imwrite(f"./train_tmp_img/epoch_{e}.jpg", pred)
            cv2.imwrite(f"./train_tmp_img/epoch_{e}_real.jpg", img_real)



if __name__ == '__main__':
    net = Model(6, args.asr).to(device)
    # #UNet-Plus
    # from unet_plus import Model
    # net = Model(6, args.asr,
    #             use_cross_attn=args.cross_attn,
    #             upsample_mode=args.upsample,
    #             width_mult=args.width_mult).to(device)

    train(net, args.epochs, args.batchsize, args.lr)
