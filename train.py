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
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--asr', type=str, default="hubert", choices=["hubert", "wenet"])
    parser.add_argument('--weighted_sampling', action='store_true')

    # 感知损失
    parser.add_argument('--perceptual', type=str, default='mobilenetv3',
                        choices=['vgg', 'resnet', 'mobilenetv3', 'lpips', 'none'])
    parser.add_argument('--perc_backbone', type=str, default='mbv3_small',
                        choices=['resnet18', 'resnet34', 'resnet50',
                                 'vgg19', 'vgg16',
                                 'mbv3_large', 'mbv3_small'])
    parser.add_argument('--perc_layer', type=str, default='mv3_12',
                        choices=['l2', 'l3', 'l4', 'c32',
                                 'mv3_12', 'mv3_16', 'mv3_last'])
    parser.add_argument('--sync_w', type=float, default=10.0)

    # 新增训练优化参数
    parser.add_argument('--use_scheduler', action='store_true', help='使用学习率调度')
    parser.add_argument('--early_stop', type=int, default=60, help='早停耐心值')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--attention_mode', type=str, default='bottleneck',
                        choices=['none', 'bottleneck', 'all'])
    parser.add_argument('--dropout', type=float, default=0.1)

    # 损失权重
    parser.add_argument('--pixel_weight', type=float, default=1.0)
    parser.add_argument('--perc_weight', type=float, default=0.01)

    return parser.parse_args()


args = get_args()


class Perceptual(nn.Module):
    def __init__(self, kind='mobilenetv3', backbone='mbv3_small', layer='mv3_12'):
        super().__init__()
        self.kind = kind
        self.layer = layer

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

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
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.crit = nn.L1Loss()  # 使用L1损失更稳定
        else:
            self.backbone = None

    def _imnet_norm(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, x_fake, x_real):
        if self.backbone is None:
            return torch.zeros((), device=x_fake.device)

        xf = self._imnet_norm(x_fake)
        xr = self._imnet_norm(x_real).detach()
        return self.crit(self.backbone(xf), self.backbone(xr))


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
    # 损失函数
    content_loss = Perceptual(kind=args.perceptual,
                              backbone=args.perc_backbone,
                              layer=args.perc_layer).to(device)

    if args.use_syncnet:
        if args.syncnet_checkpoint == "":
            raise ValueError("Using syncnet, you need to set 'syncnet_checkpoint'")
        syncnet = SyncNet_color(args.asr).eval().to(device)
        try:
            state = torch.load(args.syncnet_checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(args.syncnet_checkpoint, map_location=device)
        syncnet.load_state_dict(state)

    os.makedirs(args.save_dir, exist_ok=True)

    # 数据加载
    dataset_dir_list = [args.dataset_dir]
    dataset_list, dataloader_list = [], []
    for dataset_dir in dataset_dir_list:
        dataset, loader = make_dataloader(dataset_dir, batch_size)
        dataset_list.append(dataset)
        dataloader_list.append(loader)

    # 优化器
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)  # 使用AdamW
    criterion = nn.L1Loss()

    # 学习率调度器
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epoch, eta_min=lr * 0.01
        )

    # 早停设置
    best_loss = float('inf')
    patience_counter = 0

    # 训练循环
    for e in range(epoch):
        net.train()
        random_i = random.randint(0, len(dataset_dir_list) - 1)
        dataset = dataset_list[random_i]
        train_dataloader = dataloader_list[random_i]

        epoch_loss = 0
        num_batches = 0

        with tqdm(total=len(dataset), desc=f'Epoch {e + 1}/{epoch}', unit='img') as p:
            for batch in train_dataloader:
                imgs, labels, audio_feat = batch
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                audio_feat = audio_feat.to(device, non_blocking=True)

                preds = net(imgs, audio_feat)

                # SyncNet损失
                sync_loss = 0
                if args.use_syncnet:
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

                # 总损失
                loss_perc = content_loss(preds, labels)
                loss_pixel = criterion(preds, labels)

                loss = (args.pixel_weight * loss_pixel +
                        args.perc_weight * loss_perc +
                        (args.sync_w * sync_loss if args.use_syncnet else 0.0))

                p.set_postfix(**{'loss': float(loss.item())})

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # 梯度裁剪
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

                optimizer.step()
                p.update(imgs.shape[0])

                epoch_loss += loss.item()
                num_batches += 1

        # 平均损失
        avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Epoch {e + 1} Average Loss: {avg_loss:.6f}")

        # 学习率调度
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Learning Rate: {current_lr:.6f}")

        # 保存模型
        if e % 5 == 0:
            torch.save(net.state_dict(), os.path.join(args.save_dir, f'epoch_{e}.pth'))

        # 最佳模型保存和早停
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(net.state_dict(), os.path.join(args.save_dir, 'best.pth'))
            print(f"Best model saved with loss: {best_loss:.6f}")
        else:
            patience_counter += 1
            if args.early_stop > 0 and patience_counter >= args.early_stop:
                print(f"Early stopping at epoch {e + 1}")
                break

        # 可视化结果
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

    # 保存最终模型
    torch.save(net.state_dict(), os.path.join(args.save_dir, 'final.pth'))
    print(f"Training completed. Best loss: {best_loss:.6f}")


if __name__ == '__main__':
    # 使用优化的模型
    net = Model(
        n_channels=6,
        mode=args.asr,
        use_attention=args.attention_mode,
        dropout_rate=args.dropout
    ).to(device)

    train(net, args.epochs, args.batchsize, args.lr)