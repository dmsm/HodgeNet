import argparse
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hodgenet import HodgeNetModel
from meshdata import OrigamiDataset


def main(args):
    torch.set_default_dtype(torch.float64)  # needed for eigenvalue problems
    torch.manual_seed(0)
    np.random.seed(0)

    dataset = OrigamiDataset(
        edge_features_from_vertex_features=['vertices'],
        triangle_features_from_vertex_features=['vertices'])

    def mycollate(b): return b
    dataloader = DataLoader(dataset, batch_size=args.bs,
                            num_workers=0, collate_fn=mycollate)

    example = dataset[0]
    hodgenet_model = HodgeNetModel(
        example['int_edge_features'].shape[1],
        example['triangle_features'].shape[1],
        num_output_features=args.n_out_features, mesh_feature=True,
        num_eigenvectors=args.n_eig, num_extra_eigenvectors=args.n_extra_eig,
        resample_to_triangles=False,
        num_bdry_edge_features=example['bdry_edge_features'].shape[1],
        num_vector_dimensions=args.num_vector_dimensions)

    origami_model = nn.Sequential(
        hodgenet_model,
        nn.Linear(args.n_out_features*args.num_vector_dimensions *
                  args.num_vector_dimensions, 32),
        nn.LayerNorm(32),
        nn.LeakyReLU(),
        nn.Linear(32, 16),
        nn.LayerNorm(16),
        nn.LeakyReLU(),
        nn.Linear(16, 2))

    optimizer = optim.AdamW(origami_model.parameters(), lr=args.lr)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    train_writer = SummaryWriter(os.path.join(
        args.out, datetime.datetime.now().strftime('train-%m%d%y-%H%M%S')),
                                 flush_secs=1)

    def epoch_loop(dataloader, epochname, epochnum, writer, optimize=True):
        epoch_loss, epoch_size = 0, 0
        pbar = tqdm(total=len(dataloader),
                    desc='{} {}'.format(epochname, epochnum))
        for batchnum, batch in enumerate(dataloader):
            if optimize:
                optimizer.zero_grad()

            batch_loss = 0

            dirs = origami_model(batch)
            dirs = F.normalize(dirs, p=2, dim=-1)
            for mesh, dir_estimate in zip(batch, dirs):
                gt_dir = mesh['dir'].to(dir_estimate.device)
                batch_loss += 1 - (gt_dir * dir_estimate).sum(-1)

            batch_loss /= len(batch)

            pbar.set_postfix({
                'loss': batch_loss.item(),
            })
            pbar.update(1)

            epoch_loss += batch_loss.item()
            epoch_size += 1

            if optimize:
                batch_loss.backward()
                nn.utils.clip_grad_norm_(origami_model.parameters(), 1)
                optimizer.step()

            writer.add_scalar('Loss', batch_loss.item(),
                              epochnum*len(dataloader)+batchnum)

        pbar.close()

    for epoch in range(args.n_epochs):
        origami_model.train()
        epoch_loop(dataloader, 'Epoch', epoch, train_writer)

        torch.save({
            'origami_model_state_dict': origami_model.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, os.path.join(args.out, f'{epoch}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='out/origami')
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--n_eig', type=int, default=32)
    parser.add_argument('--n_extra_eig', type=int, default=32)
    parser.add_argument('--n_out_features', type=int, default=32)
    parser.add_argument('--num_vector_dimensions', type=int, default=4)

    args = parser.parse_args()
    main(args)
