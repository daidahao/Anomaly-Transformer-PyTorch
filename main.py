import argparse
import time
import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm

from data import Dataset
from model import AnomalyTransformer, symmetrical_kl_divergence


def train(model: AnomalyTransformer, dataloader: DataLoader, \
    learning_rate: float, k: float, n_epochs: int, device: str, \
    save_path: str):
    
    train_start = time.time()

    # Initialise optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss(reduction='mean')

    for epoch in range(n_epochs):
        epoch_start = time.time()
        print(f'Epoch {epoch + 1}/{n_epochs}')
        loss1_list, loss2_list = [], []
        recon_list = []

        for data, _ in tqdm(dataloader):
            # shape: (batch_size, window_size, n_features)
            data = data.to(device, dtype=torch.float32)
            # Forward pass
            optimizer.zero_grad()
            output, series_list, prior_list = model(data)
            # Compute loss
            series_losses, prior_losses = [], []
            for series, prior in zip(series_list, prior_list):
                prior = prior / prior.sum(-1, keepdim=True)
                series_loss = symmetrical_kl_divergence(prior.detach(), series)
                prior_loss = symmetrical_kl_divergence(series.detach(), prior)
                series_losses.append(series_loss); prior_losses.append(prior_loss)
            series_loss = torch.mean(torch.stack(series_losses))
            prior_loss = torch.mean(torch.stack(prior_losses))
            recon_loss = mse(output, data)
            loss1 = recon_loss - k * series_loss
            loss2 = recon_loss + k * prior_loss
            loss1_list.append(loss1.item()); loss2_list.append(loss2.item())
            recon_list.append(recon_loss.item())
            # Backward pass
            loss1.backward(retain_graph=True)
            loss2.backward()
            # Update parameters
            optimizer.step()
        print(f'Epoch {epoch + 1}/{n_epochs} took {time.time() - epoch_start:.2f}s')
        print(f'Loss1: {np.average(loss1_list):.4f} | Loss2: {np.average(loss2_list):.4f}')
        print(f'Reconstruction loss: {np.average(recon_list):.4f}')
    print(f'Training took {time.time() - train_start:.2f}s')
    # Save model
    torch.save(model.state_dict(), save_path)

def test(model: AnomalyTransformer, dataloader: DataLoader, \
    device: str, model_path: str, save_path: str):
    # Load model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Initialise optimizer
    mse = nn.MSELoss(reduction='none')

    scores, labels = [], []
    for data, label in tqdm(dataloader):
        # shape: (batch_size, window_size, n_features)
        data = data.to(device, dtype=torch.float32)
        # Forward pass
        output, series_list, prior_list = model(data)
        # Compute loss
        series_losses, prior_losses = [], []
        for series, prior in zip(series_list, prior_list):
            prior = prior / prior.sum(-1, keepdim=True)
            # shape: (batch_size, n_heads, window_size)
            series_loss = symmetrical_kl_divergence(prior.detach(), series)
            prior_loss = symmetrical_kl_divergence(series.detach(), prior)
            series_losses.append(series_loss); prior_losses.append(prior_loss)
        # shape: (batch_size, )
        series_loss = torch.stack(series_losses).mean(dim=[0, 2, 3])
        prior_loss = torch.stack(prior_losses).mean(dim=[0, 2, 3])
        recon_loss = mse(output, data).mean(dim=[1, 2])
        score = torch.sigmoid((-series_loss - prior_loss)/50) * recon_loss
        scores.append(score.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())
    scores, labels = np.concatenate(scores), np.concatenate(labels)
    # save to csv
    df = pd.DataFrame({'score': scores, 'label': labels})
    df.to_csv(save_path, index=False)




if __name__ == '__main__':

    ## Parse arguments
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--data', type=str, default='data/train.csv')
    parser.add_argument('--n_features', type=int, default=51)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--skiprows', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    # model parameters
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.)
    # train flag
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    # train arguments
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--n_epochs', type=int, default=3)
    parser.add_argument('--k', type=float, default=0.3)
    # test arguments
    parser.add_argument('--save_results', type=str, default='models/test.csv')
    # save model
    parser.add_argument('--save', type=str, default='models/model.pt')

    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f'Using device: {device}')

    # Initialise model
    model = AnomalyTransformer(args.n_features, args.window_size, \
        args.d_model, args.n_heads, args.d_ff, args.n_layers, args.dropout)
    model = model.to(device)

    ## Load data
    dataset = Dataset(args.data, args.n_features, args.window_size, \
        stride=args.stride, skiprows=args.skiprows)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=args.train)

    ## Train
    if args.train:
        train(model, dataloader, args.learning_rate, args.k, \
            args.n_epochs, device, args.save)
    elif args.test:
        test(model, dataloader, device, args.save, args.save_results)
