import os
import torch
import numpy as np
from torch import nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torchvision.utils import make_grid
from vqvae import DAVIS, Model
from convlstm import ConvLSTM, Config
# import pdb

option = 1

device = torch.device("cuda:0")
# num_training_updates = 500000
num_training_updates = 20000
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 512
commitment_cost = 0.25
decay = 0.99
learning_rate = 1e-3

model_param_path = "./model_46k.pt"
model = Model(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay).to(device)
model.load_state_dict(torch.load(model_param_path, map_location='cuda:0'))

model_encoder = nn.Sequential(model._encoder,
                              model._pre_vq_conv)
model_vqvae = model._vq_vae
model_decoder = model._decoder

config = Config()
model_lstm = ConvLSTM(config).to(device)
optimizer = optim.Adam(model_lstm.parameters(), lr=learning_rate, amsgrad=False)

## Dataset Parameters
mb_size = 1  # each batch use only one video
num_frames = 12  # each video uses 10 consecutive frames for training
davis_root = '/playpen-raid2/qinliu/data/DAVIS'

trainset = DAVIS(root=davis_root, num_frames=num_frames, train=True)
valset = DAVIS(root=davis_root, num_frames=num_frames, train=False)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=mb_size,
                                           shuffle=True, num_workers=1)

val_loader = torch.utils.data.DataLoader(valset, batch_size=mb_size,
                                         shuffle=True, num_workers=1)

train_res_recon_error = []
# train_res_perplexity = []

model_lstm.train()
fig_idx = 0
for i in range(num_training_updates):
    data, _ = next(iter(train_loader))
    data = torch.squeeze(data, dim=0)
    data = data.to(device)
    optimizer.zero_grad()

    if option == 1:
        z = model_encoder(data)
        z = z.unsqueeze(dim=0)
        z_pred = model_lstm(z)
    elif option == 2:
        z = model_encoder(data)
        _, z, _, _ = model_vqvae(z)
        z = z.unsqueeze(dim=0)
        z_pred = model_lstm(z)
        # pdb.set_trace()
        z_pred = (z_pred * (num_embeddings/2)).int() / (num_embeddings/2)
    recon_error = F.mse_loss(z_pred[:, :-1, ...], z[:, 1:, ...])
    loss = recon_error #+ vq_loss
    loss.backward()

    optimizer.step()

    train_res_recon_error.append(recon_error.item())
    # train_res_perplexity.append(perplexity.item())

    if (i + 1) % 20 == 0:
        print('%d iterations' % (i + 1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        # print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))

    if (i + 1) % 1000 == 0:
        model.eval()

        val_data, _ = next(iter(val_loader))
        val_data = torch.squeeze(val_data, dim=0)
        val_data = val_data.to(device)

        val_z = model_encoder(val_data)
        if option == 1:
            val_z = val_z.unsqueeze(dim=0)
            val_z_pred = model_lstm(val_z)
            val_z_pred = torch.squeeze(val_z_pred, dim=0)
            _, val_z_pred, _, _ = model_vqvae(val_z_pred)
            # pdb.set_trace()
        else:
            _, val_z, _, _ = model_vqvae(val_z)
            val_z = val_z.unsqueeze(dim=0)
            val_z_pred = model_lstm(val_z)
            val_z_pred = torch.squeeze(val_z_pred, dim=0)
            val_z_pred = (val_z_pred * (num_embeddings/2)).int() / (num_embeddings/2)
        val_recon = model_decoder(val_z_pred)

        grid_img = make_grid(val_recon.cpu().data, nrow=4)
        grid_img = grid_img.numpy()

        fig = plt.figure(figsize=(16, 12))
        plt.imshow(np.transpose(grid_img, (1, 2, 0)), interpolation='nearest')

        if not os.path.exists('./out_lstm/'):
            os.makedirs('./out_lstm/')

        plt.savefig('./out_lstm/{}.png'.format(str(fig_idx).zfill(3)), bbox_inches='tight')
        fig_idx += 1
        plt.close(fig)

        # save the model
        torch.save(model_lstm.state_dict(), './out_lstm/model_lstm.pt')

        model.train()
