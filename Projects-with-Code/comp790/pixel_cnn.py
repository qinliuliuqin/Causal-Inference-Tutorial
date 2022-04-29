import numpy as np
import os
import torch
import torch.nn.functional as F
from vq_vae import DAVIS
from vq_vae import Model as Model_VQ_VAE
from utils import GatedPixelCNN
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


# Parameters for VQ-VQE
device = torch.device("cuda:3")
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 512
commitment_cost = 0.25
decay = 0.99

vq_vae = Model_VQ_VAE(num_hiddens, num_residual_layers, num_residual_hiddens,
                      num_embeddings, embedding_dim,
                      commitment_cost, decay).to(device)
vq_vae.load_state_dict(torch.load('out/model_46k.pt'))
vq_vae.eval()


# Parameters for Dataset
num_frames = 12  # each video uses 10 consecutive frames for training
davis_root = '/playpen-raid2/qinliu/data/DAVIS'

trainset = DAVIS(root=davis_root, num_frames=num_frames, train=True)
valset = DAVIS(root=davis_root, num_frames=num_frames, train=False)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                           shuffle=True, num_workers=1)

val_loader = torch.utils.data.DataLoader(valset, batch_size=1,
                                         shuffle=True, num_workers=1)

# Train GatedPixelCNN to learn the prior of latent code indices
num_training_updates = 10000


model = GatedPixelCNN(num_embeddings, 128, num_embeddings)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# train pixelcnn
model_save_dir = 'checkpoints/pixel_cnn'
print_freq = 1
save_freq = 100
eval_freq = 100
for i in range(1, num_training_updates):
    inputs, _ = next(iter(train_loader))
    inputs = torch.squeeze(inputs, dim=0)
    inputs = inputs.to(device)
    inputs_shape = inputs.shape

    indices = vq_vae.get_code_indices(inputs)
    indices = indices.view(
        num_frames, inputs_shape[2] // 4, inputs_shape[3] // 4)

    indices = indices.to(device)
    one_hot_indices = F.one_hot(
        indices, num_embeddings).float().permute(0, 3, 1, 2).contiguous()

    outputs = model(one_hot_indices)

    loss = F.cross_entropy(outputs, indices)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % print_freq == 0:
        print("\t [{}/{}]: loss {}".format(i, num_training_updates, loss.item()))

    if i % save_freq == 0:

        if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

        print("save the model at iteration: ", i)
        torch.save(model.state_dict(), os.path.join(model_save_dir, 'pixel_cnn_model.pt'.format(i)))

    if i % save_freq == 0:
        model.eval()

        n_samples = 1
        prior_size = (64, 128) # h, w
        priors = torch.zeros((n_samples,) + prior_size, dtype=torch.long).to(device)

        # Iterate over the priors because generation has to be done sequentially pixel by pixel.
        for row in range(prior_size[0]):
            for col in range(prior_size[1]):
                # Feed the whole array and retrieving the pixel value probabilities for the next
                # pixel.
                with torch.inference_mode():
                    one_hot_priors = F.one_hot(priors, num_embeddings).float().permute(0, 3, 1, 2).contiguous()
                    logits = model(one_hot_priors)
                    probs = F.softmax(logits[:, :, row, col], dim=-1)
                    # Use the probabilities to pick pixel values and append the values to the priors.
                    priors[:, row, col] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)

        # Perform an embedding lookup and Generate new images
        with torch.inference_mode():
            z = vq_vae._vq_vae.quantize(priors)
            z = z.permute(0, 3, 1, 2).contiguous()
            pred = vq_vae._decoder(z)

        grid_img = make_grid(pred.cpu().data, nrow=1)
        grid_img = grid_img.numpy()

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(grid_img, (1, 2, 0)), interpolation='nearest')

        plt.savefig(f'{model_save_dir}/{i}.png', bbox_inches='tight')
        plt.close(fig)

        model.train()
