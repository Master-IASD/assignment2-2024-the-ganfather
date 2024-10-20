import torch
import os

# Specify the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def D_train(x, G, D, D_optimizer, criterion):
    #======================= Train the discriminator =======================#
    D.zero_grad()

    # Train discriminator on real data
    x_real = x.to(device)
    y_real = torch.ones(x_real.shape[0], 1, device=device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # Train discriminator on fake data
    z = torch.randn(x_real.shape[0], 100, device=device)  # Latent vector
    x_fake = G(z)
    y_fake = torch.zeros(x_real.shape[0], 1, device=device)

    D_output = D(x_fake)

    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # Gradient backpropagation & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return D_loss.item(), D_real_score, D_fake_score  # Return scores for analysis


def G_train(x, G, D, G_optimizer, criterion):
    #======================= Train the generator =======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100, device=device)  # Latent vector
    y = torch.ones(z.shape[0], 1, device=device)  # Target for generator

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # Gradient backpropagation & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.item(), D_output  # Return D_output for analysis


def save_models(G, D, folder):
    os.makedirs(folder, exist_ok=True)  # Ensure the folder exists
    torch.save(G.state_dict(), os.path.join(folder, 'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder, 'D.pth'))


def load_model(model, folder, is_generator=True):
    model_type = 'G' if is_generator else 'D'
    ckpt = torch.load(os.path.join(folder, f'{model_type}.pth'), map_location=device)  # Load model to the correct device
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return model