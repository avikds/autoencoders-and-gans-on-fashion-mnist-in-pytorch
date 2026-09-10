"""
Autoencoders and GANs on Fashion-MNIST in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_fashion_pixels
import os
import gzip
import tempfile
import urllib.request
import numpy as np
import torch

def load_fashion_pixels(n_train=5000, n_test=1000):
    # Fashion-MNIST IDX files
    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    # Cache directory
    cache_dir = tempfile.gettempdir()

    # Download each file only once
    paths = {}
    for key, filename in files.items():
        cached_path = os.path.join(cache_dir, f"fashion_{filename}")
        paths[key] = cached_path

        if not os.path.exists(cached_path):
            urllib.request.urlretrieve(
                base_url + filename,
                cached_path
            )

    # Read and parse IDX image files
    with gzip.open(paths["train_images"], "rb") as f:
        train_images_data = f.read()

    with gzip.open(paths["test_images"], "rb") as f:
        test_images_data = f.read()

    # IDX image header is 16 bytes
    train_images = np.frombuffer(
        train_images_data,
        dtype=np.uint8,
        offset=16
    ).reshape(-1, 28, 28)

    test_images = np.frombuffer(
        test_images_data,
        dtype=np.uint8,
        offset=16
    ).reshape(-1, 28, 28)

    # Read and parse IDX label files
    with gzip.open(paths["train_labels"], "rb") as f:
        train_labels_data = f.read()

    with gzip.open(paths["test_labels"], "rb") as f:
        test_labels_data = f.read()

    # IDX label header is 8 bytes
    train_labels = np.frombuffer(
        train_labels_data,
        dtype=np.uint8,
        offset=8
    )

    test_labels = np.frombuffer(
        test_labels_data,
        dtype=np.uint8,
        offset=8
    )

    # Limit to requested sample counts
    train_images = train_images[:n_train]
    test_images = test_images[:n_test]
    train_labels = train_labels[:n_train]
    test_labels = test_labels[:n_test]

    # Flatten 28x28 images -> 784-dimensional vectors
    # Convert to float32 and scale pixels from [0, 255] -> [0, 1]
    X_train = torch.from_numpy(
        train_images.reshape(-1, 784).copy()
    ).to(torch.float32) / 255.0

    X_test = torch.from_numpy(
        test_images.reshape(-1, 784).copy()
    ).to(torch.float32) / 255.0

    # Labels as int64 tensors
    y_train = torch.from_numpy(train_labels.astype(np.int64))
    y_test = torch.from_numpy(test_labels.astype(np.int64))

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

# Step 2 - pca_reconstruction_error
def pca_reconstruction_error(X, n_components):
    # Convert PyTorch tensors to NumPy arrays when necessary
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    X = np.asarray(X, dtype=np.float64)

    # Center the data
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean

    # SVD of the centered data
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)

    # Top right-singular vectors
    components = Vt[:n_components]

    # Project onto the principal components
    Z = Xc @ components.T

    # Reconstruct and add the original mean back
    X_reconstructed = Z @ components + mean

    # Mean squared error per pixel over all entries
    mse = np.mean((X - X_reconstructed) ** 2)

    return float(mse)

# Step 3 - LinearAutoencoder
class LinearAutoencoder(nn.Module):
    def __init__(self, n_inputs=784, n_codings=30):
        super().__init__()

        self.encoder = nn.Linear(n_inputs, n_codings)
        self.decoder = nn.Linear(n_codings, n_inputs)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        codings = self.encode(x)
        return self.decoder(codings)

# Step 4 - train_autoencoder
import torch.nn as nn
import torch.nn.functional as F

def train_autoencoder(model, X, epochs=5, lr=0.005, batch_size=64, seed=42, noise_std=0.0):
    # Set the random seed for reproducible shuffling and noise
    torch.manual_seed(seed)

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    losses = []

    n_samples = X.shape[0]

    for _ in range(epochs):
        # Shuffle the samples for each epoch
        indices = torch.randperm(n_samples, device=X.device)

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            batch_indices = indices[start:start + batch_size]
            target = X[batch_indices]

            # Add Gaussian noise only when requested
            if noise_std != 0.0:
                inp = target + noise_std * torch.randn_like(target)
            else:
                inp = target

            optimizer.zero_grad()

            reconstruction = model(inp)
            loss = F.mse_loss(reconstruction, target)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Mean loss across batches for this epoch
        losses.append(epoch_loss / n_batches)

    return losses

def reconstruction_error(model, X):
    # Preserve the model's original training/evaluation state
    was_training = model.training

    model.eval()

    with torch.no_grad():
        reconstruction = model(X)
        mse = F.mse_loss(reconstruction, X).item()

    # Restore original state
    if was_training:
        model.train()

    return float(mse)

# Step 5 - StackedAutoencoder
class StackedAutoencoder(nn.Module):
    def __init__(self, n_inputs=784, hidden=100, n_codings=30):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_codings)
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_codings, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_inputs),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        codings = self.encode(x)
        return self.decoder(codings)

# Step 6 - TiedAutoencoder
class TiedAutoencoder(nn.Module):
    def __init__(self, n_inputs=784, hidden=100, n_codings=30):
        super().__init__()

        # Encoder parameters
        self.W1 = nn.Parameter(torch.empty(hidden, n_inputs))
        self.b1 = nn.Parameter(torch.zeros(hidden))

        self.W2 = nn.Parameter(torch.empty(n_codings, hidden))
        self.b2 = nn.Parameter(torch.zeros(n_codings))

        # Decoder-only biases
        self.b3 = nn.Parameter(torch.zeros(hidden))
        self.b4 = nn.Parameter(torch.zeros(n_inputs))

        # Initialize weights like nn.Linear
        nn.init.kaiming_uniform_(self.W1, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.W2, a=5 ** 0.5)

    def encode(self, x):
        hidden_representation = torch.relu(x @ self.W1.T + self.b1)
        return hidden_representation @ self.W2.T + self.b2

    def forward(self, x):
        codings = self.encode(x)

        hidden_representation = torch.relu(codings @ self.W2 + self.b3)
        reconstruction = hidden_representation @ self.W1 + self.b4

        return torch.sigmoid(reconstruction)

# Step 7 - denoising_gain
def denoising_gain(model, X, noise_std=0.3, seed=0):
    # Seed noise generation for reproducibility
    torch.manual_seed(seed)

    # Add Gaussian noise and clip pixel values to [0, 1]
    noisy = torch.clamp(
        X + noise_std * torch.randn_like(X),
        0.0,
        1.0
    )

    # Preserve the model's original state
    was_training = model.training
    model.eval()

    with torch.no_grad():
        noisy_mse = F.mse_loss(noisy, X)
        reconstruction_mse = F.mse_loss(model(noisy), X)

    # Restore the original training/evaluation state
    if was_training:
        model.train()

    return float((noisy_mse / reconstruction_mse).item())

# Step 8 - kl_sparsity_loss
def kl_sparsity_loss(activations, target=0.1):
    # Mean activation of each hidden unit across the batch
    p = activations.mean(dim=0)

    # Avoid log(0) and division by zero
    p = torch.clamp(p, 1e-6, 1.0 - 1e-6)

    # KL divergence: KL(target || p)
    loss = (
        target * torch.log(target / p)
        + (1.0 - target) * torch.log((1.0 - target) / (1.0 - p))
    )

    # Sum over hidden units
    return loss.sum()

class SparseAutoencoder(nn.Module):
    def __init__(self, n_inputs=784, hidden=300):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden, n_inputs),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        codings = self.encode(x)
        return self.decoder(codings)

# Step 9 - train_sparse
def train_sparse(model, X, weight=0.1, target=0.1, epochs=5, lr=0.005, batch_size=64, seed=42):
    # Seed shuffling for reproducibility
    torch.manual_seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    losses = []

    n_samples = X.shape[0]

    for _ in range(epochs):
        # Shuffle the training data for each epoch
        indices = torch.randperm(n_samples, device=X.device)

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            batch_indices = indices[start:start + batch_size]
            batch = X[batch_indices]

            optimizer.zero_grad()

            # Forward pass
            reconstruction = model(batch)
            codings = model.encode(batch)

            # Reconstruction loss
            reconstruction_loss = F.mse_loss(reconstruction, batch)

            # Sparsity penalty
            sparsity_loss = kl_sparsity_loss(codings, target)

            # Total loss
            loss = reconstruction_loss + weight * sparsity_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Mean total loss across batches
        losses.append(epoch_loss / n_batches)

    return losses

def mean_activation(model, X):
    # Preserve the model's original state
    was_training = model.training
    model.eval()

    with torch.no_grad():
        mean_act = model.encode(X).mean().item()

    # Restore original state
    if was_training:
        model.train()

    return float(mean_act)

# Step 10 - codings_classifier
def codings_classifier(autoencoder, data, epochs=20, lr=0.01, seed=42):
    # Seed classifier initialization and training
    torch.manual_seed(seed)

    # Extract codings without tracking gradients
    autoencoder.eval()

    with torch.no_grad():
        X_train_codings = autoencoder.encode(data["X_train"])
        X_test_codings = autoencoder.encode(data["X_test"])

    y_train = data["y_train"]
    y_test = data["y_test"]

    # Linear classifier: n_codings -> 10 classes
    n_codings = X_train_codings.shape[1]
    classifier = nn.Linear(n_codings, 10)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    # Full-batch training
    classifier.train()

    for _ in range(epochs):
        optimizer.zero_grad()

        logits = classifier(X_train_codings)
        loss = F.cross_entropy(logits, y_train)

        loss.backward()
        optimizer.step()

    # Evaluate on test codings
    classifier.eval()

    with torch.no_grad():
        logits = classifier(X_test_codings)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y_test).float().mean().item()

    return float(accuracy)

# Step 11 - VAE
class VAE(nn.Module):
    def __init__(self, n_inputs=784, hidden=100, n_codings=10):
        super().__init__()

        self.n_codings = n_codings

        self.hidden = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU()
        )

        self.mu = nn.Linear(hidden, n_codings)
        self.logvar = nn.Linear(hidden, n_codings)

        self.decoder = nn.Sequential(
            nn.Linear(n_codings, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_inputs),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.hidden(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# Step 12 - vae_loss
def kl_divergence(mu, logvar):
    # KL divergence from N(mu, exp(logvar)) to N(0, I)
    kl = -0.5 * torch.sum(
        1.0 + logvar - mu.pow(2) - torch.exp(logvar),
        dim=1
    )

    # Mean over the batch
    return kl.mean()

def vae_loss(recon, x, mu, logvar):
    batch_size = x.shape[0]

    # Reconstruction loss: summed BCE for each image,
    # then averaged over the batch
    reconstruction_loss = (
        F.binary_cross_entropy(
            recon,
            x,
            reduction="sum"
        ) / batch_size
    )

    # Add KL divergence term
    return reconstruction_loss + kl_divergence(mu, logvar)

# Step 13 - train_vae
def train_vae(model, X, epochs=10, lr=0.002, batch_size=64, seed=42):
    # Seed shuffling and VAE sampling for reproducibility
    torch.manual_seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    losses = []

    n_samples = X.shape[0]

    for _ in range(epochs):
        # Shuffle samples for each epoch
        indices = torch.randperm(n_samples, device=X.device)

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            batch_indices = indices[start:start + batch_size]
            batch = X[batch_indices]

            optimizer.zero_grad()

            # Forward pass through the VAE
            recon, mu, logvar = model(batch)

            # ELBO loss
            loss = vae_loss(recon, batch, mu, logvar)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Mean loss over batches in the epoch
        losses.append(epoch_loss / n_batches)

    return losses

# Step 14 - generate_images
def generate_images(model, n, seed=0):
    # Seed latent-code sampling for reproducibility
    torch.manual_seed(seed)

    # Sample from the standard normal distribution
    z = torch.randn(n, model.n_codings)

    # Preserve the model's original state
    was_training = model.training
    model.eval()

    # Decode without tracking gradients
    with torch.no_grad():
        images = model.decode(z)

    # Restore the original state
    if was_training:
        model.train()

    return images

def interpolate_codings(z1, z2, steps):
    # Generate evenly spaced interpolation factors from 0 to 1
    t = torch.linspace(
        0.0,
        1.0,
        steps,
        device=z1.device,
        dtype=z1.dtype
    ).unsqueeze(1)

    # Linear interpolation from z1 to z2, inclusive
    return (1.0 - t) * z1.unsqueeze(0) + t * z2.unsqueeze(0)

# Step 15 - Generator
class Generator(nn.Module):
    def __init__(self, n_codings=30, hidden=100, n_outputs=784):
        super().__init__()

        self.n_codings = n_codings

        self.net = nn.Sequential(
            nn.Linear(n_codings, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_outputs),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, n_inputs=784, hidden=100):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# Step 16 - gan_step
def gan_step(G, D, real, opt_g, opt_d):
    batch_size = real.shape[0]

    # -------------------------
    # Discriminator phase
    # -------------------------
    z = torch.randn(batch_size, G.n_codings)
    fake = G(z)

    real_logits = D(real)
    fake_logits = D(fake.detach())

    real_targets = torch.ones_like(real_logits)
    fake_targets = torch.zeros_like(fake_logits)

    d_loss = (
        F.binary_cross_entropy_with_logits(real_logits, real_targets)
        + F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
    )

    opt_d.zero_grad()
    d_loss.backward()
    opt_d.step()

    # -------------------------
    # Generator phase
    # -------------------------
    fake_logits_for_g = D(fake)

    generator_targets = torch.ones_like(fake_logits_for_g)

    g_loss = F.binary_cross_entropy_with_logits(
        fake_logits_for_g,
        generator_targets
    )

    opt_g.zero_grad()
    g_loss.backward()
    opt_g.step()

    return float(d_loss.item()), float(g_loss.item())

# Step 17 - train_gan (not yet solved)
# TODO: implement

# Step 18 - save_vae (not yet solved)
# TODO: implement

