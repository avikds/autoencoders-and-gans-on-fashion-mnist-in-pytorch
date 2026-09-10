"""
Autoencoders and GANs on Fashion-MNIST in PyTorch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Autoencoders and GANs on Fashion-MNIST (Hands-On ML, chapter 18).

Story: compress real images with a linear autoencoder (checked against PCA) and a
stacked one; tie weights, denoise, sparsify, reuse the codings for a classifier;
then generate new garments with a variational autoencoder and a GAN, and save the VAE.
"""
import os
import tempfile
import numpy as np
import torch


def main() -> None:
    d = load_fashion_pixels(n_train=5000, n_test=1000)
    X, Xt = d["X_train"], d["X_test"]

    # ---- 1. Compress ----
    pca30 = pca_reconstruction_error(X, 30)
    torch.manual_seed(0)
    linear = LinearAutoencoder(784, 30)
    train_autoencoder(linear, X, epochs=10)
    torch.manual_seed(0)
    stacked = StackedAutoencoder(784, 100, 30)
    train_autoencoder(stacked, X, epochs=10)
    print(f"reconstruction MSE with 30 codings: PCA {pca30:.4f}   linear AE {reconstruction_error(linear, Xt):.4f}   stacked AE {reconstruction_error(stacked, Xt):.4f}")

    # ---- 2. Regularize and reuse ----
    torch.manual_seed(0)
    tied = TiedAutoencoder(784, 100, 30)
    train_autoencoder(tied, X, epochs=10)
    n_tied = sum(p.numel() for p in tied.parameters())
    n_stacked = sum(p.numel() for p in stacked.parameters())
    print(f"tied weights: {n_tied:,} parameters vs {n_stacked:,}, MSE {reconstruction_error(tied, Xt):.4f}")
    torch.manual_seed(0)
    denoiser = StackedAutoencoder(784, 100, 30)
    train_autoencoder(denoiser, X, epochs=8, noise_std=0.3)
    print(f"denoising: reconstruction is {denoising_gain(denoiser, Xt):.1f}x closer to the clean image than the noisy input")
    torch.manual_seed(0)
    sparse = SparseAutoencoder(784, 300)
    before = mean_activation(sparse, Xt)
    train_sparse(sparse, X, epochs=5)
    print(f"sparsity: mean coding activation {before:.2f} -> {mean_activation(sparse, Xt):.2f} (target 0.10)")
    print(f"linear classifier on the stacked AE's 30 codings: {codings_classifier(stacked, d, epochs=100):.3f} test accuracy")

    # ---- 3. VAE ----
    torch.manual_seed(0)
    vae = VAE(784, 100, 10)
    losses = train_vae(vae, X, epochs=10)
    samples = generate_images(vae, 16, seed=1)
    print(f"\nVAE: loss {losses[0]:.0f} -> {losses[-1]:.0f} nats/image; 16 samples with mean pixel {float(samples.mean()):.2f}, spread across samples {float(samples.std(0).mean()):.3f}")
    with torch.no_grad():
        mu, _ = vae.encode(Xt[:2])
    path_z = interpolate_codings(mu[0], mu[1], 8)
    morph = vae.decode(path_z)
    print(f"interpolating between two test garments: {morph.shape[0]} frames of {morph.shape[1]} pixels")

    # ---- 4. GAN ----
    torch.manual_seed(0)
    G, D = Generator(30, 100, 784), Discriminator(784, 100)
    h = train_gan(G, D, X, epochs=1)
    torch.manual_seed(1)
    with torch.no_grad():
        fakes = G(torch.randn(500, 30))
    print(f"GAN after one epoch: D loss {h['d_loss'][-1]:.2f}, G loss {h['g_loss'][-1]:.2f}, discriminator accuracy {discriminator_accuracy(D, Xt[:500], fakes):.2f}")

    # ---- 5. Ship the VAE ----
    path = os.path.join(tempfile.gettempdir(), "fashion_vae.pt")
    save_vae(vae, {"n_inputs": 784, "hidden": 100, "n_codings": 10}, path)
    reloaded = load_vae(path)
    same = bool(torch.equal(generate_images(reloaded, 4, seed=7), generate_images(vae, 4, seed=7)))
    print(f"saved and reloaded VAE generates identical samples: {same}")


if __name__ == "__main__":
    main()

