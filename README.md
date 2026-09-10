# Autoencoders and GANs on Fashion-MNIST in PyTorch

Chapter 18 of Hands-On Machine Learning on real images. Compress Fashion-MNIST with a linear autoencoder and check it against PCA, then a stacked autoencoder that beats it; tie weights, denoise, add a sparsity penalty, and reuse the codings for a classifier. Then generate: a variational autoencoder with the reparameterization trick and the ELBO loss, sampling and interpolation in coding space, and a GAN trained with the two-player step. Save the VAE at the end.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_fashion_pixels
- [x] **2.** pca_reconstruction_error
- [x] **3.** LinearAutoencoder
- [x] **4.** train_autoencoder
- [x] **5.** StackedAutoencoder
- [x] **6.** TiedAutoencoder
- [x] **7.** denoising_gain
- [x] **8.** kl_sparsity_loss
- [x] **9.** train_sparse
- [x] **10.** codings_classifier
- [x] **11.** VAE
- [x] **12.** vae_loss
- [x] **13.** train_vae
- [x] **14.** generate_images
- [x] **15.** Generator
- [x] **16.** gan_step
- [x] **17.** train_gan
- [x] **18.** save_vae

## Results

```
reconstruction MSE with 30 codings: PCA 0.0154   linear AE 0.0179   stacked AE 0.0156
tied weights: 82,414 parameters vs 163,814, MSE 0.0152
denoising: reconstruction is 2.1x closer to the clean image than the noisy input
sparsity: mean coding activation 0.50 -> 0.10 (target 0.10)
linear classifier on the stacked AE's 30 codings: 0.777 test accuracy

VAE: loss 369 -> 257 nats/image; 16 samples with mean pixel 0.21, spread across samples 0.172
interpolating between two test garments: 8 frames of 784 pixels
GAN after one epoch: D loss 0.74, G loss 2.39, discriminator accuracy 0.88
saved and reloaded VAE generates identical samples: True
```
