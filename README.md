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
- [ ] **5.** StackedAutoencoder
- [ ] **6.** TiedAutoencoder
- [ ] **7.** denoising_gain
- [ ] **8.** kl_sparsity_loss
- [ ] **9.** train_sparse
- [ ] **10.** codings_classifier
- [ ] **11.** VAE
- [ ] **12.** vae_loss
- [ ] **13.** train_vae
- [ ] **14.** generate_images
- [ ] **15.** Generator
- [ ] **16.** gan_step
- [ ] **17.** train_gan
- [ ] **18.** save_vae

---

Built on Deep-ML.
