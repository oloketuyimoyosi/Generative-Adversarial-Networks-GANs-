"""
Generative Adversarial Network (GAN) in PyTorch
================================================
This script implements a basic GAN for generating MNIST-like handwritten digits.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# Hyperparameters
latent_dim = 100
hidden_dim = 256
image_dim = 28 * 28
batch_size = 64
num_epochs = 50
learning_rate = 0.0002


class Generator(nn.Module):
    """Generator Network: Generates fake images from random noise."""
    
    def __init__(self, latent_dim, hidden_dim, image_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, image_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """Discriminator Network: Distinguishes real images from fake ones."""
    
    def __init__(self, image_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


def load_data(batch_size):
    """Load and prepare MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return train_loader


def train_gan(generator, discriminator, train_loader, num_epochs, device):
    """Train the GAN."""
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)
    
    G_losses = []
    D_losses = []
    
    print("Starting Training...")
    for epoch in range(num_epochs):
        for batch_idx, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.view(-1, image_dim).to(device)
            
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Train Discriminator
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)
            
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(noise)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(noise)
            outputs = discriminator(fake_images)
            
            g_loss = criterion(outputs, real_labels)
            
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()
        
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
    
    print("Training completed!")
    return G_losses, D_losses


def plot_losses(G_losses, D_losses):
    """Plot training losses."""
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_losses.png')
    print("Training losses plot saved as 'training_losses.png'")


def generate_and_save_images(generator, device, num_images=64):
    """Generate and save images."""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, latent_dim).to(device)
        fake_images = generator(noise).cpu().view(-1, 28, 28)
    
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < len(fake_images):
            ax.imshow(fake_images[i], cmap='gray')
        ax.axis('off')
    plt.suptitle('Generated Images from GAN', fontsize=16)
    plt.tight_layout()
    plt.savefig('generated_images.png')
    print("Generated images saved as 'generated_images.png'")


def main():
    """Main function to run the GAN training."""
    print("=" * 50)
    print("GAN Training in PyTorch")
    print("=" * 50)
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader = load_data(batch_size)
    print(f"Dataset loaded: {len(train_loader)} batches")
    
    # Initialize networks
    print("\nInitializing networks...")
    generator = Generator(latent_dim, hidden_dim, image_dim).to(device)
    discriminator = Discriminator(image_dim, hidden_dim).to(device)
    print("Generator and Discriminator initialized")
    
    # Train
    G_losses, D_losses = train_gan(
        generator, discriminator, train_loader, num_epochs, device
    )
    
    # Plot losses
    print("\nPlotting training losses...")
    plot_losses(G_losses, D_losses)
    
    # Generate images
    print("\nGenerating sample images...")
    generate_and_save_images(generator, device)
    
    # Save models
    print("\nSaving models...")
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
    print("Models saved successfully!")
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)


if __name__ == '__main__':
    main()
