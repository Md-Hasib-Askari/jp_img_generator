import os
import streamlit as st
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from generator import Generator

# Config
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For compatibility with some environments

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 100
label_dim = 10
img_dim = 784

G = Generator(z_dim, label_dim, img_dim).to(device)
G.load_state_dict(torch.load("generator.pth", map_location=device))
G.eval()

# Streamlit UI
st.title("Digit Image Generator (CGAN)")
digit = st.number_input("Enter a digit (0-9):", min_value=0, max_value=9, step=1)
generate = st.button("Generate Images")

if generate:
    with torch.no_grad():
        z = torch.randn(5, z_dim).to(device)
        labels = torch.tensor([digit]*5).to(device)
        gen_images = G(z, labels).view(-1, 1, 28, 28).cpu()

        grid = make_grid(gen_images, nrow=5, normalize=True, pad_value=1)

        fig, ax = plt.subplots()
        ax.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
        ax.axis("off")
        st.pyplot(fig)
