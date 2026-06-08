from forward_noising import forward_diffusion_sample
from unet import SimpleUnet
from dataloader import load_transformed_dataset
import torch.nn.functional as F
import torch
from torch.optim import Adam
import logging
from tqdm import trange
import cv2 as cv
import csv
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train.log", mode="w"),
        logging.StreamHandler()
    ]
)

def get_loss(model, x_0, t, device):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    predicted_noise = model(x_noisy, t)
    loss=F.mse_loss(predicted_noise, noise)
    
    return loss


if __name__ == "__main__":
    model = SimpleUnet()
    T = 300
    BATCH_SIZE = 1
    epochs = 5000

    dataloader = load_transformed_dataset(batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    logging.info(f"Using device: {device}")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    loss_history = []
    step_history = []
    global_step = 0

    for epoch in range(epochs):
        for batch_idx, (batch, _) in enumerate(dataloader):
            optimizer.zero_grad()

            batch = batch.to(device)
            t = torch.randint(0, T, (batch.shape[0],), device=device).long()
            loss = get_loss(model, batch, t, device)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            loss_history.append(loss_value)
            step_history.append(global_step)

            if batch_idx % 50 == 0:
                logging.info(
                    f"Epoch {epoch} | Batch index {batch_idx:03d} | Step {global_step:06d} | Loss: {loss_value:.6f}"
                )
            global_step += 1

    torch.save(model.state_dict(), f"./ddpm_mse_epochs_{epochs}.pth")

    with open("loss_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        for step, loss_value in zip(step_history, loss_history):
            writer.writerow([step, loss_value])

    plt.figure(figsize=(8, 5))
    plt.plot(step_history, loss_history)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=200)
    plt.show()