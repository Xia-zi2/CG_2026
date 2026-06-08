import torch
import torch.nn.functional as F
from torch.optim import Adam
import logging
import csv
import matplotlib.pyplot as plt

from dataloader import load_transformed_dataset
from unet import SimpleUnet


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("flow_train.log", mode="w"),
        logging.StreamHandler()
    ]
)


def flow_matching_loss(model, x1, T, device):
    batch_size = x1.shape[0]

    x0 = torch.randn_like(x1)
    t = torch.randint(0, T, (batch_size,), device=device).long()

    tau = t.float() / max(1, T - 1)
    tau = tau.view(batch_size, 1, 1, 1)

    x_t = (1.0 - tau) * x0 + tau * x1
    target_v = x1 - x0
    predicted_v = model(x_t, t)

    loss = F.mse_loss(predicted_v, target_v)
    return loss


if __name__ == "__main__":
    model = SimpleUnet()
    T = 300
    BATCH_SIZE = 1
    epochs = 5000

    dataloader = load_transformed_dataset(batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
            loss = flow_matching_loss(model, batch, T, device)
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

    torch.save(model.state_dict(), f"./flow_matching_epochs_{epochs}.pth")

    with open("flow_loss_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        for step, loss_value in zip(step_history, loss_history):
            writer.writerow([step, loss_value])

    plt.figure(figsize=(8, 5))
    plt.plot(step_history, loss_history)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Flow Matching Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("flow_loss_curve.png", dpi=200)
    plt.show()