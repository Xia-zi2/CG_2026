import torch
from dataloader import show_tensor_image
from unet import SimpleUnet


@torch.no_grad()
def sample_flow_matching(model, device, img_size, T):
    img = torch.randn((1, 3, img_size, img_size), device=device)
    show_tensor_image(img, i=0, mode="flow_generation")

    dt = 1.0 / T

    for i in range(T):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        v = model(img, t)
        img = img + dt * v

        if i % 10 == 0:
            show_tensor_image(img, i=i + 1, mode="flow_generation")

    return img


def test_flow_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 256
    T = 300

    model = SimpleUnet().to(device)
    model.load_state_dict(torch.load("./flow_matching_epochs_5000.pth", map_location=device))
    model.eval()

    sample_flow_matching(model, device, img_size, T)


if __name__ == "__main__":
    test_flow_generation()