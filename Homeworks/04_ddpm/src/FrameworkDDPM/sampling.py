import cv2 as cv
import torch
from torchvision import transforms
from dataloader import show_tensor_image
from forward_noising import (
    betas,
    get_index_from_list,
    posterior_variance,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    forward_diffusion_sample,
)
from mask import load_image_bgr, load_mask_gray
from unet import SimpleUnet


@torch.no_grad()
def sample_timestep(model, x, t):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    posterior_variance_t = get_index_from_list(
        posterior_variance, t, x.shape
    )

    predicted_noise = model(x, t)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )

    if t[0] == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
    return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def forward_one_step_from_xt(x_t, t_next):
    beta_t = get_index_from_list(betas, t_next, x_t.shape)
    alpha_t = 1.0 - beta_t
    noise = torch.randn_like(x_t)
    return torch.sqrt(alpha_t) * x_t + torch.sqrt(beta_t) * noise


@torch.no_grad()
def sample_plot_image(model, device, img_size,T):
    img = torch.randn((1, 3, img_size, img_size), device=device)
    show_tensor_image(img)

    for i in reversed(range(T)):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(model, img, t)
        if i % 10==0:
            show_tensor_image(img , i=T-i,mode="generation")
    return img

def test_image_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 256
    T = 300

    model = SimpleUnet().to(device)
    model.load_state_dict(torch.load("./ddpm_mse_epochs_5000.pth", map_location=device))
    model.eval()

    sample_plot_image(model, device, img_size, T)

@torch.no_grad()
def inpaint(model, device, img, mask, t_max=50):
    schedule = make_repaint_schedule(
        t_max=t_max,
        jump_length=10,
        jump_n_sample=10,
    )

    model.eval()
    img = img.to(device)
    mask = mask.to(device)

    if mask.shape[1] == 1:
        mask_3 = mask.repeat(1, 3, 1, 1)
    else:
        mask_3 = mask

    t_init = torch.full((img.shape[0],), t_max, device=device, dtype=torch.long)

    x = torch.randn_like(img)
    x_known, _ = forward_diffusion_sample(img, t_init, device)
    x = mask_3 * x_known + (1.0 - mask_3) * x

    for k in range(len(schedule) - 1):
        t_cur_val = schedule[k]
        t_next_val = schedule[k + 1]

        t_cur = torch.full((img.shape[0],), t_cur_val, device=device, dtype=torch.long)
        t_next = torch.full((img.shape[0],), t_next_val, device=device, dtype=torch.long)

        if t_next_val == t_cur_val - 1:
            x = sample_timestep(model, x, t_cur)

        elif t_next_val == t_cur_val + 1:
            x = forward_one_step_from_xt(x, t_next)

        else:
            continue

        if t_next_val == 0:
            x_known = img
        else:
            x_known, _ = forward_diffusion_sample(img, t_next, device)

        x = mask_3 * x_known + (1.0 - mask_3) * x

        if (k + 1) % 5 == 0:
            show_tensor_image(x, i=k + 1, mode="inpaint")

    return x


def test_image_inpainting():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 256

    model = SimpleUnet().to(device)
    model.load_state_dict(torch.load("./ddpm_mse_epochs_5000.pth", map_location=device))
    model.eval()

    img_bgr = load_image_bgr("./inpaint/corrupted_input.png", img_size=img_size)
    mask = load_mask_gray("./inpaint/mask.png", img_size=img_size, device=device)

    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0).to(device)
    img_tensor = img_tensor * 2.0 - 1.0

    result = inpaint(model, device, img_tensor, mask, t_max=50)
    return result

def make_repaint_schedule(t_max=50, jump_length=10, jump_n_sample=10):
    jumps = {}

    for j in range(0, t_max - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1

    t = t_max
    schedule = []

    while t >= 1:
        t = t - 1
        schedule.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] -= 1
            for _ in range(jump_length):
                t = t + 1
                schedule.append(t)

    return schedule

if __name__ == "__main__":
    test_image_generation()
    #test_image_inpainting()