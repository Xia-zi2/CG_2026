import os
import cv2 as cv
import numpy as np
import torch


def load_image_bgr(path, img_size=256):
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return cv.resize(img, (img_size, img_size))


def load_mask_gray(path, img_size=256, device="cpu"):
    mask = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot load mask: {path}")
    mask = cv.resize(mask, (img_size, img_size), interpolation=cv.INTER_NEAREST)
    mask = (mask > 127).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)


def make_freehand_mask(img_bgr, brush_size=12, device="cpu"):
    os.makedirs("inpaint", exist_ok=True)

    show = img_bgr.copy()
    mask_np = np.ones(img_bgr.shape[:2], dtype=np.float32)
    drawing = False

    def draw(event, x, y, flags, param):
        nonlocal drawing, show, mask_np
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False

        if drawing and event in (cv.EVENT_LBUTTONDOWN, cv.EVENT_MOUSEMOVE):
            cv.circle(show, (x, y), brush_size, (0, 0, 255), -1)
            cv.circle(mask_np, (x, y), brush_size, 0.0, -1)

    print("左键拖动绘制 mask，按 r 重置，按 q 确认结束")
    cv.namedWindow("Draw Mask")
    cv.setMouseCallback("Draw Mask", draw)

    while True:
        cv.imshow("Draw Mask", show)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            show = img_bgr.copy()
            mask_np = np.ones(img_bgr.shape[:2], dtype=np.float32)

    cv.destroyWindow("Draw Mask")

    corrupted_bgr = img_bgr.copy()
    corrupted_bgr[mask_np < 0.5] = 0

    cv.imwrite("inpaint/mask.png", (mask_np * 255).astype(np.uint8))
    cv.imwrite("inpaint/corrupted_input.png", corrupted_bgr)

    return torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)


if __name__ == "__main__":
    img = load_image_bgr("./datasets-1/test/cls0/1.jpg", img_size=256)
    make_freehand_mask(img)