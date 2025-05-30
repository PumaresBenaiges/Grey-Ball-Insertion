import os
import torch
import pandas as pd
import cv2
import numpy as np
from torchvision.utils import make_grid
from PIL import Image
import unet
import utils
import DatasetCreation as DC
from torch.utils.data import DataLoader

model_path = os.path.join('training_res_after_centering', "checkpoint_epoch_100.pt")
path_dir_save = 'combined'
os.makedirs(path_dir_save, exist_ok=True)

@torch.no_grad()
def save_image(tensor, fp, format=None, **kwargs):
    grid = make_grid(tensor, **kwargs)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)

def load_model(checkpoint_path):
    model = unet.UNetMobileNetV3()
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    return model

def overlay(base_img, overlay_img):
    """
    base_img: uint8 BGR
    overlay_img: uint8 BGR with correct gamma
    """
    mask = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(binary_mask)

    base_bg = cv2.bitwise_and(base_img, base_img, mask=mask_inv)
    overlay_fg = cv2.bitwise_and(overlay_img, overlay_img, mask=binary_mask)

    combined = cv2.add(base_bg, overlay_fg)
    return combined

if __name__ == '__main__':
    batch_size = 16
    num_workers = 0
    model = load_model(model_path)

    ball_data_val = pd.read_csv('ball_data_val.csv')
    transformations_data = pd.read_csv('homography_transformation.csv')
    input_paths_val, output_paths_val = utils.get_image_paths(ball_data_val)
    input_paths_val = {
        'seat_rows': 'scenes\\seat_rows.NEF'
    }
    input_images_val = utils.load_input_scenes(input_paths_val)
    print('Data and images loaded.')

    val_dataset = DC.SceneDataset(input_images_val, ball_data_val, output_paths_val, transformations_data)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    fixed_val_batch = next(iter(val_loader))
    fixed_val_input_image, fixed_val_input_cropped, fixed_val_target_image, fixed_val_mask = [
        t[:6].to(device) for t in fixed_val_batch
    ]

    model.eval()
    with torch.no_grad():
        output_val = model(fixed_val_input_cropped, fixed_val_input_image)
        fixed_val_mask_view = fixed_val_mask.view(fixed_val_mask.size(0), 1, fixed_val_mask.size(-2), fixed_val_mask.size(-1))
        output_val = output_val * fixed_val_mask_view.expand(-1, 3, -1, -1)

        for i in range(6):
            # Convert model output to RGB with gamma correction
            output_tensor = output_val[i].detach().cpu().clamp(0, 1)
            output_np = output_tensor.permute(1, 2, 0).numpy()
            output_np = np.where(
                output_np <= 0.0031308,
                12.92 * output_np,
                1.055 * np.power(output_np, 1 / 2.4) - 0.055
            )
            output_np_uint8 = (output_np * 255).clip(0, 255).astype(np.uint8)
            output_np_uint8 = cv2.cvtColor(output_np_uint8, cv2.COLOR_RGB2BGR)

            # Input image (already color-corrected)
            input_np = fixed_val_input_cropped[i].detach().cpu().permute(1, 2, 0).numpy()
            input_np_uint8 = (input_np * 255).clip(0, 255).astype(np.uint8)

            combined = overlay(input_np_uint8, output_np_uint8)

            file_name = f"combined_overlay_{i}.jpg"
            path_save = os.path.join(path_dir_save, file_name)
            cv2.imwrite(path_save, combined)
