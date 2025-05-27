import pandas as pd
import rawpy
from ultralytics import SAM

import json
import os
import numpy as np
import cv2

import utils

def get_nef_dimensions(nef_path):
    with rawpy.imread(nef_path) as raw:
        height = raw.sizes.raw_height
        width = raw.sizes.raw_width
    return width, height

def erode_keep_ball(mask, ball_radius):
    # Compute kernel size from radius
    kernel_size = int(0.5 * ball_radius) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    eroded = cv2.erode(mask, kernel, iterations=1)
    dilate = cv2.dilate(eroded, kernel, iterations=1)

    return dilate


def fitCircle(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    # Fit a minimum enclosing circle
    cnt = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(cnt)

    # Create clean circular mask
    h, w = mask.shape
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, center=(int(x), int(y)), radius=int(radius), color=1, thickness=-1)

    return circle_mask


def cut_roi(rgb, x_approx, y_approx):
    h, w = rgb.shape[:2]
    x1, y1 = max(x_approx - ROI_RADIUS, 0), max(y_approx - ROI_RADIUS, 0)
    x2, y2 = min(x_approx + ROI_RADIUS, w), min(y_approx + ROI_RADIUS, h)

    rgb_roi = rgb[y1:y2, x1:x2]  # Apply same crop to rgn
    return rgb_roi, x1, y1


# === Expand back to original image size ===
def extend_mask(mask, h, w, x1, y1):
    full_mask = np.zeros((h, w), dtype=np.uint8)
    mask_h, mask_w = mask.shape

    # Place circle_mask in the original image space
    full_mask[y1:y1 + mask_h, x1:x1 + mask_w] = mask
    return full_mask


def predict_mask(rgb_image, model, point):
    h, w = rgb_image.shape[:2]
    rgb_image, x1, y1 = cut_roi(rgb_image, point[0], point[1])
    if DEBUG:
        cv2.imwrite(os.path.join(SAVE_DIR, 'cut.jpg'), rgb_image)
    point[0] = point[0] - x1
    point[1] = point[1] - y1
    image = np.asarray(rgb_image, dtype=np.uint8)

    results = model(image, points=[point], labels=[1])  # 1 = foreground

    if results and results[0].masks is not None:
        mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
        radius = calculate_radius(mask)
        erodedMask = erode_keep_ball(mask, radius)
        circle_mask = fitCircle(erodedMask)
        full_mask = extend_mask(circle_mask, h, w, x1, y1)

        if DEBUG:
            cv2.imwrite(os.path.join(SAVE_DIR, 'sam.jpg'), mask*255)
            cv2.imwrite(os.path.join(SAVE_DIR, 'erosion.jpg'), erodedMask * 255)
            cv2.imwrite(os.path.join(SAVE_DIR, 'circle_mask.jpg'), circle_mask * 255)

        return full_mask

    return []


def calculate_radius(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest_contour)
        return  float(min(bw, bh) / 2)
    else:
        return 0.0


def calculate_mask_params(mask):
    mask_uint8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest_contour)
        bbox = [int(x), int(y), int(bw), int(bh)]
        center = [int(x + bw / 2), int(y + bh / 2)]
        radius = int(min(bw, bh) / 2)
        return bbox, center, radius
    else:
        return [0, 0, 0, 0], [0, 0], 0.0


def save_mask_json(mask, bbox, center, radius, image_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{image_name}.json")

    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    pixels = [[int(x), int(y)] for x, y in zip(xs, ys)]

    data = {
        "size": [int(w), int(h)],
        "bbox": bbox,
        "center": center,
        "radius": radius,
        "pixels": pixels
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_bbox_from_json(image_name, input_dir):
    filepath = os.path.join(input_dir, f"{image_name}.json")
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["bbox"]  # [x, y, width, height]


def load_center_from_json(image_name, input_dir):
    filepath = os.path.join(input_dir, f"{image_name}.json")
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["center"]  # [cx, cy]


def load_mask_json(image_name, input_dir):
    filepath = os.path.join(input_dir, f"{image_name}.json")
    with open(filepath, "r") as f:
        data = json.load(f)

    w, h = data["size"]
    mask = np.zeros((h, w), dtype=np.uint8)

    for x, y in data["pixels"]:
        mask[y, x] = 255

    return mask


def crop_image(image, cx, cy, crop_size=256):
    h, w = image.shape[:2]
    half_crop = crop_size // 2

    # Compute cropping box
    x1 = cx - half_crop
    y1 = cy - half_crop
    x2 = cx + half_crop
    y2 = cy + half_crop

    # Compute how much padding is needed
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    # Apply padding if needed
    if any([pad_left, pad_top, pad_right, pad_bottom]):
        image = cv2.copyMakeBorder(
            image,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # Black padding
        )

    # Adjust coordinates to the padded image
    x1 += pad_left
    y1 += pad_top
    x2 += pad_left
    y2 += pad_top

    # Final 256x256 crop with no distortion
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image



CHECKPOINT_PATH = "sam2.1_l.pt"
OUTPUT_DIR = "scene_shots_masked"
ROI_RADIUS = 300
SAVE_DIR = 'save'
DEBUG = False

if __name__ == '__main__':
    model = SAM(CHECKPOINT_PATH)

    ball_data = pd.read_csv('ball_data_modified.csv')
    _, output_paths = utils.get_image_paths(ball_data)
    num_img = len(output_paths)
    curr_index_img = 0

    # Iterate through scene_id (key) and image path (value)
    for scene_id, shot_id, image_path in output_paths:
        if scene_id != scene_id:
            continue
        # Get curret line of the balldata
        idx = ball_data.index[ball_data['image_name'] == shot_id]
        if idx.empty:
            continue
        i = idx[0]

        # Get previous ball center
        cx = ball_data.at[i, 'circle_x']
        cy = ball_data.at[i, 'circle_y']
        point = [cx, cy]



        rgb_image = utils.load_image(image_path)

        # Switch cx and cy if image rotated
        w, h = get_nef_dimensions(image_path)
        h_image, w_image = rgb_image.shape[:2]
        if h_image != h and w_image != w:
            temp = cx
            cx = cy
            cy = cx

        currMask = predict_mask(rgb_image, model, point)
        bbox, center, radius = calculate_mask_params(currMask)

        masked_image = rgb_image * currMask[:, :, np.newaxis]
        scene_output_dir = os.path.join(OUTPUT_DIR, scene_id)
        output_path = os.path.join(scene_output_dir, f'{shot_id}.jpg')

        os.makedirs(scene_output_dir, exist_ok=True)
        cropped_masked_image = crop_image(masked_image, center[0], center[1])
        if DEBUG:
            cv2.imwrite(os.path.join(SAVE_DIR, 'cropped_image.jpg'), cropped_masked_image)
        cv2.imwrite(output_path, cropped_masked_image)

        #save_mask_json(currMask, bbox, center, radius, shot_id, scene_output_dir)

        ball_data.at[i, 'circle_x'] = center[0]
        ball_data.at[i, 'circle_y'] = center[1]
        ball_data.at[i, 'circle_radiuos'] = radius
        curr_index_img = curr_index_img + 1
        print(f"{curr_index_img}/{num_img} shot {shot_id} with center at: {center[0]}, {center[1]}; radius = {radius}")


    ball_data.to_csv('ball_data_upd.csv', index=False)
    print("Saved as ball_data_upd.csv")