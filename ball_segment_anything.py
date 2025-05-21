import pandas as pd
import rawpy
from ultralytics import SAM

import json
import os
import numpy as np
import cv2

import utils


def readImage(nef_path):
    with rawpy.imread(nef_path) as raw:
        rgb_image = raw.postprocess(output_bps=8)
    return rgb_image

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
        raise ValueError("No object found in mask.")

    # Fit a minimum enclosing circle
    cnt = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(cnt)

    # Create clean circular mask
    h, w = mask.shape
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, center=(int(x), int(y)), radius=int(radius), color=1, thickness=-1)

    return circle_mask

def predict_mask(nef_path, model, point):
    rgb_image = readImage(nef_path)
    image = np.asarray(rgb_image, dtype=np.uint8)

    results = model(image, points=[point], labels=[1])  # 1 = foreground

    if results and results[0].masks is not None:
        mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
        radius = calculate_radius(mask)
        erodedMask = erode_keep_ball(mask, radius)
        circle_mask = fitCircle(erodedMask)
        return circle_mask * 255

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
    mask_uint8 = (mask > 127).astype(np.uint8)
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
    ys, xs = np.where(mask > 127)
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



CHECKPOINT_PATH = "sam2.1_l.pt"
OUTPUT_DIR = "scene_shots_masks"

if __name__ == '__main__':
    model = SAM(CHECKPOINT_PATH)

    ball_data = pd.read_csv('ball_data.csv')
    _, output_paths = utils.get_image_paths(ball_data)
    num_img = len(output_paths)
    curr_index_img = 0

    # Iterate through scene_id (key) and image path (value)
    for scene_id, shot_id, image_path in output_paths:

        # Get curret line of the balldata
        idx = ball_data.index[ball_data['image_name'] == shot_id]
        if idx.empty:
            continue
        i = idx[0]

        # Get previous ball center
        cx = ball_data.at[i, 'circle_x']
        cy = ball_data.at[i, 'circle_y']
        point = [cx, cy]

        currMask = predict_mask(image_path, model, point)
        bbox, center, radius = calculate_mask_params(currMask)
        scene_output_dir = os.path.join(OUTPUT_DIR, scene_id)
        #save_mask_json(currMask, bbox, center, radius, shot_id, scene_output_dir)

        ball_data.at[i, 'circle_x'] = center[0]
        ball_data.at[i, 'circle_y'] = center[1]
        ball_data.at[i, 'circle_radiuos'] = radius
        curr_index_img = curr_index_img + 1
        print(f"{curr_index_img}/{num_img} shot {shot_id} with center at: {center[0]}, {center[1]}; radius = {radius}")

    ball_data.to_csv('ball_data_modified.csv', index=False)
    print("Saved as ball_data_modified.csv")