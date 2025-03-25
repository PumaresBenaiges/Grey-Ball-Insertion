import os
import rawpy
import  imageio
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

IM_SIZE=(2844,4284,3)

def load_image(path):
    # Load and process input image (scene)
    with rawpy.imread(path) as raw:
        image = raw.postprocess()
        height, width, channels = image.shape
        if width < height:
            image = np.rot90(image)
    return image

def load_ball_position_data(path):
    df = pd.read_csv(path)
    df = df[['image_name', 'circle_x', 'circle_y', 'circle_radiuos']]
    return df

def create_probability_map(df):
    x = df['circle_x']
    y = df['circle_y']
    radius = df['circle_radiuos']
    xx, yy = np.meshgrid(np.arange(IM_SIZE[1]), np.arange(IM_SIZE[0]))
    dist = np.sqrt((xx-x) ** 2 + (yy-y) ** 2)
    mask = np.exp(-dist ** 2 / (2 * (radius / 2) ** 2))
    mask = mask.astype(np.float32)
    return mask

def get_image_paths():
    input_paths = []
    output_paths = []
    ball_data = pd.DataFrame()
    # Go over each scene
    for scene in os.listdir('scenes'):
        input_path = os.path.join('scenes', scene)
        input_paths.append(input_path)
        scene_id = os.path.splitext(scene)[0]
        shots_dir = os.path.join('scenes_shots', scene_id)

        df = pd.read_csv(os.path.join('illumination_gt', scene_id + '.csv'))
        df = df[['image_name', 'circle_x', 'circle_y', 'circle_radiuos']]
        ball_data = pd.concat([ball_data, df], axis=0)

        # Go over each shot
        for shot in os.listdir(shots_dir):
            output_path = os.path.join(shots_dir, shot)
            output_paths.append((scene_id, shot, output_path))
    print(f"Found {len(input_paths)} scenes.")
    print(f"Found {len(output_paths)} scene shots.")
    return input_paths, ball_data, output_paths

def load_input_scenes(input_paths):
    # TODO preprocessing, normalization?
    # image = (tf.cast(image, tf.float32) / 127.5) - 1  # Normalize to [-1, 1]
    # Load and process input image
    input_images = {}
    for input_path in input_paths:
        scene_id = os.path.basename(input_path)[:-4]
        input_image = load_image(input_path)
        input_images[scene_id] = input_image
    return input_images

def dataset_generator(input_images, df, output_images_paths):
    for path in output_images_paths:
        output_image = load_image(path[2])
        im_name = path[1][:-4] + '.jpg'
        df2 = df[df['image_name']==im_name]
        mask = create_probability_map(df2)
        try:
            mask = create_probability_map(df2)
        except:
            print('error: wrong df')
            mask = []
        yield input_images[path[0]], mask, output_image

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print(tf.config.list_physical_devices('GPU'))  # Should return empty list []

    print("Available CPUs:", multiprocessing.cpu_count())
    print("TensorFlow sees CPUs:", len(tf.config.list_physical_devices('CPU')))


    input_paths, ball_data, output_paths = get_image_paths()
    input_images = load_input_scenes(input_paths)

    # Create TensorFlow dataset
    # TODO: add probability image of ball position
    output_signature = (
        tf.TensorSpec(shape=IM_SIZE, dtype=tf.uint8),  # TODO type?
        tf.TensorSpec(shape=IM_SIZE, dtype=tf.uint8),
        tf.TensorSpec(shape=IM_SIZE, dtype=tf.uint8)
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(input_images, ball_data, output_paths),
        output_signature=output_signature
    )

    """    # Count total number of elements in dataset
    num_elements = dataset.reduce(0, lambda x, _: x + 1).numpy()
    print("Number of elements in dataset:", num_elements)
    from tqdm import tqdm
    for input_image, output_image in tqdm(dataset):
        pass  # or process them"""

    # Show first 3 samples from dataset
    for i, (input_image, mask, output_image) in enumerate(dataset.take(3)):
        print(f"Sample {i + 1}:")
        print("Input image shape:", input_image.shape)
        print("Output image shape:", output_image.shape)

        # Optional: show the input and output images
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(input_image.numpy())
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Output Image")
        plt.imshow(output_image.numpy())
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Mask")
        plt.imshow(mask.numpy())
        plt.axis('off')

        plt.savefig(f'dataset_sample_{i}.png')
        plt.close()