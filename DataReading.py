import os
import rawpy
import  imageio
import numpy as np
import pandas as pd
# import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def load_image(path):
    # Load and process input image (scene)
    with rawpy.imread(path) as raw:
        image = raw.postprocess()
        height, width, channels = image.shape
        if width < height:
            image = np.rot90(image)
    return image
"""
Parallel version"""
def load_images(scenes_dir='scenes', shots_dir='scenes_shots', max_workers=8):
    input_images = {}
    output_images = []

    scene_tasks = []
    shot_tasks = []

    # Collect tasks first
    for scene in os.listdir(scenes_dir):
        scene_path = os.path.join(scenes_dir, scene)
        scene_id = os.path.splitext(scene)[0]
        input_images[scene_id] = []
        scene_tasks.append((scene_id, scene_path))

        # Collect shot tasks
        scene_shots_dir = os.path.join(shots_dir, scene_id)
        for shot in os.listdir(scene_shots_dir):
            shot_path = os.path.join(scene_shots_dir, shot)
            shot_tasks.append((scene_id, shot_path))

    # Load images in parallel
    print(f"Loading {len(scene_tasks)} scenes and {len(shot_tasks)} shots using {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Scene futures
        future_to_scene = {executor.submit(load_image, path): scene_id for scene_id, path in scene_tasks}
        for future in as_completed(future_to_scene):
            scene_id = future_to_scene[future]
            try:
                image = future.result()
                input_images[scene_id].append(image)
            except Exception as e:
                print(f"Failed to load scene {scene_id}: {e}")

        # Shot futures
        future_to_shot = {executor.submit(load_image, path): (scene_id, path) for scene_id, path in shot_tasks}
        for future in as_completed(future_to_shot):
            scene_id, shot_path = future_to_shot[future]
            try:
                image = future.result()
                output_images.append((scene_id, image))
            except Exception as e:
                print(f"Failed to load shot {shot_path}: {e}")

    print(f"Found {len(input_images)} scenes.")
    print(f"Found {len(output_images)} scene shots.")
    return input_images, output_images

def load_ball_position_data(path):
    df = pd.read_csv(path)
    df = df[['image_name', 'circle_x', 'circle_y', 'circle_radiuos']]
    return df

def create_probability_map(x,y,radius, im_size):
    xx, yy = np.meshgrid(np.arange(im_size[0], im_size[1]))
    dist = np.sqrt((xx-x) ** 2 + (yy-y) ** 2)
    mask = np.exp(-dist ** 2 / (2 * (radius / 2) ** 2))
    mask = mask.astype(np.float32)
    return mask

def load_data(scenes_dir='scenes', scenes_shots_dir='scenes_shots'):
    input_images = {}
    output_images = []
    ball_data = pd.DataFrame()

    for scene in os.listdir(scenes_dir):
        scene_id = os.path.splitext(scene)[0]
        scene_path = os.path.join(scenes_dir, scene)
        input_images[scene_id] = load_image(scene_path)

        # Load dataframe with ball position
        # df = load_ball_position_data(os.path.join('illumination_gt', scene_id+'.csv'))
        try:
            df = pd.read_csv(os.path.join('illumination_gt', scene_id+'.csv'))
            df = df[['image_name', 'circle_x', 'circle_y', 'circle_radiuos']]
            ball_data = pd.concat([ball_data, df], axis=0)
        except:
            print("filenotfound ", scene_id)

        # Go over each shot
        shots_dir = os.path.join(scenes_shots_dir, scene_id)
        for shot in os.listdir(shots_dir):
            shot_path = os.path.join(shots_dir, shot)
            output_images.append((scene_id, shot, load_image(shot_path)))
        print(f"Scene {scene_id} done.")
    print(f"Found {len(input_images)} scenes.")
    print(f"Found {len(output_images)} scene shots.")
    return input_images, output_images, ball_data



"""
def get_image_paths():    
    input_paths = []
    output_paths = []
    # Go over each scene
    for scene in os.listdir('scenes'):
        input_path = os.path.join('scenes', scene)
        input_paths.append(input_path)
        scene_id = os.path.splitext(scene)[0]
        shots_dir = os.path.join('scenes_shots', scene_id)
        # Go over each shot
        for shot in os.listdir(shots_dir):
            output_path = os.path.join(shots_dir, shot)
            output_paths.append((scene_id, output_path))
    print(f"Found {len(input_paths)} scenes.")
    print(f"Found {len(output_paths)} scene shots.")
    return input_paths, output_paths
def load_input_scenes(input_paths):
    # TODO preprocessing, normalization?
    input_images = {}
    for input_path in input_paths:
        scene_id = os.path.basename(input_path)[:-4]
        # Load and process input image (scene)
        with rawpy.imread(input_path) as raw:
            input_image = raw.postprocess()
            height, width, channels = input_image.shape
            if width < height:
                input_image = np.rot90(input_image)
        input_images[scene_id] = input_image
    return input_images
def load_output_shot(output_path):
    # TODO preprocessing, normalization?
    # image = (tf.cast(image, tf.float32) / 127.5) - 1  # Normalize to [-1, 1]
    # Load and process output image (ground truth)
    with rawpy.imread(output_path[1]) as raw:
        output_image = raw.postprocess()
        height, width, channels = output_image.shape
        if width < height:
            output_image = np.rot90(output_image)
    return (output_path[0], output_image)
def load_output_shots(output_paths, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(load_output_shot, output_paths))
"""

def dataset_generator(input_images, output_images):
    for image in output_images:
        yield input_images[image[0]], image[1]

if __name__ == '__main__':
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print(tf.config.list_physical_devices('GPU'))  # Should return empty list []

    print("Available CPUs:", multiprocessing.cpu_count())
    print("TensorFlow sees CPUs:", len(tf.config.list_physical_devices('CPU')))"""
    im1, im2 = load_images()

    input_images, output_images, ball_data = load_data()




    """    # Create TensorFlow dataset
    # TODO: add probability image of ball position
    output_signature = (
        tf.TensorSpec(shape=(2844,4284,3), dtype=tf.uint8),  # TODO type?
        tf.TensorSpec(shape=(2844,4284,3), dtype=tf.uint8)
        # tf.TensorSpec(shape=(2844,4284,3), dtype=tf.uint8)
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(input_images, output_images),
        output_signature=output_signature
    )"""

    """    # Count total number of elements in dataset
    num_elements = dataset.reduce(0, lambda x, _: x + 1).numpy()
    print("Number of elements in dataset:", num_elements)
    from tqdm import tqdm
    for input_image, output_image in tqdm(dataset):
        pass  # or process them

    # Show first 3 samples from dataset
    for i, (input_image, output_image) in enumerate(dataset.take(3)):
        print(f"Sample {i + 1}:")
        print("Input image shape:", input_image.shape)
        print("Output image shape:", output_image.shape)

        # Optional: show the input and output images
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(input_image.numpy())
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Output Image")
        plt.imshow(output_image.numpy())
        plt.axis('off')

        plt.savefig(f'dataset_sample_{i}.png')
        plt.close()"""