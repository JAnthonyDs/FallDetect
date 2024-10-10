import cv2
import os
from moviepy.editor import VideoFileClip, vfx

def augment_video(video_path, output_path, augmentation_type):
    clip = VideoFileClip(video_path)
    augmented_clip = None

    if augmentation_type == 'flip':
        augmented_clip = clip.fx(vfx.mirror_x)
    elif augmentation_type == 'rotate':
        augmented_clip = clip.rotate(10)  
    elif augmentation_type == 'brightness':
        augmented_clip = clip.fx(vfx.colorx, factor=1.5)  
    elif augmentation_type == 'contrast':
        augmented_clip = clip.fx(vfx.lum_contrast, lum=50, contrast=1.5)
    
    augmented_clip.write_videofile(output_path, audio=False)
    augmented_clip.close()

def augment_dataset(input_txt_path, augmentations=['flip', 'rotate', 'brightness']):
    with open(input_txt_path, 'r') as f:
        lines = f.readlines()

    new_annotations = []

    for line in lines:
        video_path, label = line.strip().split()
        base_name = os.path.basename(video_path).split('.')[0]
        dir_name = os.path.dirname(video_path)

        for aug in augmentations:
            new_video_name = f"{base_name}_{aug}.mp4"
            new_video_path = os.path.join(dir_name, new_video_name)

            augment_video(video_path, new_video_path, aug)

            new_annotations.append(f"{new_video_path} {label}")

    with open(input_txt_path, 'a') as f:  
        for annotation in new_annotations:
            f.write(annotation + '\n')

input_txt = '/root/fallDetection/dt/labelTrain.txt'

augment_dataset(input_txt)
