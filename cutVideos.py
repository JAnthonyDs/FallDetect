import os
from moviepy.video.io.VideoFileClip import VideoFileClip

txt_file = '/root/fallDetection/fallDataset/val_annotations.txt'
output_dir = '/root/fallDetection/dt/val/'
dir_file_txt = '/root/fallDetection/dt/'

os.makedirs(output_dir, exist_ok=True)

def cut_video(video_path, start_frame, end_frame, output_path):
    with VideoFileClip(video_path) as video:
        fps = video.fps
        start_time = start_frame / fps
        end_time = end_frame / fps
        video_cut = video.subclip(start_time, end_time)
        video_cut.write_videofile(output_path, codec='libx264')

labels = []

with open(txt_file, 'r') as f:
    for line in f:
        video_path, start_frame, end_frame, label = line.strip().split()
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        label = int(label)
        
        video_name = os.path.basename(video_path).replace('.avi', f'_{start_frame}_{end_frame}.mp4')
        output_path = os.path.join(output_dir, video_name)
        
        cut_video(video_path, start_frame, end_frame, output_path)
        
        labels.append(f"{output_path} {label}")

# Salva o arquivo de labels
with open(os.path.join(dir_file_txt, 'labelVal.txt'), 'w') as f:
    for label in labels:
        f.write(label + '\n')

print("Processamento concluído. Vídeos cortados e labels gerados.")
