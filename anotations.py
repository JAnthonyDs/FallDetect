import pandas as pd
import os

csv_path = '/root/fallDetection/fallDataset/data_tuple3.csv'

annotations = pd.read_csv(csv_path)

print(annotations.head())

train_annotations = []
val_annotations = []

for _, row in annotations.iterrows():
    video_path = f"/root/fallDetection/fallDataset/dataset/dataset/chute{int(row['chute'])}/cam{int(row['cam'])}.avi"
    start_frame = int(row['start'])
    end_frame = int(row['end'])
    label = int(row['label'])
    annotation_line = f"{video_path} {label}"
    annotation_line = f"{video_path} {start_frame} {end_frame} {label}"
    
    if len(train_annotations) < 300:
        train_annotations.append(annotation_line)
    else:
        val_annotations.append(annotation_line)

    if int(row['chute'] == 18):
        break
# Salvar as anotações em arquivos de texto
with open('/root/fallDetection/fallDataset/train_annotations.txt', 'w') as f:
    f.write("\n".join(train_annotations))

with open('/root/fallDetection/fallDataset/val_annotations.txt', 'w') as f:
    f.write("\n".join(val_annotations))
