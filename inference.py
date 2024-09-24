from mmaction.apis import inference_recognizer, init_recognizer
from mmengine import Config

config = './fall_detection_exps/config.py'
config = Config.fromfile(config)
checkpoint = './FallCheckPoints/epoch_108.pth'
#108

model = init_recognizer(config, checkpoint,device='cuda:0')

from operator import itemgetter
video = './ArthurCaindoShortVideo.mp4'
label = './fallDataset/dataset/labels.txt'
results = inference_recognizer(model, video)

pred_scores = results.pred_score.tolist()
score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
top5_label = score_sorted[:5]

labels = open(label).readlines()
labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in top5_label]

print('The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])