import mlflow
import mlflow.pytorch
from mmaction.apis import init_recognizer
from mmengine import Config
from mmengine.evaluator import BaseMetric
from torch.optim import SGD
from mmengine.runner import Runner
import os.path as osp
import mmengine
from mmengine.registry import MODELS, DATASETS, METRICS  
from mmaction.models import Recognizer2D
from mmaction.datasets import VideoDataset  
from mmaction.datasets.transforms import DecordInit, DecordDecode, MultiScaleCrop, Flip, FormatShape, PackActionInputs
from mmengine.registry import TRANSFORMS
import torch
from typing import Sequence, List
import numpy as np
from PIL import Image
from mmaction.datasets.transforms import SampleFrames

TRANSFORMS.register_module()(MultiScaleCrop)
TRANSFORMS.register_module()(DecordDecode)
TRANSFORMS.register_module()(Flip) 
TRANSFORMS.register_module()(PackActionInputs)
TRANSFORMS.register_module()(DecordInit)
TRANSFORMS.register_module()(SampleFrames)
TRANSFORMS.register_module()(FormatShape) 
DATASETS.register_module()(VideoDataset)

@MODELS.register_module()
class CustomModel(Recognizer2D):
    def __init__(self, backbone, cls_head, *args, **kwargs):
        super()._init_(backbone=backbone, cls_head=cls_head, *args, **kwargs)


@METRICS.register_module()
class Metrics(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = []

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            try:
                pred_scores = sample['pred_score']
                pred_labels = sample['pred_label']
                gt_labels = sample['gt_label']
            except KeyError as e:
                print(f"Chave não encontrada: {e}")
                continue

            if pred_scores.dim() == 1:
                preds = pred_labels
            elif pred_scores.dim() == 2:
                preds = pred_scores.argmax(dim=1)
            else:
                raise ValueError("Dimensão inesperada para pred_scores.")

            self.results.append({
                'preds': pred_labels.cpu(),
                'gt_labels': gt_labels.cpu()
            })

    def compute_metrics(self, results):
        if not self.results:
            raise ValueError("PrecisionMetric got empty `self.results`. Ensure that data is being added in `process` method.")

        preds = torch.cat([item['preds'] for item in self.results])
        gt_labels = torch.cat([item['gt_labels'] for item in self.results])

        unique_labels = torch.unique(gt_labels)

        precision_per_class = {}
        recall_per_class = {}
        tp_total = 0
        fp_total = 0
        fn_total = 0

        for label in unique_labels:
            tp = ((preds == label) & (gt_labels == label)).sum().item()
            fp = ((preds == label) & (gt_labels != label)).sum().item()
            fn = ((preds != label) & (gt_labels == label)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            precision_per_class[label.item()] = precision
            recall_per_class[label.item()] = recall

            tp_total += tp
            fp_total += fp
            fn_total += fn

        macro_precision = sum(precision_per_class.values()) / len(unique_labels)
        macro_recall = sum(recall_per_class.values()) / len(unique_labels)

        f_score = (2 * macro_recall * macro_precision) / (macro_recall + macro_precision) if (macro_recall + macro_precision) > 0 else 0

        correct = (preds == gt_labels).sum().item()
        total = len(gt_labels)
        accuracy = correct / total if total > 0 else 0

        return dict(
            precision=macro_precision,
            recall=macro_recall,
            f_score=f_score,
            accuracy=accuracy
        )

    

#Retirado da documentação
@METRICS.register_module()  # register the Accuracy class to the METRICS registry
class SimpleAccuracy(BaseMetric):
    """ Accuracy Evaluator

    Default prefix: ACC

    Metrics:
        - accuracy (float): classification accuracy
    """

    default_prefix = 'ACC'  # set default_prefix

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]):
        """Process one batch of data and predictions. The processed
        Results should be stored in `self.results`, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """

        # fetch classification prediction results and category labels
        result = {
            'pred': data_samples['pred_label'],
            'gt': data_samples['data_sample']['gt_label']
        }

        # store the results of the current batch into self.results
        self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        # aggregate the classification prediction results and category labels for all samples
        preds = np.concatenate([res['pred'] for res in results])
        gts = np.concatenate([res['gt'] for res in results])

        # calculate the classification accuracy
        acc = (preds == gts).sum() / preds.size

        # return evaluation metric results
        return {'accuracy': acc}


# Carregar a configuração do modelo
cfg = Config.fromfile('./mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py')

cfg.data_root = './dt/train/'
cfg.data_root_val = './dt/val/'
cfg.ann_file_train = './dt/labelTrain.txt'
cfg.ann_file_val = './dt/labelVal.txt'
cfg.test_dataloader.dataset.ann_file = './dt/labelVal.txt'
cfg.test_dataloader.dataset.data_prefix.video = './dt/val/'
cfg.train_dataloader.dataset.ann_file = './dt/labelTrain.txt'
cfg.train_dataloader.dataset.data_prefix.video = './dt/train/'
cfg.val_dataloader.dataset.ann_file = './dt/labelVal.txt'
cfg.val_dataloader.dataset.data_prefix.video = './dt/val/'


cfg.model.cls_head.num_classes = 2  

cfg.train_dataloader.dataset.pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=2, num_clips=3),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs'),

]

cfg.val_dataloader.dataset.pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=2, num_clips=3),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    # dict(
    #     type='MultiScaleCrop',
    #     input_size=224,
    #     scales=(1, 0.875, 0.75, 0.66),
    #     random_crop=False,
    #     max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    # dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs'),

]

cfg.test_dataloader.dataset.pipeline = test_pipeline = [
    # dict(type='DecordInit'),
    # dict(type='SampleFrames', clip_len=2, frame_interval=1, num_clips=3),
    # dict(
    #     type='SampleFrames',
    #     clip_len=1,
    #     frame_interval=2,
    #     num_clips=3,
    #     test_mode=True),
    # dict(type='DecordDecode'),
    # dict(type='Resize', scale=(-1, 256)),
    # # dict(type='TenCrop', size=224),
    # dict(type='FormatShape', input_format='NCHW'),
    # dict(type='PackActionInputs')
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=2, num_clips=3, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    # dict(type='TenCrop', size=224),  # Se desejar fazer o TenCrop
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs'),

]


# Configurações adicionais
cfg.work_dir = './FallCheckPoints'
cfg.train_dataloader.batch_size = 4
cfg.val_dataloader.batch_size = 4
cfg.test_dataloader.batch_size = 4
cfg.optim_wrapper.optimizer.lr = 0.0005
cfg.train_cfg.max_epochs = 120
cfg.train_dataloader.num_workers = 2
cfg.val_dataloader.num_workers = 2
cfg.test_dataloader.num_workers = 2

# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
# mlflow.set_experiment("Fall Detection")

model = init_recognizer(cfg, checkpoint='https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth')

runner = Runner(
    model=model, 
    work_dir=cfg.work_dir,
    train_dataloader=cfg.train_dataloader,
    test_dataloader=cfg.test_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=cfg.optim_wrapper.optimizer.lr, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=cfg.train_cfg.max_epochs, val_interval=1),
    val_dataloader=cfg.val_dataloader,
    val_cfg=dict(),
    train_evaluator=dict(type=Metrics),
    val_evaluator=dict(type=Metrics),  # Usando a métrica personalizada
    test_cfg=dict(type='TestLoop'),  
    test_evaluator=dict(type=Metrics),
)



runner.train()

# with mlflow.start_run():
#     mlflow.log_param("Batch_size", cfg.train_dataloader.batch_size)
#     mlflow.log_param("Learning_rate", cfg.optim_wrapper.optimizer.lr)
#     mlflow.log_param("Epochs", cfg.train_cfg.max_epochs)

#     mmengine.mkdir_or_exist(osp.abspath(cfg.work_dir))

#     runner.train()

#     results = runner.test()

#     for metric, value in results.items():
#         mlflow.log_metric(metric, value)
        
#     mlflow.pytorch.log_model(runner.model, "model")