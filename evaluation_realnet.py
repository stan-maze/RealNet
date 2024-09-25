import warnings
import argparse
import torch
from datasets.data_builder import build_dataloader
from easydict import EasyDict
import yaml
import os
from utils.misc_helper import set_seed
from models.model_helper import ModelHelper
from utils.eval_helper import performances
from sklearn.metrics import precision_recall_curve
import numpy as np
from utils.visualize import export_segment_images
from utils.eval_helper import Report
from train_realnet import update_config
from utils.categories import Categories
from datetime import datetime
import tabulate


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description="evaluation RealNet")
parser.add_argument("--config", default="experiments/{}/realnet.yaml")
parser.add_argument("--dataset", default="LEISI_V2",choices=['LEISI_V2','MVTec-AD','VisA','MPDD','BTAD'])
parser.add_argument("--class_name", default="LEISI_V2",choices=[
        # mvtec-ad
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
        # visa
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
        #mpdd
        "bracket_black",
        "bracket_brown",
        "bracket_white",
        "connector",
        "metal_plate",
        "tubes",
        # btad
         "01",
         "02",
         "03",
        #  LEISI_V2
        'LEISI_V2'
        ] )


def main(test_file=None, export = True, ckpt = 'ckpt_best.pth.tar'):
    args = parser.parse_args()

    class_name_list=Categories[args.dataset]

    assert args.class_name in class_name_list

    args.config=args.config.format(args.dataset)

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.exp_path = os.path.dirname(args.config)

    args.checkpoints_folder = os.path.join(config.exp_path, config.saver.checkpoints_dir,args.class_name)

    print(f'ckpt: {ckpt}')
    args.model_path=os.path.join(args.checkpoints_folder,ckpt)

    config=update_config(config,args)
    if test_file:
        print(f'testing: ./data/LEISI_V2/samples/{test_file}')
        config.dataset.test.meta_file = f'./data/LEISI_V2/samples/{test_file}'
        config.vis_path = os.path.join(config.exp_path, config.saver.vis_dir, test_file[:2])
    else:
        print(f'testing: {config.dataset.test.meta_file}')
        config.vis_path = os.path.join(config.exp_path, config.saver.vis_dir)


    set_seed(config.random_seed)

    config.evaluator.metrics['auc'].append({'name':'pro'})

    os.makedirs(config.vis_path, exist_ok=True)

    _, val_loader = build_dataloader(config.dataset,distributed=False)

    model = ModelHelper(config.net)
    model.cpu()

    state_dict=torch.load(args.model_path)
    model.load_state_dict(state_dict['state_dict'],strict=False)

    ret_metrics = validate(config,val_loader, model,args.class_name, export = export )
    print_metrics(ret_metrics, config.evaluator.metrics, args.class_name)


def print_metrics(ret_metrics, config, class_name):
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = list(clsnames - set(["mean"]))
    clsnames.sort()

    if config.get("auc", None):
        auc_keys = [k for k in ret_metrics.keys() if "auc" in k]
        evalnames = list(set([k.rsplit("_", 2)[1] for k in auc_keys]))
        evalnames.sort()

        record = Report(["clsname"] + evalnames)

        for clsname in clsnames:
            clsvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for evalname in evalnames
            ]
            record.add_one_record([clsname] + clsvalues)

        print(f"\n{record}")



def validate(config,val_loader, model,class_name, export = True):

    model.eval()

    fileinfos = []
    preds = []
    masks = []

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            # forward
            outputs = model(input,train=False)

            for j in range(len(outputs['filename'])):
                fileinfos.append(
                    {
                        "filename": str(outputs["filename"][j]),
                        "height": int(outputs["height"][j]),
                        "width": int(outputs["width"][j]),
                        "clsname": str(outputs["clsname"][j]),
                    }
                )
            preds.append(outputs["anomaly_score"].cpu().numpy())
            masks.append(outputs["mask"].cpu().numpy())


    print(f'prediction finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


    preds = np.squeeze(np.concatenate(np.asarray(preds), axis=0),axis=1)  # N x H x W
    masks = np.squeeze(np.concatenate(np.asarray(masks), axis=0),axis=1)  # N x H x W

    ret_metrics = performances(class_name, preds, masks, config.evaluator.metrics)

    print(f'image F1 cacu finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    # if not export:
    #     return ret_metrics


    preds_cls = []
    masks_cls = []
    image_paths = []

    for fileinfo, pred, mask in zip(fileinfos, preds, masks):
        preds_cls.append(pred[None, ...])
        masks_cls.append(mask[None, ...])
        image_paths.append(fileinfo['filename'])

    preds_cls = np.concatenate(np.asarray(preds_cls), axis=0)  # N x H x W
    masks_cls = np.concatenate(np.asarray(masks_cls), axis=0)  # N x H x W
    masks_cls[masks_cls != 0.0] = 1.0

    # pixel level
    # precision, recall, thresholds = precision_recall_curve(masks_cls.flatten(), preds_cls.flatten())
    # a = 2 * precision * recall
    # b = precision + recall
    # f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    # seg_threshold = thresholds[np.argmax(f1)]
    # print(f1[np.argmax(f1)])
    # print(precision[np.argmax(f1)])
    # print(recall[np.argmax(f1)])
    # print(f'seg_threshold: {seg_threshold}')

    # image level
    N, _, _ = masks.shape
    image_masks_cls = (masks_cls.reshape(N, -1).sum(axis=1) != 0).astype(np.int)
    image_preds_cls = preds_cls.reshape(N, -1).max(axis=1)
    precision, recall, thresholds = precision_recall_curve(image_masks_cls, image_preds_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = 2
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.where(b != 0, a / b, 0)

        a = (1 + beta**2) * precision * recall
        b = (beta**2) * precision + recall
        fb = np.where(b != 0, a / b, 0)

    max_fb_index = np.argmax(fb)
    seg_threshold = thresholds[max_fb_index]
    # seg_threshold = np.max(image_preds_cls[image_preds_cls < seg_threshold])

    print(tabulate.tabulate([[fb[max_fb_index], f1[max_fb_index], precision[max_fb_index], recall[max_fb_index]]], 
                    headers=["Fβ", "F1", "Precision", "Recall"], tablefmt="pretty"))
    print(f'seg_threshold: {seg_threshold}')


    print(f'caculation finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    export_segment_images(config, image_paths, masks_cls, preds_cls, seg_threshold, class_name)
    print(f'export finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    return ret_metrics


if __name__ == "__main__":
    ckpt = '256_nowbest_0.75_ckpt_best.pth.tar'
    # print('='*50)
    # main(export = False)
    # print('='*50)
    # main(test_file = f'tmp.json', ckpt = ckpt)
    print('='*50)
    main(test_file = f'ez_test.json', ckpt = ckpt)
    print('='*50)
    main(test_file = f'hd_test.json', ckpt = ckpt)
    print('='*50)

