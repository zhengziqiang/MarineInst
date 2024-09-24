import os.path
import time
import pycocotools.mask as mask_util
import numpy as np
import cv2
import json
import glob
import os
import argparse
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, help="model type", choices=['vit_h', 'vit_l', 'vit_b'])
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to the checkpoint")
    parser.add_argument("--iou_threshold", type=float, required=True, help="the iou_threshol")
    parser.add_argument("--sta_threshold", type=float, required=True, help="the iou_threshol")
    parser.add_argument("--test_img_path", type=str, required=True, help="the test image path")
    parser.add_argument("--output_dir", type=str, required=True, help="path to save the model")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--gpu", type=int, default=0, help="batch size")
    parser.add_argument("--image_size", type=int, default=1024, help="image size")
    device = "cuda"
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sam_checkpoint = args.checkpoint_path
    model_type = args.model_type
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=args.iou_threshold,
        stability_score_thresh=args.sta_threshold,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    starr_time=time.time()
    for files in sorted(glob.glob(os.path.join(args.test_img_path,"*.*"))):
        image = cv2.imread(files)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[0], image.shape[1]
        p,n=os.path.split(files)
        print(n)
        try:
            masks = mask_generator.generate(image)
        except:
            continue

        print(len(masks))
        output_json = {}
        img_json = {}
        img_json['image_id'] = 0
        img_json['width'] = width
        img_json['height'] = height
        img_json['file_name'] = n
        output_json['image'] = img_json
        out_anno = []
        anno_id = 0
        for tmp in masks:
            anno_json = {}
            seg = tmp['segmentation']
            fortran_ground_truth_binary_mask = np.asfortranarray(seg)
            compressed_rle = mask_util.encode(fortran_ground_truth_binary_mask)
            compressed_rle['counts'] = str(compressed_rle['counts'], encoding="utf-8")
            anno_json['segmentation'] = compressed_rle
            anno_json['bbox'] = tmp['bbox']
            anno_json['area'] = tmp['area']
            anno_json['bbox'] = tmp['bbox']
            anno_json['predicted_iou'] = tmp['predicted_iou']
            anno_json['crop_box'] = tmp['crop_box']
            anno_json['stability_score'] = tmp['stability_score']
            anno_json['point_coords'] = tmp['point_coords']
            anno_json['cate_preds'] = tmp['cate_preds'] # 0 for non-instance masks; 1 for instance masks
            anno_json['id'] = anno_id
            out_anno.append(anno_json)
            anno_id += 1
        output_json['annotations'] = out_anno
        with open(os.path.join(args.output_dir, n[:-4] + ".json"), "w") as fp:
            json.dump(output_json, fp)

if __name__ == "__main__":
    main()
