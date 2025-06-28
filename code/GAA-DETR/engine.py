# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from util.misc import NestedTensor 

import cv2
import numpy as np
from util.box_ops import box_cxcywh_to_xyxy

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, atten_loss_type: str = 'l1'):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) if not isinstance(v, str) else v for k, v in t.items()} for t in targets]
        # ========================================= difference ================================================= # 
        # prepare to get query attention
        for name, parameters in model.named_parameters():
            if name == 'query_embed.weight':
                pq = parameters
            if name == 'transformer.decoder.layers.5.multihead_attn.in_proj_weight':
                in_proj_weight = parameters
            if name == 'transformer.decoder.layers.5.multihead_attn.in_proj_bias':
                in_proj_bias = parameters
        
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        cq = []  
        pk = []  
        memory = [] 
 
        hooks = [
            model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            model.transformer.encoder.register_forward_hook(
                lambda self, input, output: memory.append(output)
            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
            model.transformer.decoder.layers[-1].norm1.register_forward_hook(
                lambda self, input, output: cq.append(output)
            ),
            model.backbone[-1].register_forward_hook(
                lambda self, input, output: pk.append(output)
            ),
        ]

        outputs = model(samples)
       
       # get query attention
        
        for hook in hooks:
            hook.remove()
        
        conv_features = conv_features[0]  
        enc_attn_weights = enc_attn_weights[0] 
        dec_attn_weights = dec_attn_weights[0]
        # memory = memory[0] 
        # cq = cq[0]  
        # pk = pk[0] 
        # batch_size= pk.size()[0]
        # pk = pk.flatten(-2).permute(2, 0, 1) 
        # pq = pq.unsqueeze(1).repeat(1, 1, 1)  
        
        # q = pq + cq  
        # # q = pq  
        # # ------------------------------------------------------#
        # #   1) k = pk，则可视化： (cq + oq)*pk
        # #   2_ k = pk + memory，则可视化 (cq + oq)*(memory + pk)
        # #   读者可自行尝试
        # # ------------------------------------------------------#
        # # k = pk
        # # k = memory
        # k = pk + memory
        # # ------------------------------------------------------#
 
        # _b = in_proj_bias
        # _start = 0
        # _end = 256
        # _w = in_proj_weight[_start:_end, :]
        # if _b is not None:
        #     _b = _b[_start:_end]
        # q = torch.nn.functional.linear(q, _w, _b)
 
        # _b = in_proj_bias
        # _start = 256
        # _end = 256 * 2
        # _w = in_proj_weight[_start:_end, :]
        # if _b is not None:
        #     _b = _b[_start:_end]
        # k = torch.nn.functional.linear(k, _w, _b)
 
        # scaling = float(256) ** -0.5
        # q = q * scaling
        # q = q.contiguous().view(batch_size, 100, 8, 32).transpose(1, 2) 
        # v_dim = k.size()[0]
        # k = k.contiguous().view(batch_size, v_dim, 8, 32).transpose(1, 2).transpose(-2,-1)
        # attn_output_weights = torch.matmul(q, k)
 
        # attn_output_weights = attn_output_weights.view(batch_size, 8, 100, v_dim)
        # attn_output_weights = attn_output_weights.view(batch_size * 8, 100, v_dim)
        # attn_output_weights = torch.nn.functional.softmax(attn_output_weights, dim=-1)
        # attn_output_weights = attn_output_weights.view(batch_size, 8, 100, v_dim)
        
        h, w = conv_features['0'].tensors.shape[-2:]
        B, n = dec_attn_weights.shape[0],dec_attn_weights.shape[1]
        model_atten = dec_attn_weights.reshape(B, n, h, w)

        # different loss function
        loss_dict = criterion(outputs, targets, model_atten, atten_loss_type) 
        # ========================================= difference end ============================================= # 
       
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, atten = False, atten_loss_type = 'l1'):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    if atten == False:
        for samples, targets in metric_logger.log_every(data_loader, 10, header):
            samples = samples.to(device)
            targets = [{k: v.to(device) if not isinstance(v, str) else v for k, v in t.items()} for t in targets]

            outputs = model(samples)
            loss_dict = criterion(outputs, targets, None, 'l1')
            weight_dict = criterion.weight_dict

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                                **loss_dict_reduced_scaled,
                                **loss_dict_reduced_unscaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            if 'segm' in postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            if coco_evaluator is not None:
                coco_evaluator.update(res)

            if panoptic_evaluator is not None:
                res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
                for i, target in enumerate(targets):
                    image_id = target["image_id"].item()
                    file_name = f"{image_id:012d}.png"
                    res_pano[i]["image_id"] = image_id
                    res_pano[i]["file_name"] = file_name

                panoptic_evaluator.update(res_pano)

    else:
        for samples, targets in metric_logger.log_every(data_loader, 10, header):
            samples = samples.to(device)
            targets = [{k: v.to(device) if not isinstance(v, str) else v for k, v in t.items()} for t in targets]
            for name, parameters in model.named_parameters():
                if name == 'query_embed.weight':
                    pq = parameters
                if name == 'transformer.decoder.layers.5.multihead_attn.in_proj_weight':
                    in_proj_weight = parameters
                if name == 'transformer.decoder.layers.5.multihead_attn.in_proj_bias':
                    in_proj_bias = parameters
            
            conv_features, enc_attn_weights, dec_attn_weights = [], [], []
            cq = []  
            pk = []  
            memory = [] 
    
            hooks = [
                model.backbone[-2].register_forward_hook(
                    lambda self, input, output: conv_features.append(output)
                ),
                model.transformer.encoder.register_forward_hook(
                    lambda self, input, output: memory.append(output)
                ),
                model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                    lambda self, input, output: enc_attn_weights.append(output[1])
                ),
                model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                    lambda self, input, output: dec_attn_weights.append(output[1])
                ),
                model.transformer.decoder.layers[-1].norm1.register_forward_hook(
                    lambda self, input, output: cq.append(output)
                ),
                model.backbone[-1].register_forward_hook(
                    lambda self, input, output: pk.append(output)
                ),
            ]

            outputs = model(samples)
        
            for hook in hooks:
                hook.remove()
            
            conv_features = conv_features[0]  
            enc_attn_weights = enc_attn_weights[0] 
            dec_attn_weights = dec_attn_weights[0]
            
            h, w = conv_features['0'].tensors.shape[-2:]
            B, n = dec_attn_weights.shape[0],dec_attn_weights.shape[1]
            model_atten = dec_attn_weights.reshape(B, n, h, w)

            probas = outputs['pred_logits'].softmax(-1)[:, :, :-1]
            keep = probas.max(-1).values > 0.5
            max_idx = probas.max(-1).values.argmax(-1)

            summed_attention_map = model_atten.sum(axis=1)
            print(summed_attention_map.size())
            for i in range(B):
                pred_bboxs = outputs['pred_boxes'][i][keep[i]]
                pred_attens = model_atten[i][keep[i]]
                img_file = targets[i]['image_file']
                img = cv2.imread(img_file)
                W, H = img.shape[0], img.shape[1]
                
                bboxs = targets[i]['boxes']
                target_heatmap = targets[i]['target-gaze_heatmap']
                go_heatmap = targets[i]['gaze-only_heatmap']
                expert_all_atten = target_heatmap.sum(axis=0) + go_heatmap.sum(axis=0)
                print(expert_all_atten.size())
                expert_all_atten = torch.nn.functional.interpolate(expert_all_atten.unsqueeze(0).unsqueeze(0), size=(w, h), mode='bilinear', align_corners=False)
                print(expert_all_atten.size())
                expert_all_atten = expert_all_atten.squeeze().cpu().numpy()
                print(expert_all_atten.shape)
                expert_all_atten = cv2.resize(expert_all_atten, (W, H), interpolation=cv2.INTER_LINEAR)
                print(expert_all_atten.shape)
                expert_all_atten = cv2.normalize(expert_all_atten, None, 0, 255, cv2.NORM_MINMAX)
                expert_all_atten = cv2.applyColorMap(expert_all_atten.astype(np.uint8), cv2.COLORMAP_JET)
                alpha = 0.5
                combined_image = cv2.addWeighted(img, 1 - alpha, expert_all_atten, alpha, 0)
                cv2.imwrite('C:\\Users\\kongyan\\Desktop\\experiment\\Gaze-DETR+\\ours\\DETR+Gaze_Detection\\10\\visualiaztion\\image_atten_expert\\val_test\\'+ img_file.split('\\')[-1], combined_image)

                for j in range(len(bboxs)):
                    bbox = bboxs[j]
                    pred_box = (box_cxcywh_to_xyxy(bbox)).cpu().numpy()
                    cv2.rectangle(combined_image, (int(pred_box[0] * W), int(pred_box[1] * H)), (int(pred_box[2] * W), int(pred_box[3] * H)), (255,0,0), 3)
                cv2.imwrite('C:\\Users\\kongyan\\Desktop\\experiment\\Gaze-DETR+\\ours\\DETR+Gaze_Detection\\10\\visualiaztion\\image_atten_expert\\val_test\\'+ img_file.split('\\')[-1].split('.')[0]+'_withbox'+'.jpg', combined_image)

                # if len(pred_bboxs) == 0:
                #     pred_atten = model_atten[i][max_idx[i]].cpu().numpy()
                #     pred_atten = cv2.resize(pred_atten, (W, H), interpolation=cv2.INTER_LINEAR)
                #     pred_atten = cv2.normalize(pred_atten, None, 0, 255, cv2.NORM_MINMAX)
                #     pred_atten = cv2.applyColorMap(pred_atten.astype(np.uint8), cv2.COLORMAP_JET)
                #     alpha = 0.5
                #     combined_image = cv2.addWeighted(img, 1 - alpha, pred_atten, alpha, 0)
                #     # bbox = outputs['pred_boxes'][i][max_idx[i]]
                #     # pred_box = (box_cxcywh_to_xyxy(bbox)).cpu().numpy()
                #     # cv2.rectangle(combined_image, (int(pred_box[0] * w), int(pred_box[1] * h)), (int(pred_box[2] * w), int(pred_box[3] * h)), (255,0,0), 3)
                #     cv2.imwrite('C:\\Users\\kongyan\\Desktop\\experiment\\Gaze-DETR+\\ours\\DETR+Gaze_Detection\\10\\visualiaztion\\model_pred_go_atten\\val_test\\'+ img_file.split('\\')[-1], combined_image)
                
                # else:
                #     for j in range(len(pred_bboxs)):
                #         pred_atten = pred_attens[j].cpu().numpy()
                #         pred_atten = cv2.resize(pred_atten, (W, H), interpolation=cv2.INTER_LINEAR)
                #         pred_atten = cv2.normalize(pred_atten, None, 0, 255, cv2.NORM_MINMAX)
                #         pred_atten = cv2.applyColorMap(pred_atten.astype(np.uint8), cv2.COLORMAP_JET)
                #         alpha = 0.5
                #         combined_image = cv2.addWeighted(img, 1 - alpha, pred_atten, alpha, 0)
                #         bbox = pred_bboxs[j]
                #         pred_box = (box_cxcywh_to_xyxy(bbox)).cpu().numpy()
                #         cv2.rectangle(combined_image, (int(pred_box[0] * W), int(pred_box[1] * H)), (int(pred_box[2] * W), int(pred_box[3] * H)), (255,0,0), 3)
                #         cv2.imwrite('C:\\Users\\kongyan\\Desktop\\experiment\\Gaze-DETR+\\ours\\DETR+Gaze_Detection\\10\\visualiaztion\\model_pred_atten\\val_test\\'+ img_file.split('\\')[-1].split('.')[0]+'_'+str(j)+'.jpg', combined_image)
                
                # all_atten = summed_attention_map[i].cpu().numpy()
                # all_atten = cv2.resize(all_atten, (W, H), interpolation=cv2.INTER_LINEAR)
                # all_atten = cv2.normalize(all_atten, None, 0, 255, cv2.NORM_MINMAX)
                # all_atten = cv2.applyColorMap(all_atten.astype(np.uint8), cv2.COLORMAP_JET)
                # alpha = 0.5
                # combined_image = cv2.addWeighted(img, 1 - alpha, all_atten, alpha, 0)
                # cv2.imwrite('C:\\Users\\kongyan\\Desktop\\experiment\\Gaze-DETR+\\ours\\DETR+Gaze_Detection\\10\\visualiaztion\\image_atten\\val_test\\'+ img_file.split('\\')[-1], combined_image)

                # for j in range(len(pred_bboxs)):
                #     bbox = pred_bboxs[j]
                #     pred_box = (box_cxcywh_to_xyxy(bbox)).cpu().numpy()
                #     cv2.rectangle(combined_image, (int(pred_box[0] * W), int(pred_box[1] * H)), (int(pred_box[2] * W), int(pred_box[3] * H)), (255,0,0), 3)
                # cv2.imwrite('C:\\Users\\kongyan\\Desktop\\experiment\\Gaze-DETR+\\ours\\DETR+Gaze_Detection\\10\\visualiaztion\\image_atten\\val_test\\'+ img_file.split('\\')[-1].split('.')[0]+'_withbox'+'.jpg', combined_image)


            # # convert boxes from [0; 1] to image scales
            # print(outputs['pred_boxes'].size())  # [1, 100, 4]
            # bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
            # print('检测一张图像所需的时间：{}s'.format(time.time() - start_time))
            # # =============================================== #
    
            # # 检测结果可视化
            # scores = probas[keep]
            # plot_results(im, scores, bboxes_scaled)

            # different loss function
            loss_dict = criterion(outputs, targets, model_atten, atten_loss_type, if_save=True)
            weight_dict = criterion.weight_dict

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                                **loss_dict_reduced_scaled,
                                **loss_dict_reduced_unscaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            if 'segm' in postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            if coco_evaluator is not None:
                coco_evaluator.update(res)

            if panoptic_evaluator is not None:
                res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
                for i, target in enumerate(targets):
                    image_id = target["image_id"].item()
                    file_name = f"{image_id:012d}.png"
                    res_pano[i]["image_id"] = image_id
                    res_pano[i]["file_name"] = file_name

                panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
