# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import cv2
import numpy as np
import torch.nn.functional as F
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
import torch.nn.functional as F

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_atten: float = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_atten = cost_atten
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, out_atten, atten_loss_type):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        #================================================ difference ===================================================== #
        # Decompose the gaze attention
        if out_atten != None:
            W, H = out_atten.size()[2], out_atten.size()[3]
            device = out_atten.device

            tgt_tar_hmap = torch.cat([F.interpolate(v['target-gaze_heatmap'].unsqueeze(1), size=(W, H), mode='bilinear', align_corners=False).squeeze(1) for v in targets])
            tgt_go_hmap = torch.cat([F.interpolate(v['gaze-only_heatmap'].unsqueeze(1), size=(W, H), mode='bilinear', align_corners=False).squeeze(1)for v in targets])
            N_tar, N_go = tgt_tar_hmap.size()[0], tgt_go_hmap.size()[0]
            # normalize attention
            min_vals = torch.min(out_atten)  
            max_vals = torch.max(out_atten)  
            out_atten = (out_atten - min_vals) / (max_vals - min_vals + 1e-6)
            if len(tgt_tar_hmap) != 0:  
                min_vals= torch.min(tgt_tar_hmap)  
                max_vals = torch.max(tgt_tar_hmap)  
                tgt_tar_hmap = (tgt_tar_hmap - min_vals) / (max_vals - min_vals + 1e-6) 
            if len(tgt_go_hmap) != 0:  
                min_vals = torch.min(tgt_go_hmap) 
                max_vals = torch.max(tgt_go_hmap)  
                tgt_go_hmap = (tgt_go_hmap - min_vals) / (max_vals - min_vals + 1e-6) 

            if atten_loss_type == 'l1':
                cost_tgt_atten = torch.cdist(out_atten.view(bs * num_queries, W * H), tgt_tar_hmap.view(N_tar, W * H).float(), p=1)
                cost_go_atten = torch.cdist(out_atten.view(bs * num_queries, W * H), tgt_go_hmap.view(N_go, W * H).float(), p=1)

            elif atten_loss_type == 'mse':
                out_atten_tgt = out_atten.view(bs * num_queries, W , H).unsqueeze(1).expand(bs * num_queries, N_tar, W, H)
                out_atten_go = out_atten.view(bs * num_queries, W , H).unsqueeze(1).expand(bs * num_queries, N_go, W, H)

                tgt_tar_hmap_exp = tgt_tar_hmap.unsqueeze(0).expand(bs * num_queries, N_tar, W, H)
                tgt_go_hmap_exp = tgt_go_hmap.unsqueeze(0).expand(bs * num_queries, N_go, W, H)

                out_atten_tgt_mask = out_atten_tgt > 1e-4
                out_atten_go_mask = out_atten_go > 1e-4
                tgt_tar_hmap_mask = tgt_tar_hmap_exp > 1e-4
                tgt_go_hmap_mask = tgt_go_hmap_exp > 1e-4

                tgt_mask = out_atten_tgt_mask | tgt_tar_hmap_mask
                go_mask = out_atten_go_mask | tgt_go_hmap_mask

                tgt_square_error = (out_atten_tgt - tgt_tar_hmap_exp)**2
                go_square_error = (out_atten_go - tgt_go_hmap_exp)**2

                cost_tgt_atten = tgt_square_error.sum(dim=(2,3)) / tgt_mask.sum(dim=(2,3))
                cost_go_atten = go_square_error.sum(dim=(2,3)) / go_mask.sum(dim=(2,3))

            elif atten_loss_type == 'mask':
                out_atten_tgt = out_atten.view(bs * num_queries, W , H).unsqueeze(1).expand(bs * num_queries, N_tar, W, H)
                out_atten_go = out_atten.view(bs * num_queries, W , H).unsqueeze(1).expand(bs * num_queries, N_go, W, H)

                tgt_tar_hmap_exp = tgt_tar_hmap.unsqueeze(0).expand(bs * num_queries, N_tar, W, H)
                tgt_go_hmap_exp = tgt_go_hmap.unsqueeze(0).expand(bs * num_queries, N_go, W, H)

                out_atten_tgt_mask = out_atten_tgt > 1e-4
                out_atten_go_mask = out_atten_go > 1e-4
                tgt_tar_hmap_mask = tgt_tar_hmap_exp > 1e-4
                tgt_go_hmap_mask = tgt_go_hmap_exp > 1e-4

                tgt_intersection = torch.sum(out_atten_tgt_mask * tgt_tar_hmap_mask, dim=(2, 3)) 
                tgt_union = torch.sum(out_atten_tgt_mask, dim=(2, 3)) + torch.sum(tgt_tar_hmap_mask, dim=(2, 3))
                tgt_dice_coeff = (2 * tgt_intersection + 1e-4) / (tgt_union + 1e-4)

                go_intersection = torch.sum(out_atten_go_mask * tgt_go_hmap_mask, dim=(2, 3)) 
                go_union = torch.sum(out_atten_go_mask, dim=(2, 3)) + torch.sum(tgt_go_hmap_mask, dim=(2, 3))
                go_dice_coeff = (2 * go_intersection + 1e-4) / (go_union + 1e-4)

                cost_tgt_atten = 1 - tgt_dice_coeff
                cost_go_atten = 1- go_dice_coeff

            elif atten_loss_type == 'cosine':
                cost_tgt_atten = 10 * -F.cosine_similarity(out_atten.view(bs * num_queries, -1).unsqueeze(1), tgt_tar_hmap.view(N_tar, W * H).unsqueeze(0), dim=2)
                cost_go_atten = 10 * -F.cosine_similarity(out_atten.view(bs * num_queries, -1).unsqueeze(1), tgt_go_hmap.view(N_go, W * H).unsqueeze(0), dim=2)
                
            elif atten_loss_type == 'kl':
                kl_tgt_atten = F.kl_div(torch.log(out_atten.view(bs * num_queries, -1).unsqueeze(1) + 1e-8), tgt_tar_hmap.view(N_tar, W * H).unsqueeze(0) + 1e-8, reduction='none')
                kl_go_atten = F.kl_div(torch.log(out_atten.view(bs * num_queries, -1).unsqueeze(1) + 1e-8), tgt_go_hmap.view(N_go, W * H).unsqueeze(0) + 1e-8, reduction='none')
                cost_tgt_atten = kl_tgt_atten.sum(dim=2) 
                cost_go_atten = kl_go_atten.sum(dim=2)

            if cost_tgt_atten.numel() != 0:
                sum_attention = torch.sum(tgt_tar_hmap.view(N_tar, W * H), dim=1, keepdim=True) 
                mask = sum_attention != 0  
                cost_tgt_atten *= mask.T
            

            tgt_sizes = [len(v["boxes"]) for v in targets]
            go_sizes = [len(v["gaze-only_heatmap"]) for v in targets]

            cost_bbox_new = []
            index = 0
            for i in range(bs):
                cost_bbox_new.append(cost_bbox[:, index : index + tgt_sizes[i]])
                cost_bbox_new.append(cost_bbox.max() * torch.ones((bs * num_queries, go_sizes[i])).to(device)) if cost_bbox.numel() != 0 else cost_bbox_new.append(torch.ones((bs * num_queries, go_sizes[i])).to(device)) 
                index += tgt_sizes[i]
            cost_bbox = torch.cat(cost_bbox_new, dim=1)

            cost_class_new = []
            index = 0
            for i in range(bs):
                cost_class_new.append(cost_class[:, index : index + tgt_sizes[i]])
                cost_class_new.append(cost_class.max() * torch.ones((bs * num_queries, go_sizes[i])).to(device)) if cost_class.numel() != 0 else cost_class_new.append(torch.ones((bs * num_queries, go_sizes[i])).to(device)) 
                index += tgt_sizes[i]
            cost_class = torch.cat(cost_class_new, dim=1)

            cost_giou_new = []
            index = 0
            for i in range(bs):
                cost_giou_new.append(cost_giou[:, index : index + tgt_sizes[i]])
                cost_giou_new.append(cost_giou.max() * torch.ones((bs * num_queries, go_sizes[i])).to(device)) if cost_giou.numel() != 0 else cost_giou_new.append(torch.ones((bs * num_queries, go_sizes[i])).to(device)) 
                index += tgt_sizes[i]
            cost_giou = torch.cat(cost_giou_new, dim=1)
            
            cost_atten = []
            tgt_index = 0
            go_index = 0
            for i in range(bs):
                cost_atten.append(cost_tgt_atten[:, tgt_index : tgt_index + tgt_sizes[i]])
                cost_atten.append(cost_go_atten[:, go_index : go_index + go_sizes[i]])
                tgt_index += tgt_sizes[i]
                go_index += go_sizes[i]
            cost_atten = torch.cat(cost_atten, dim=1)
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + self.cost_atten * cost_atten
            C = C.view(bs, num_queries, -1).cpu()

            all_sizes = [len(v["boxes"]) + len(v["gaze-only_heatmap"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(all_sizes, -1))]
            tgt_list = []
            go_list = []
            bs = 0
            for i_array, j_array in indices:
                i_tgt_l, j_tgt_l, i_go_l, j_go_l= [], [], [], []
                for i, j in zip(i_array, j_array):
                    if j < tgt_sizes[bs]:
                        i_tgt_l.append(i)
                        j_tgt_l.append(j)
                    else:
                        i_go_l.append(i)
                        j_go_l.append(j - tgt_sizes[bs])
                tgt_list.append((torch.as_tensor(i_tgt_l, dtype=torch.int64), torch.as_tensor(j_tgt_l, dtype=torch.int64)))
                go_list.append((torch.as_tensor(i_go_l, dtype=torch.int64), torch.as_tensor(j_go_l, dtype=torch.int64)))
                bs += 1
            # print('add attention:',tgt_list,go_list)
            return tgt_list, go_list
        #============================================end difference ===================================================== #

        else:
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], []

        

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou, cost_atten= args.set_cost_atten)
