import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os

""""
Things to do : 
1- Add more metrics maybe ? not dice only maybe HD95 
"""
class RewardModel(nn.Module):
    def __init__(self , save_dir):
        super(RewardModel, self).__init__()
        self.canny_threshold1 = 100
        self.canny_threshold2 = 200
        self.save_dir = save_dir 
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
    def tensor_to_numpy(self, tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        array = tensor.numpy()
        # Ensure values are in uint8 range [0, 255]
        if array.max() <= 1.0:
            array = (array * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
        return array

    def detect_edges(self, mask, save_name=None, is_predicted=False):
        """
        Detect edges in a batch of masks and optionally save them as images.
        mask: tensor of shape [B, H, W] or [H, W]
        save_name: base name for saving the images (if provided)
        is_predicted: flag to indicate if the mask is predicted or ground truth
        """
        if len(mask.shape) == 2:  # Single mask
            mask_np = self.tensor_to_numpy(mask)
            edges = cv2.Canny(mask_np, self.canny_threshold1, self.canny_threshold2)
            edges_tensor = torch.from_numpy(edges).to(mask.device)
            
            if save_name:
                file_type = "predicted" if is_predicted else "ground_truth"
                save_path = os.path.join(self.save_dir, f"{save_name}_{file_type}_edges.png")
                cv2.imwrite(save_path, edges)
            
            return edges_tensor
        else:  # Batch of masks
            edges_batch = []
            for i, single_mask in enumerate(mask):
                mask_np = self.tensor_to_numpy(single_mask)
                edges = cv2.Canny(mask_np, self.canny_threshold1, self.canny_threshold2)
                edges_batch.append(torch.from_numpy(edges))
                
                if save_name:
                    file_type = "predicted" if is_predicted else "ground_truth"
                    save_path = os.path.join(self.save_dir, f"{save_name}_{i}_{file_type}_edges.png")
                    cv2.imwrite(save_path, edges)
            
            return torch.stack(edges_batch).to(mask.device)

    def compute_reward(self, predicted_edges, ground_truth_edges):
        """
        Compute reward based on edge overlap
        """
        predicted_edges = predicted_edges.float()
        ground_truth_edges = ground_truth_edges.float()
        
        intersection = (predicted_edges * ground_truth_edges).sum(dim=[-2, -1])
        union = predicted_edges.sum(dim=[-2, -1]) + ground_truth_edges.sum(dim=[-2, -1])
        
        reward = 2 * intersection / (union + 1e-5)
        return reward

    def forward(self, predicted_masks, ground_truth_masks, save_name=None):
        """
        Forward pass of the reward model
        predicted_masks: tensor of shape [B, H, W]
        ground_truth_masks: tensor of shape [B, H, W]
        Returns: tensor of shape [B] containing rewards
        """
        device = predicted_masks.device
        
        predicted_edges = self.detect_edges(predicted_masks, save_name=save_name, is_predicted=True)
        ground_truth_edges = self.detect_edges(ground_truth_masks, save_name=save_name, is_predicted=False)
        
        predicted_edges = predicted_edges.to(device)
        ground_truth_edges = ground_truth_edges.to(device)
        
        rewards = self.compute_reward(predicted_edges, ground_truth_edges)
        return rewards
