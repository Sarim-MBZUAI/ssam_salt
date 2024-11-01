import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os
from PIL import Image
import numpy as np
sys.path.append("/home/abdelrahman.elsayed/sarim_code")
from model import *
from torch.distributions import Bernoulli
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


"""
Things to consider TO DO:
1- Advantage calculation is just now the rewards , update this
2- Should the Reward Model's output be normalized or something?
"""

class SegmentationRLHF:
    def __init__(self, pretrained_model, model_config, label_dict, reward_model, learning_rate=1e-4):
        self.policy = pretrained_model
        self.policy = self.policy.to(device)
        self.reward_model = reward_model
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, weight_decay=0.01)
        self.old_policy = Prompt_Adapted_SAM(model_config, label_dict, training_strategy='svdtuning')
        self.old_policy.load_state_dict(pretrained_model.state_dict())
        self.old_policy = self.old_policy.to(device)
        
        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Add gradient clipping
        self.max_grad_norm = 1.0
        
    def compute_policy_distributions(self, images, label):
        model_output = self.policy(images, label)
        logits = self.process_model_output(model_output, label)
        return F.sigmoid(logits)  # Apply sigmoid to get probabilities

    def compute_log_probs(self, probs, actions):
        if probs.dim() == 4:
            probs = probs.squeeze(1)
        if actions.dim() == 4:
            actions = actions.squeeze(1)
        
        actions = actions.float()
        distribution = Bernoulli(probs)
        log_probs = distribution.log_prob(actions)
        return log_probs

    def train_step(self, images, ground_truth, label, epoch, batch_size=5):
        self.policy.train()
        try:
            images = images.to(device)
            ground_truth = ground_truth.to(device)
            
            # Get probabilities from old policy
            with torch.no_grad():
                old_probs = self.compute_policy_distributions(images, label)
                
            # Process ground truth
            ground_truth = (ground_truth >= 0.5).float()
            ground_truth = ground_truth.unsqueeze(1)
            
            # Compute log probabilities for old policy
            old_log_probs = self.compute_log_probs(old_probs, ground_truth)
            
            # Compute rewards
            with torch.no_grad():
                predicted_segs = (old_probs >= 0.5).float()
                rewards = self.reward_model(predicted_segs.squeeze(1), ground_truth.squeeze(1), save_name=epoch)
                advantages = rewards.detach() - rewards.mean()  # Subtract baseline
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages
            
            # PPO update
            loss = self.ppo_update(images, label, ground_truth, old_log_probs, advantages)
            return loss
            
        except Exception as e:
            print(f"Error in train_step: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

    def ppo_update(self, images, label, actions, old_log_probs, advantages, 
                   epsilon=0.2, value_clip=0.2, c1=1, c2=0.01):
        try:
            # Get current policy distributions
            current_probs = self.compute_policy_distributions(images, label)
            current_log_probs = self.compute_log_probs(current_probs, actions)
            
            # Compute probability ratio
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            # Ensure advantages have correct shape
            advantages = advantages.unsqueeze(-1).unsqueeze(-1)
            
            # Compute surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute entropy for exploration
            entropy = Bernoulli(current_probs).entropy().mean()
            
            # Compute total loss
            total_loss = policy_loss - c2 * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
            
            self.optimizer.step()
            
            return total_loss.item()
            
        except Exception as e:
            print(f"Error in ppo_update: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

def train_rlhf(pretrained_model, model_config, label_dict, reward_model, train_loader, val_loader, num_epochs):
    rlhf_trainer = SegmentationRLHF(pretrained_model, model_config, label_dict, reward_model)
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        for batch_idx, (images, ground_truth, img_path, label) in enumerate(train_loader):
            loss = rlhf_trainer.train_step(images, ground_truth, label, epoch)
            epoch_loss += loss
            pbar.set_postfix({'loss': epoch_loss / (batch_idx + 1)})
            pbar.update(1)
        
        pbar.close()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} finished with average loss: {avg_loss:.4f}")
        
        # Validation step
        val_loss, val_dice = rlhf_trainer.validate(val_loader, epoch, save_dir="DIAS_val_images")
        
        # Learning rate scheduling
        rlhf_trainer.scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(rlhf_trainer.policy.state_dict(), 'best_model.pth')
        
        # Update old policy after each epoch
        rlhf_trainer.old_policy.load_state_dict(rlhf_trainer.policy.state_dict())
    
    print("Final Validation")
    rlhf_trainer.validate(val_loader , 1 , save_dir="DIAS_val_images")
    return rlhf_trainer.policy

    
    