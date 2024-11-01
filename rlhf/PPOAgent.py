import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.distributions import Categorical
import sys
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
    def __init__(self, pretrained_model, model_config, label_dict, reward_model, learning_rate=1e-5):
        self.policy = pretrained_model
        self.policy = self.policy.to(device)
        self.reward_model = reward_model
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.old_policy = Prompt_Adapted_SAM(model_config, label_dict, training_strategy='svdtuning')
        self.old_policy.load_state_dict(pretrained_model.state_dict())
        self.old_policy = self.old_policy.to(device)
        
    def debug_tensor(self, tensor, name):
        print(f"\nDebugging {name}:")
        print(f"Shape: {tensor.shape}")
        print(f"Contains NaN: {torch.isnan(tensor).any()}")
        if torch.any(~torch.isnan(tensor)):
            print(f"Min value: {tensor[~torch.isnan(tensor)].min()}")
            print(f"Max value: {tensor[~torch.isnan(tensor)].max()}")
            print(f"Mean value: {tensor[~torch.isnan(tensor)].mean()}")
        else:
            print("All values are NaN")

    def process_model_output(self, model_output, label):
        # Extract logits from model output
        logits = model_output[0]
        
        # Debug the raw output
        self.debug_tensor(logits, "Raw model output")
        
        if torch.isnan(logits).all():
            print("WARNING: Model producing all NaN outputs. Checking model parameters...")
            has_nan = False
            for name, param in self.policy.named_parameters():
                if torch.isnan(param).any():
                    print(f"Found NaN in parameter: {name}")
                    has_nan = True
            if has_nan:
                raise ValueError("Model parameters contain NaN values. Model needs to be reinitialized.")
        
        
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        
        self.debug_tensor(logits, "Processed model output")
        return logits

    def compute_policy_distributions(self, images, label):
        # Forward pass
        model_output = self.policy(images, label)
        
        # Process the output
        logits = self.process_model_output(model_output, label)
        
        self.debug_tensor(logits, "Final probabilities")
        return logits

    def compute_log_probs(self, probs, actions):
        if probs.dim() == 4:
            probs = probs.squeeze(1)
        
        if actions.dim() == 4 :
            actions = actions.squeeze(1)
        
        actions = actions.float()
        
        distribution = Bernoulli(probs) # Bernouli since we are working with binary stuff
        log_probs = distribution.log_prob(actions)
        
        self.debug_tensor(log_probs, "Log probabilities")
        return log_probs

    def train_step(self, images, ground_truth, label, epoch,batch_size=5):
        try:
            with torch.no_grad():
                images = images.to(device)
                ground_truth = ground_truth.to(device)
                
                # Get probabilities from old policy
                model_output = self.old_policy(images, label)
                logits = self.process_model_output(model_output, label)
                
                ground_truth = (ground_truth >= 0.5).float()
                ground_truth = ground_truth.unsqueeze(1)
                self.debug_tensor(ground_truth, "Ground truth")
                
                old_log_probs = self.compute_log_probs(logits, ground_truth)
                
                predicted_segs = (logits >= 0.5).float()
                self.debug_tensor(predicted_segs,"Predicted Segs")
                self.debug_tensor(ground_truth , "Ground Truth")
                rewards = self.reward_model(predicted_segs.squeeze(1), ground_truth.squeeze(1), save_name=epoch)
                advantages = rewards.detach()
                
                self.debug_tensor(advantages, "Computed advantages")

            loss = self.ppo_update(images, label, ground_truth, old_log_probs, advantages)
            self.old_policy.load_state_dict(self.policy.state_dict())
            return loss
            
        except Exception as e:
            print(f"Error in train_step: {str(e)}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
            raise e

    def ppo_update(self, images, label, actions, old_log_probs, advantages, 
                   epsilon=0.2, value_clip=0.2, c1=1, c2=0.01):
        try:
            current_probs = self.compute_policy_distributions(images, label)
            current_log_probs = self.compute_log_probs(current_probs, actions)
            
            ratio = torch.exp(current_log_probs - old_log_probs)
            self.debug_tensor(ratio, "Probability ratio")
            
            advantages = advantages.unsqueeze(-1).unsqueeze(-1)
            self.debug_tensor(advantages, "Reshaped advantages") # to make it match the shape [B,H,W] ? Need to check this again
            
            # Compute surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            entropy = Bernoulli(current_probs).entropy().mean() # again Bernoulli for the binary stuff
            
            total_loss = policy_loss - c2 * entropy
            self.debug_tensor(total_loss, "Total loss")
            
            self.optimizer.zero_grad()
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            return total_loss.item()
            
        except Exception as e:
            print(f"Error in ppo_update: {str(e)}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
            raise e

def train_rlhf(pretrained_model, model_config, label_dict, reward_model, train_loader, num_epochs):
    rlhf_trainer = SegmentationRLHF(pretrained_model, model_config, label_dict, reward_model)
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
        print(f"Epoch {epoch} finished with average loss: {epoch_loss / len(train_loader):.4f}")
