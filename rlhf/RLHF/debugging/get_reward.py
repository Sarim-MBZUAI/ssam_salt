import sys 
import os
import cv2
import torch

sys.path.append("/home/abdelrahman.elsayed/RLHF")
from RewardModel import RewardModel

test_dir = "/home/abdelrahman.elsayed/crf_experiments/{'n_iter': 5, 'filter_size': 3, 'smoothness_weight': 0.5}"
r_model = RewardModel()

test_label_dir = os.path.join(test_dir, 'labels')
test_pred_label_dir = os.path.join(test_dir, 'pred_labels')

def load_image_from_path(image_path):
    """Load an image from the given file path."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    return image

def preprocess_image(image):
    """Convert a loaded image (numpy array) to a PyTorch tensor."""
    image = torch.from_numpy(image).float()  # Convert to torch tensor and ensure it's float
    return image  # Shape is now (H, W)

# Loop through the files in the test directories
for f in os.listdir(test_label_dir):
    if f.endswith('png'):
        gt_path = os.path.join(test_label_dir, f)
        pred_path = os.path.join(test_pred_label_dir, f)
        
        # Load images
        gt = load_image_from_path(gt_path)
        pred = load_image_from_path(pred_path)

        # Convert the images to PyTorch tensors and move them to the correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gt_tensor = preprocess_image(gt).to(device)
        pred_tensor = preprocess_image(pred).to(device)

        # Get rewards and save edge images
        save_name = f"example_{os.path.splitext(f)[0]}"  # Use the file name as part of the save name
        reward = r_model(pred_tensor.unsqueeze(0), gt_tensor.unsqueeze(0), save_name=save_name)
        
        # Print the reward for each image
        print(f"Reward for {f}: {reward.item()}")

if __name__ == '__main__':
    print("Calculating rewards and saving edges")
    for f in os.listdir(test_label_dir):
        if f.endswith('png'):
            gt_path = os.path.join(test_label_dir, f)
            pred_path = os.path.join(test_pred_label_dir, f)

            # Load and preprocess the images
            gt = load_image_from_path(gt_path)
            pred = load_image_from_path(pred_path)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gt_tensor = preprocess_image(gt).to(device)
            pred_tensor = preprocess_image(pred).to(device)

            print(gt_tensor.shape)
            print(pred_tensor.shape)

            # Get rewards and save edge images
            save_name = f"example_{os.path.splitext(f)[0]}"
            reward = r_model(pred_tensor.unsqueeze(0), gt_tensor.unsqueeze(0), save_name=save_name)
            
            # Print the reward for each image
            print(f"Reward for {f}: {reward.item()}")
