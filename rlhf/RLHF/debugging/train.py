import sys
sys.path.append("/home/abdelrahman.elsayed/RLHF")
sys.path.append("/home/abdelrahman.elsayed/sarim_code/datasets")
sys.path.append("/home/abdelrahman.elsayed/sarim_code")
from model import *
from PPOAgent import SegmentationRLHF , train_rlhf
from RewardModel import RewardModel
import pandas as pd
import torch
import yaml
import torch.utils
from arcade import ArcadeDataset

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

def get_data(data_config , model_config , target):
    data_split_csv_path = data_config["data"]["data_split_csv"]
    data_split = pd.read_csv(data_split_csv_path)

    dataset_dict = {}
    dataloader_dict = {}

    use_norm = True
    no_text_mode = False

    for split in ["train", "val"]:
        # Filter the CSV for the current split
        split_data = data_split[data_split["split"] == split]["imgs"].tolist()

        # Pass the filtered data to the dataset class (ArcadeDataset)
        dataset_dict[split] = ArcadeDataset(
            config=data_config,
            file_list=split_data,  # Pass file_list as (image_path, mask_path) tuples
            shuffle_list=True,
            is_train=(split == "train"),
            apply_norm=use_norm,
            no_text_mode=no_text_mode,
        )

        # Create DataLoader for each dataset
        dataloader_dict[split] = torch.utils.data.DataLoader(
            dataset_dict[split],
            batch_size=model_config["training"]["batch_size"],
            shuffle=True,
            num_workers=4,
        )

    # Get dataset sizes
    dataset_sizes = {split: len(dataset_dict[split]) for split in ["train", "val"]}

    # Create label dictionary
    label_dict = {
        name: i for i, name in enumerate(data_config["data"]["label_names"])
    }

    # Print dataset sizes
    print(f"Train dataset size: {dataset_sizes['train']}")
    print(f"Val dataset size: {dataset_sizes['val']}")

    # Get dataset sizes
    dataset_sizes = {split: len(dataset_dict[split]) for split in ["train", "val"]}

# Create label dictionary
    label_dict = {
        name: i for i, name in enumerate(data_config["data"]["label_names"])
    }

    # Print dataset sizes
    print(f"Train dataset size: {dataset_sizes['train']}")
    print(f"Val dataset size: {dataset_sizes['val']}")
    
    return dataloader_dict[target] , label_dict


def load_yaml(path):
    with open(path, 'r') as file :
       config = yaml.safe_load(file)
    return config

# first of all we load the configs
model_config = load_yaml("/home/abdelrahman.elsayed/sarim_code/model_svdtuning.yml")
data_config = load_yaml("/home/abdelrahman.elsayed/sarim_code/config_arcade.yml")

dataloader,label_dict = get_data(data_config , model_config , "train")

#load the model
model_config['use_lora'] = False
model = Prompt_Adapted_SAM(model_config , label_dict , training_strategy='svdtuning')
model.load_state_dict(torch.load('/home/abdelrahman.elsayed/modelsDIAS/final_model.pth', map_location=device))

reward_model = RewardModel()
train_rlhf(model ,model_config,label_dict, reward_model , dataloader,10)
