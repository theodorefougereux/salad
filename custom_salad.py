import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from tqdm import tqdm
import argparse
import pandas as pd
import os
import numpy as np
from pathlib import Path
from PIL import Image
import faiss
import matplotlib.pyplot as plt
import yaml
import sys

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))


sys.path.append(current_dir)



from vpr_model import VPRModel


class CustomDataset(Dataset):
    def __init__(self, input_transform=None, metadata_path=None, reference_path=None, keyframes_path=None):
        print("retrieving data and metadata from:")
        print(metadata_path, reference_path, keyframes_path)
        self.metadata_df = pd.read_csv(metadata_path)
        
        # Get reference images (database)
        self.dbImages = [f.split('.')[0] for f in os.listdir(reference_path) 
                         if (f.endswith('.jpg') or f.endswith('.png'))]
        # Filter dbImages to ensure they are present in metadata
        self.dbImages = [img for img in self.dbImages 
                         if img in self.metadata_df['image_id'].values]
           
        # Get query images (keyframes from SLAM)
        self.qImages = [f for f in os.listdir(keyframes_path) 
                        if f.endswith('.png') or f.endswith('.jpg')]
        
        # Set up labels for position data
        num_references = len(self.dbImages)
        num_queries = len(self.qImages)
        labels = []
        
        for image_name in self.dbImages:
            row = self.metadata_df[self.metadata_df['image_id'] == image_name]
            labels.append((row['lat'].values[0], row['long'].values[0]))
            
        # Add placeholder coordinates for queries
        for i in range(num_queries):
            labels.append((0, 0))
            
        self.ground_truth = torch.tensor(labels)
        self.input_transform = input_transform
        self.reference_path = reference_path
        self.keyframes_path = keyframes_path
        
        # Combined list: reference images followed by query images
        self.images = self.dbImages + self.qImages
        self.num_references = num_references
        self.num_queries = num_queries
        
        
    def __getitem__(self, index):
        if index < self.num_references:
            # Load reference image
            img_path = os.path.join(self.reference_path, f"{self.images[index]}.jpg")
            img = Image.open(img_path)
        else:
            # Load query image
            img_path = os.path.join(self.keyframes_path, self.images[index])
            img = Image.open(img_path)
            
        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)


def input_transform(image_size=None):
    """Create image transform pipeline"""
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    # check if image size is a pair of integers or an integer
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    if image_size:
        return T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])


def get_top_matches(r_list, q_list, num_queries=10, top_k=5, faiss_gpu=False):
    """Get top matching reference images for each query image using FAISS"""
    print("r_list size:", r_list.shape)
    print("q_list size:", q_list.shape)
    embed_size = r_list.shape[1]
    
    if faiss_gpu:
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = True
        flat_config.device = 0
        faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
    else:
        faiss_index = faiss.IndexFlatL2(embed_size)

    # Add references to index
    faiss_index.add(r_list)

    # Search for queries in the index
    _, predictions = faiss_index.search(q_list, top_k)
    
    return predictions


def display_images(images, titles=None, rows=1, cols=5, figsize=(15, 3), name=None, output_path=None):
    """Helper function to display and save images"""
    cols = min(6, cols)
    plt.figure(figsize=figsize)
    
    for i, image in enumerate(images):
        if i < cols:
            plt.subplot(rows, cols, i + 1)
            plt.imshow(image)
            plt.axis('off')
            if titles:
                plt.title(titles[i])
    
    if name is not None and output_path is not None:
        save_path = os.path.join(output_path, name)
        print(f"Saving at {save_path}")
        plt.savefig(save_path)
        plt.close()

def get_descriptors(model, dataloader, device, num_references=0, descriptor_ckpt_path=None):
    """Calculate or load cached descriptors for the dataset."""
   
    print('num_references:', num_references)    
    print('descriptor_ckpt_path:', descriptor_ckpt_path)
    
    loaded_weights = False
    descriptor_list = []

    if descriptor_ckpt_path and os.path.exists(descriptor_ckpt_path):
        embeddings_references = torch.load(descriptor_ckpt_path)
        loaded_weights = True
        print('Loaded descriptors from cache.')
        print('Cache size:', embeddings_references.shape)
    else:
        embeddings_references = None

    count = 0

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        for imgs, _ in tqdm(dataloader, desc='Calculating descriptors...'):
            batch_size = imgs.size(0)

            if loaded_weights and count + batch_size <= embeddings_references.size(0):
                output = embeddings_references[count:count + batch_size]
            else:
                output = model(imgs.to(device)).cpu()

            descriptor_list.append(output)
            count += batch_size

    descriptors = torch.cat(descriptor_list, dim=0)

    # Save descriptors if needed
    if descriptor_ckpt_path and not loaded_weights:
        torch.save(descriptors[:num_references], descriptor_ckpt_path)

    return descriptors


def load_model(ckpt_path):
    """Load VPR model from checkpoint"""
    model = VPRModel(
        backbone_arch='dinov2_vitb14',
        backbone_config={
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
        },
        agg_arch='SALAD',
        agg_config={
            'num_channels': 768,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        },
    )

    model.load_state_dict(torch.load(ckpt_path))
    model = model.eval()
    model = model.to('cuda')
    print(f"Loaded model from {ckpt_path} successfully!")
    return model


def run_salad(config):
    """Main function to run SALAD with configuration from pipeline"""
    torch.backends.cudnn.benchmark = True
    
    # Extract parameters from config
    ckpt_path = config['model_checkpoint']
    metadata_path = config['metadata_path']
    reference_path = config['reference_path']
    keyframes_path = config['keyframes_path']
    output_path = config['output_path_glob']+ config["name"]
    batch_size = config.get('batch_size', 512)
    image_size = config.get('image_size', 322)
    top_k = config.get('top_k', 20)
    descriptor_ckpt_path = config.get('descriptor_ckpt_path', None)
    print('descriptor_ckpt_path:', descriptor_ckpt_path)
    # check if output_path+match_log.txt is already in the output_path
    if os.path.isfile(os.path.join(output_path, "match_log.txt")):
        print("Match log already exists in output path.")
        return
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
   
    # Load model
    model = load_model(ckpt_path)

    # Prepare dataset and dataloader
    transform = input_transform(image_size=image_size)
    dataset = CustomDataset(
        input_transform=transform,
        metadata_path=metadata_path,
        reference_path=reference_path,
        keyframes_path=keyframes_path
    )
    
    dataloader = DataLoader(
        dataset, 
        num_workers=16, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True
    )
    
    # Get descriptors for all images
    descriptors = get_descriptors(
        model, 
        dataloader, 
        'cuda', 
        num_references=dataset.num_references,
        descriptor_ckpt_path= (descriptor_ckpt_path if descriptor_ckpt_path else None)
    )
    print(f"Number of descriptors: {descriptors.shape[0]}") 
    print(f'Descriptor dimension: {descriptors.shape[1]}')
    
    # Split descriptors into reference and query sets
    r_list = descriptors[:dataset.num_references]
    q_list = descriptors[dataset.num_references:]

    print(f'Total size: {descriptors.shape[0]}, References: {dataset.num_references}, Queries: {dataset.num_queries}')

    # Get top matches
    top_matches = get_top_matches(
        r_list, 
        q_list, 
        num_queries=dataset.num_queries, 
        top_k=top_k
    )
    
    # Display and save top matches for each query
    for query_idx in range(min(top_matches.shape[0], dataset.num_queries)):
        query_image_path = os.path.join(dataset.keyframes_path, dataset.qImages[query_idx])
        query_image = Image.open(query_image_path)
        
        top_images = []
        for match_idx in top_matches[query_idx]:
            if match_idx < len(dataset.dbImages):
                match_image_path = os.path.join(dataset.reference_path, f"{dataset.dbImages[match_idx]}.jpg")
                match_image = Image.open(match_image_path)
                top_images.append(match_image)
        if config['save_examples']:
            display_images(
                [query_image] + top_images[:5],  # Limit to 5 matches for better visualization
                rows=1, 
                cols=min(6, len(top_images) + 1),
                name=f"match_{query_idx}.png",
                output_path=output_path
            )
    
    # Log query and match IDs into a text file
    log_file_path = os.path.join(output_path, "match_log.txt")
    
    with open(log_file_path, 'w') as log_file:
        for query_idx in range(top_matches.shape[0]):
            query_id = dataset.qImages[query_idx]
            match_ids = [dataset.dbImages[match_idx] for match_idx in top_matches[query_idx] 
                        if match_idx < len(dataset.dbImages)]
            log_file.write(f"{query_id}: {', '.join(match_ids)}\n")
    
    print(f"Match log saved to {log_file_path}")
    print(f"Number of references: {dataset.num_references}")
    print(f"Number of queries: {dataset.num_queries}")
    print('========> DONE!\n\n')
    
    # # Return the match results for potential further processing
    # return {
    #     'top_matches': top_matches.cpu().numpy(),
    #     'query_images': dataset.qImages,
    #     'reference_images': dataset.dbImages,
    #     'log_file_path': log_file_path
    # }


def main():
    """Command-line entry point for standalone usage"""
    parser = argparse.ArgumentParser(
        description="Run SALAD for visual place recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the YAML configuration file")
                        
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)['salad']
        
    # Run SALAD with the loaded configuration
    
    run_salad(config)


if __name__ == '__main__':
    main()