import torch
from mmseg.apis import init_segmentor, inference_segmentor
from backbone import convnext

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def do_inference(config_file, checkpoint_file):
    """
    Does inference on a single image
    """
    # Initiate the model
    model = init_segmentor(config_file, checkpoint_file)
    
    # Make dummy data
    data = dict()
    
    with torch.no_grad():
        result = model(return_loss=False, rescale=False, **data)
    
    print(f"Result shape: {result.shape}")
    
if __name__ == "__main__":
    config_file_path = input("Enter config file path: ")
    checkpoint_file_path = input("Enter checkpoint file path: ")
    
    do_inference(config_file_path, checkpoint_file_path)