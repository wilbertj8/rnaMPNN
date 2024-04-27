import torch

def load_and_print_pt_file(file_path):
    # Load the tensor or dictionary of tensors from the .pt file
    data = torch.load(file_path, map_location=torch.device('cpu'))
    
    # Check if the data is a dictionary (common for models or complex data)
    if isinstance(data, dict):
        print("Loaded data is a dictionary with the following keys and tensor shapes:")
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f"Key: {key}, Tensor shape: {value.shape}")
                # print actual value
                # print(value)
            else:
                print(f"Key: {key}, Value: {value}")
                if isinstance(value, dict):
                    for k, v in value.items():
                        print(f"Key: {k}, Value: {type(v)}")
    else:
        # If it's just a single tensor
        print("Loaded data is a single tensor:")
        print(data)

if __name__ == "__main__":
    # Specify the path to your .pt file
    pt_file_path = '/Users/wilbertjoseph/Downloads/ProteinMPNN/training/dataset/pdb/1A3M_A.pt'
    
    # Call the function to load and print contents
    load_and_print_pt_file(pt_file_path)
