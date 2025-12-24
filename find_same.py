import torch
import os

def no_train_diff():
    file1 = "vit_b_16_caltech101.pth"
    file2 = "vit_b_16_caltech101_none_train.pth"
    #file2="vit_b_16-c867db91.pth"
    for file in [file1, file2]:
        if not os.path.exists(file):
            print(f"File {file} does not exist.")
            return
    
    try:
        # Load both model files
        model1 = torch.load(file1, map_location='cpu')
        model2 = torch.load(file2, map_location='cpu')
        
        # Check if both are state dictionaries
        if not isinstance(model1, dict) or not isinstance(model2, dict):
            print("Files are not in dictionary format (likely not state dicts)")
            return
            
        # Get all unique keys from both models
        all_keys = set(model1.keys()).union(set(model2.keys()))
        
        print(f"Comparing {len(all_keys)} keys between the two models\n")
        
        different_keys = 0
        same_keys = 0
        
        for key in sorted(all_keys):
            exists_in_1 = key in model1
            exists_in_2 = key in model2
            
            if not exists_in_1:
                print(f"Key '{key}' only exists in file2")
                different_keys += 1
                continue

            if (not exists_in_1) and (not exists_in_2):
                print("="*80)
                
            if not exists_in_2:
                print(f"Key '{key}' only exists in file1")
                different_keys += 1
                continue
                
            # Both keys exist, compare values
            tensor1 = model1[key]
            tensor2 = model2[key]
            
            # Compare tensor shapes
            if tensor1.shape != tensor2.shape:
                print(f"Key '{key}': Different shapes - {tensor1.shape} vs {tensor2.shape}")
                different_keys += 1
                continue
                
            # Compare tensor values
            if torch.equal(tensor1, tensor2):
                print(f"Key '{key}': SAME (shape: {tuple(tensor1.shape)}, size: {tensor1.numel()})")
                same_keys += 1
            else:
                print(f"Key '{key}': DIFFERENT (shape: {tuple(tensor1.shape)}, size: {tensor1.numel()})")
                different_keys += 1
        
        print(f"\nSummary: {same_keys} keys are the same, {different_keys} keys are different")
        
    except Exception as e:
        print(f"Error processing model files: {e}")
        print("Falling back to binary comparison...")
        
        # Fallback to binary comparison if torch.load fails
        with open(file2, 'rb') as f1, open(file1, 'rb') as f2:
            data1 = f1.read()
            data2 = f2.read()

            start = -1
            length = 0
            # Find difference positions
            min_len = min(len(data1), len(data2))
            for i in range(min_len):
                b1, b2 = data1[i], data2[i]
                if b1 != b2:
                    if start != -1:
                        if length > 1:
                            print(f"Difference segment: position {start}-{start+length-1}, length: {length}")
                        start = -1
                        length = 0
                elif b1 == b2 and start == -1:
                    start = i
                    length = 1
                elif b1 == b2 and start != -1:
                    length += 1
            # Handle the last segment if it's the same
            if start != -1 and length != 0:
                print(f"Same segment: position {start}-{start+length-1}, length: {length}")
                
            # Handle remaining bytes if files have different lengths
            if len(data1) != len(data2):
                longer_file = file1 if len(data1) > len(data2) else file2
                diff_start = min_len
                diff_length = abs(len(data1) - len(data2))
                print(f"Extra data in {longer_file}: position {diff_start}, length: {diff_length}")

if __name__ == "__main__":
    no_train_diff()
                
