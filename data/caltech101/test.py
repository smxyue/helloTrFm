def read_binary_file(filepath):
    """
    Open a file as binary and print its content
    
    Args:
        filepath (str): Path to the file to read
    """
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
            print(f"File: {filepath}")
            print(f"Size: {len(content)} bytes")
            print(f"First 100 bytes (in hex): {content[:100].hex()}")
            print(f"First 100 bytes (as string): {content[:100]}")
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")

# Usage
filepath = r"D:\yue\helloTrFm\data\caltech101\101_ObjectCategories\airplanes\image_0009.jpg"
read_binary_file(filepath)