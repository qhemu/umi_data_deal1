import h5py
import argparse

def print_hdf5_structure(file, indent=0):
    """
    Recursively print the structure of an HDF5 file.

    Args:
        file (h5py.File or h5py.Group): The HDF5 file or group to analyze.
        indent (int): The indentation level (used for nested groups).
    """
    for key in file.keys():
        item = file[key]
        print('  ' * indent + f"{key}: {type(item)}")  # Print the name and type of the item
        if isinstance(item, h5py.Group):
            print_hdf5_structure(item, indent + 1)  # Recursively analyze groups
        elif isinstance(item, h5py.Dataset):
            print('  ' * (indent + 1) + f"Shape: {item.shape}, Dtype: {item.dtype}")  # Print dataset details

def analyze_hdf5_file(filepath):
    """
    Analyze the structure of an HDF5 file and print its contents.

    Args:
        filepath (str): Path to the HDF5 file.
    """
    with h5py.File(filepath, 'r') as file:
        print(f"Analyzing HDF5 file: {filepath}")
        print_hdf5_structure(file)  # Start the recursive analysis

def main():
    """
    Parse command-line arguments and analyze the specified HDF5 file.
    """
    parser = argparse.ArgumentParser(description='Analyze the structure of an HDF5 file.')
    parser.add_argument('filepath', type=str, help='Path to the HDF5 file to analyze.')
    args = parser.parse_args()

    analyze_hdf5_file(args.filepath)  # Call the analysis function with the provided file path

if __name__ == "__main__":
    main()
