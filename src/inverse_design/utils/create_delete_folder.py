import os
import shutil


def create_numbered_folders(base_name="input", count=180):
    # Create folders from 1 to count
    for i in range(1, count + 1):
        folder_name = f"{base_name}_{i}"
        try:
            os.makedirs(folder_name)
            print(f"Created folder: {folder_name}")
        except FileExistsError:
            print(f"Folder already exists: {folder_name}")


def remove_numbered_folders(base_name="input", count=512):
    """Remove numbered folders from 1 to count."""
    for i in range(1, count + 1):
        folder_name = f"{base_name}_{i}"
        try:
            if os.path.exists(folder_name):
                shutil.rmtree(folder_name)
                print(f"Removed folder: {folder_name}")
            else:
                print(f"Folder does not exist: {folder_name}")
        except Exception as e:
            print(f"Error removing folder {folder_name}: {e}")


if __name__ == "__main__":
    # Example usage:
    # Create folders
    if 0:
        create_numbered_folders()
        print("Folder creation completed!")
    if 1:
        remove_numbered_folders()
        print("Folder removal completed!")
