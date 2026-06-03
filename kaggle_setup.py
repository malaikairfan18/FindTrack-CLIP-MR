import os
import subprocess
import sys
import glob
import json
import shutil

def link_datasets():
    print("=== Setting up Combined Datasets ===")
    
    # Clean up any old combined directories if they exist, to ensure a fresh, correct link
    for path in ["/kaggle/working/MeViS_combined", "/kaggle/working/YTVOS_combined"]:
        if os.path.exists(path):
            print(f"Cleaning old directory {path}...")
            shutil.rmtree(path)
            
    def get_actual_video_dirs(root_paths):
        video_dirs = []
        for root_path in root_paths:
            if not os.path.exists(root_path):
                continue
            for root, dirs, files in os.walk(root_path):
                # Check if this directory contains any .jpg files (a leaf video directory)
                if any(f.lower().endswith('.jpg') for f in files):
                    video_dirs.append(root)
        return video_dirs

    # 1. Combined MeViS
    mevis_combined = "/kaggle/working/MeViS_combined"
    os.makedirs(os.path.join(mevis_combined, "valid", "JPEGImages"), exist_ok=True)
    
    # Find all MeViS valid paths in /kaggle/input (must contain 'mevis' and not 'combined' or 'yt/vos/rvos')
    mevis_jpeg_inputs = glob.glob("/kaggle/input/**/valid/JPEGImages", recursive=True)
    mevis_jpeg_inputs = [p for p in mevis_jpeg_inputs if "mevis" in p.lower() and "combined" not in p.lower() and not ("yt" in p.lower() or "vos" in p.lower() or "rvos" in p.lower())]
    mevis_meta_inputs = glob.glob("/kaggle/input/**/valid/meta_expressions.json", recursive=True)
    mevis_meta_inputs = [p for p in mevis_meta_inputs if "mevis" in p.lower() and "combined" not in p.lower() and not ("yt" in p.lower() or "vos" in p.lower() or "rvos" in p.lower())]
    
    print(f"Found MeViS JPEG folders: {mevis_jpeg_inputs}")
    print(f"Found MeViS meta files: {mevis_meta_inputs}")
    
    # Find actual video directories recursively under the found MeViS JPEG folders
    actual_mevis_dirs = get_actual_video_dirs(mevis_jpeg_inputs)
    print(f"Found {len(actual_mevis_dirs)} actual MeViS video directories. Linking...")
    
    for src in actual_mevis_dirs:
        video_name = os.path.basename(src)
        dst = os.path.join(mevis_combined, "valid", "JPEGImages", video_name)
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except Exception as e:
                print(f"Failed to symlink {src} to {dst}: {e}")
    
    # Merge meta_expressions.json
    if mevis_meta_inputs:
        merged_meta = {"videos": {}}
        for meta_path in mevis_meta_inputs:
            try:
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)
                if "videos" in meta_data:
                    merged_meta["videos"].update(meta_data["videos"])
            except Exception as e:
                print(f"Error reading meta file {meta_path}: {e}")
        
        dst_meta = os.path.join(mevis_combined, "valid", "meta_expressions.json")
        with open(dst_meta, 'w') as f:
            json.dump(merged_meta, f)
        print(f"Created merged MeViS meta file at {dst_meta} with {len(merged_meta['videos'])} videos.")

    # 2. Combined YTVOS
    ytvos_combined = "/kaggle/working/YTVOS_combined"
    os.makedirs(os.path.join(ytvos_combined, "valid", "JPEGImages"), exist_ok=True)
    os.makedirs(os.path.join(ytvos_combined, "meta_expressions", "valid"), exist_ok=True)
    os.makedirs(os.path.join(ytvos_combined, "meta_expressions", "test"), exist_ok=True)
    
    ytvos_jpeg_inputs = glob.glob("/kaggle/input/**/valid/JPEGImages", recursive=True)
    # Filter for ytvos specifically (must contain 'yt', 'vos', or 'rvos' and not contain 'mevis')
    ytvos_jpeg_inputs = [p for p in ytvos_jpeg_inputs if ("yt" in p.lower() or "vos" in p.lower() or "rvos" in p.lower()) and "mevis" not in p.lower() and "combined" not in p.lower()]
    
    ytvos_meta_valid = glob.glob("/kaggle/input/**/meta_expressions/valid/meta_expressions.json", recursive=True)
    ytvos_meta_valid = [p for p in ytvos_meta_valid if "combined" not in p.lower() and "mevis" not in p.lower()]
    
    ytvos_meta_test = glob.glob("/kaggle/input/**/meta_expressions/test/meta_expressions.json", recursive=True)
    ytvos_meta_test = [p for p in ytvos_meta_test if "combined" not in p.lower() and "mevis" not in p.lower()]
    
    print(f"Found YTVOS JPEG folders: {ytvos_jpeg_inputs}")
    print(f"Found YTVOS valid meta files: {ytvos_meta_valid}")
    print(f"Found YTVOS test meta files: {ytvos_meta_test}")
    
    # Find actual video directories recursively under the found YTVOS JPEG folders
    actual_ytvos_dirs = get_actual_video_dirs(ytvos_jpeg_inputs)
    print(f"Found {len(actual_ytvos_dirs)} actual YTVOS video directories. Linking...")
    
    for src in actual_ytvos_dirs:
        video_name = os.path.basename(src)
        dst = os.path.join(ytvos_combined, "valid", "JPEGImages", video_name)
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except Exception as e:
                print(f"Failed to symlink {src} to {dst}: {e}")
                            
    # Copy valid meta
    if ytvos_meta_valid:
        dst_valid = os.path.join(ytvos_combined, "meta_expressions", "valid", "meta_expressions.json")
        try:
            shutil.copy(ytvos_meta_valid[0], dst_valid)
            print(f"Copied YTVOS valid meta file to {dst_valid}")
        except Exception as e:
            print(f"Failed to copy YTVOS valid meta: {e}")
            
    # Copy test meta
    if ytvos_meta_test:
        dst_test = os.path.join(ytvos_combined, "meta_expressions", "test", "meta_expressions.json")
        try:
            shutil.copy(ytvos_meta_test[0], dst_test)
            print(f"Copied YTVOS test meta file to {dst_test}")
        except Exception as e:
            print(f"Failed to copy YTVOS test meta: {e}")
            
    print("=== Dataset Setup Complete ===")

def setup():
    print("=== Installing dependencies ===")
    # Loose version installation to prevent conflicts with Kaggle's pre-installed packages
    packages = [
        "loralib",
        "ftfy",
        "omegaconf",
        "hydra-core",
        "torchscale",
        "simpletransformers",
        "accelerate",
        "gradio",
        "imageio",
        "timm"
    ]
    for pkg in packages:
        print(f"Installing {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg])

    print("=== Installing gdown ===")
    subprocess.run([sys.executable, "-m", "pip", "install", "gdown"])

    print("=== Creating weights/ directory ===")
    os.makedirs("weights", exist_ok=True)

    print("=== Downloading Alpha-CLIP weights via gdown ===")
    file_id = "1dG_j98hh7AFvhSADlhp9CpoNY-9rBHoc"
    output_path = "weights/clip_l14_336_grit_20m_4xe.pth"
    
    if os.path.exists(output_path):
        print(f"{output_path} already exists. Skipping download.")
    else:
        import gdown
        gdown.download(id=file_id, output=output_path, quiet=False)
        print("=== Alpha-CLIP weights downloaded successfully ===")

    # Set up datasets
    link_datasets()

if __name__ == '__main__':
    setup()

