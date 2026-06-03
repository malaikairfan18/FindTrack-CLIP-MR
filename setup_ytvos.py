import os
import shutil
import sys

def setup_ytvos():
    print("=== Setting up Ref-YouTube-VOS Dataset (Fast Setup) ===")
    
    # 1. Target directory
    ytvos_combined = "/kaggle/working/YTVOS_combined"
    if os.path.exists(ytvos_combined):
        print(f"Cleaning old combined directory {ytvos_combined}...")
        shutil.rmtree(ytvos_combined)
        
    os.makedirs(os.path.join(ytvos_combined, "valid", "JPEGImages"), exist_ok=True)
    os.makedirs(os.path.join(ytvos_combined, "meta_expressions", "valid"), exist_ok=True)
    os.makedirs(os.path.join(ytvos_combined, "meta_expressions", "test"), exist_ok=True)

    # 2. Discover the dataset root path
    search_root = "/kaggle/input"
    skip_dirs = {'jpegimages', 'valid', 'train', 'test', 'valid_sam', 'weights', 'outputs'}
    
    found_roots = []
    for root, dirs, files in os.walk(search_root):
        dirs[:] = [d for d in dirs if d.lower() not in skip_dirs]
        
        has_valid = os.path.exists(os.path.join(root, 'valid'))
        has_meta = os.path.exists(os.path.join(root, 'meta_exp')) or os.path.exists(os.path.join(root, 'meta_expressions'))
        
        path_lower = root.lower()
        is_ytvos = any(k in path_lower for k in ['yt', 'vos', 'rvos'])
        
        if is_ytvos and has_valid and has_meta:
            found_roots.append(root)
            dirs[:] = []  # stop descending

    if not found_roots:
        print("ERROR: Could not find any Ref-YouTube-VOS dataset root in /kaggle/input.")
        print("Please make sure the dataset is attached to your Kaggle notebook.")
        return

    dataset_root = found_roots[0]
    print(f"Found Ref-YouTube-VOS dataset root at: {dataset_root}")

    # 3. Link Video JPEGs (JPEGImages)
    # Check if root has valid/JPEGImages
    valid_jpeg_path = os.path.join(dataset_root, 'valid', 'JPEGImages')
    if not os.path.exists(valid_jpeg_path):
        # Fallback if valid contains JPEGImages directly or is named slightly differently
        valid_jpeg_path = os.path.join(dataset_root, 'valid')
    
    print(f"Linking video directories from: {valid_jpeg_path}")
    
    if os.path.exists(valid_jpeg_path):
        video_dirs = [d for d in os.listdir(valid_jpeg_path) if os.path.isdir(os.path.join(valid_jpeg_path, d))]
        linked_count = 0
        for v in video_dirs:
            src = os.path.join(valid_jpeg_path, v)
            dst = os.path.join(ytvos_combined, "valid", "JPEGImages", v)
            if not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                    linked_count += 1
                except Exception as e:
                    print(f"Failed to symlink {src} to {dst}: {e}")
        print(f"Successfully symlinked {linked_count} video directories.")
    else:
        print(f"ERROR: Could not find JPEGImages directory in {valid_jpeg_path}")

    # 4. Copy Metadata Files
    meta_valid_src = None
    meta_test_src = None
    
    for meta_folder in ['meta_expressions', 'meta_exp']:
        v_path = os.path.join(dataset_root, meta_folder, 'valid', 'meta_expressions.json')
        t_path = os.path.join(dataset_root, meta_folder, 'test', 'meta_expressions.json')
        if os.path.exists(v_path):
            meta_valid_src = v_path
        if os.path.exists(t_path):
            meta_test_src = t_path
            
    if meta_valid_src:
        dst_valid = os.path.join(ytvos_combined, "meta_expressions", "valid", "meta_expressions.json")
        try:
            shutil.copy(meta_valid_src, dst_valid)
            print(f"Copied valid meta to {dst_valid}")
        except Exception as e:
            print(f"Failed to copy valid meta: {e}")
    else:
        print("WARNING: Could not find valid meta_expressions.json")

    if meta_test_src:
        dst_test = os.path.join(ytvos_combined, "meta_expressions", "test", "meta_expressions.json")
        try:
            shutil.copy(meta_test_src, dst_test)
            print(f"Copied test meta to {dst_test}")
        except Exception as e:
            print(f"Failed to copy test meta: {e}")
    else:
        print("WARNING: Could not find test meta_expressions.json")

    print("=== Ref-YouTube-VOS Setup Complete (in ~1 second) ===")

if __name__ == '__main__':
    setup_ytvos()
