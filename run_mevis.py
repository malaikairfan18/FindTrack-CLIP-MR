import sys
import os
sys.path.insert(0, "/kaggle/working/FindTrack-CLIP")

import alphaclip
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from utils import *
import cv2
import json
import numpy as np
from PIL import Image
import torch
import torchvision as tv
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
import warnings
warnings.filterwarnings('ignore')

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation on MeViS dataset with FindTrack-CLIP-MR")
    parser.add_argument("--mode", type=str, default="mask_crop", choices=["mask_crop", "object_box_crop", "full_frame"],
                        help="CLIP Reranker mode")
    parser.add_argument("--w_finder", type=float, default=0.5, help="Finder score weight (w1)")
    parser.add_argument("--w_clip", type=float, default=0.5, help="CLIP score weight (w2)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--dataset_path", type=str, default="../DB/RVOS/MeViS", help="Path to MeViS dataset")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output predictions and force re-evaluation")
    parser.add_argument("--ref_num", type=int, default=10, help="Number of candidate frames to sample")
    parser.add_argument("--num_refs", type=int, default=3, help="Number of reference frames to select for tracking")
    parser.add_argument("--min_distance", type=int, default=15, help="Minimum frame distance for temporal diversity")
    return parser.parse_args()


def test(args):

    # initialize EVF-SAM
    tokenizer, evfsam = init_models()

    # initialize Alpha-CLIP
    clip, clip_preprocess = alphaclip.load('ViT-L/14@336px', alpha_vision_ckpt_pth='weights/clip_l14_336_grit_20m_4xe.pth', device='cuda')
    clip_preprocess_mask = transforms.Compose([transforms.Resize((336, 336)), transforms.Normalize(0.5, 0.26)])

    # initialize Cutie
    cutie = get_default_model(config='mevis_config')
    processor = InferenceCore(cutie, cfg=cutie.cfg)

    # load videos
    output_dir = 'outputs'
    save_path_prefix = os.path.join(output_dir, 'MeViS_val')
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    root = args.dataset_path
    img_folder = os.path.join(root, 'valid', 'JPEGImages')
    meta_file = os.path.join(root, 'valid', 'meta_expressions.json')
    with open(meta_file, 'r') as f:
        data = json.load(f)['videos']
    valid_videos = set(data.keys())
    video_list = sorted([video for video in valid_videos])

    # inference
    for idx_, video in enumerate(video_list):
        metas = []
        expressions = data[video]['expressions']
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        for i in range(num_expressions):
            meta = {}
            meta['video'] = video
            meta['exp'] = expressions[expression_list[i]]['exp']
            meta['exp_id'] = expression_list[i]
            meta['frames'] = data[video]['frames']
            metas.append(meta)
        meta = metas
        video_name = video
        frames = data[video]['frames']
        video_len = len(frames)

        # CHECK IF ALREADY EVALUATED (Resume capability)
        already_done = True
        if args.force:
            already_done = False
        else:
            for e in range(num_expressions):
                exp_id = meta[e]['exp_id']
                save_path = os.path.join(save_path_prefix, video_name, exp_id)
                if not os.path.exists(save_path):
                    already_done = False
                    break
                for frame in frames:
                    if not os.path.exists(os.path.join(save_path, frame + '.png')):
                        already_done = False
                        break
                if not already_done:
                    break
        
        if already_done:
            print(f"Video {idx_+1}/{len(video_list)}: {video} - Already evaluated. Skipping.")
            continue

        print(f"Video {idx_+1}/{len(video_list)}: {video}")

        # input pre-process
        imgs_beit = []
        imgs_sam = []
        imgs_clip = []
        imgs_cutie = []
        for i in range(video_len):
            img_path = os.path.join(img_folder, video_name, frames[i] + '.jpg')
            image_np = cv2.imread(img_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]

            # BEiT pre-process
            img_beit = beit3_preprocess(Image.open(img_path), 224)
            imgs_beit.append(img_beit)

            # SAM pre-process
            img_sam, resize_shape = sam_preprocess(image_np)
            imgs_sam.append(img_sam)

            # Alpha-CLIP pre-process
            img_clip = clip_preprocess(Image.open(img_path))
            imgs_clip.append(img_clip)

            # Cutie pre-process
            img_cutie = tv.transforms.ToTensor()(Image.open(img_path))
            imgs_cutie.append(img_cutie)

        # for each language
        for e in range(num_expressions):

            # make files
            video_name = meta[e]['video']
            exp = meta[e]['exp']
            exp_id = meta[e]['exp_id']
            frames = meta[e]['frames']
            save_path = os.path.join(save_path_prefix, video_name, exp_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # per-frame mask prediction and scoring
            ref_masks = []
            ref_scores_finder = []
            ref_scores_clip = []
            
            ref_num = args.ref_num
            print(f"  Exp: '{exp}'")
            candidate_indices = []
            
            for ref_idx in range(ref_num):
                i = int(ref_idx * (video_len - 1) / (ref_num - 1))
                candidate_indices.append(i)
                words = tokenizer(exp, return_tensors='pt')['input_ids'].cuda()
                ref_mask, ref_score = evfsam.inference(imgs_sam[i].unsqueeze(0).cuda(), imgs_beit[i].unsqueeze(0).cuda(), words, resize_shape, original_size_list)
                ref_mask = (ref_mask > 0).float()
                ref_masks.append(ref_mask)

                # extract finder score (segmentation confidence)
                evf_val = ref_score.item() if hasattr(ref_score, 'item') else float(ref_score)
                ref_scores_finder.append(evf_val)

                # consider vision-text alignment in addition to segmentation confidence
                clip_text = alphaclip.tokenize([exp]).cuda()
                ref_img_path = os.path.join(img_folder, video_name, frames[i] + '.jpg')
                image_np = cv2.imread(ref_img_path)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

                clip_sim = compute_clip_similarity(
                    clip, clip_preprocess, clip_preprocess_mask,
                    image_np, ref_mask, clip_text, mode=args.mode
                )
                clip_val = clip_sim.item() if hasattr(clip_sim, 'item') else float(clip_sim)
                ref_scores_clip.append(clip_val)

            # Min-Max Normalization to solve score scale dominance
            finder_min, finder_max = min(ref_scores_finder), max(ref_scores_finder)
            clip_min, clip_max = min(ref_scores_clip), max(ref_scores_clip)
            
            finder_range = finder_max - finder_min + 1e-6
            clip_range = clip_max - clip_min + 1e-6
            
            normalized_finder = [(s - finder_min) / finder_range for s in ref_scores_finder]
            normalized_clip = [(s - clip_min) / clip_range for s in ref_scores_clip]
            
            w1, w2 = args.w_finder, args.w_clip
            combined_scores = []
            for i in range(ref_num):
                score = w1 * normalized_finder[i] + w2 * normalized_clip[i]
                combined_scores.append(score)
                print(f"    Frame {frames[candidate_indices[i]]} (idx {candidate_indices[i]:02d}): "
                      f"EVF-SAM={ref_scores_finder[i]:.4f} (Norm={normalized_finder[i]:.4f}), "
                      f"CLIP={ref_scores_clip[i]:.4f} (Norm={normalized_clip[i]:.4f}), "
                      f"Combined={score:.4f}")

            # Temporal Diversity Filter for Top-K Reference Selection
            sorted_indices = np.argsort(combined_scores)[::-1]
            selected_candidate_indices = []
            
            for idx in sorted_indices:
                if len(selected_candidate_indices) >= args.num_refs:
                    break
                current_frame_pos = candidate_indices[idx]
                diverse = True
                for sel_idx in selected_candidate_indices:
                    sel_frame_pos = candidate_indices[sel_idx]
                    if abs(current_frame_pos - sel_frame_pos) < args.min_distance:
                        diverse = False
                        break
                if diverse:
                    selected_candidate_indices.append(idx)
            
            # Fallback if we couldn't find enough diverse references
            if len(selected_candidate_indices) < args.num_refs:
                for idx in sorted_indices:
                    if len(selected_candidate_indices) >= args.num_refs:
                        break
                    if idx not in selected_candidate_indices:
                        selected_candidate_indices.append(idx)

            # Sort selected references chronologically
            selected_candidate_indices.sort()
            selected_refs = [candidate_indices[idx] for idx in selected_candidate_indices]
            earliest_ref_idx = selected_refs[0]
            earliest_candidate_idx = selected_candidate_indices[0]
            
            print("  => Selected Reference Frames:")
            for idx in selected_candidate_indices:
                f_idx = candidate_indices[idx]
                print(f"     Frame {frames[f_idx]} (idx {f_idx:02d}) with Combined Score: {combined_scores[idx]:.4f}")

            # forward pass
            for i in range(earliest_ref_idx, video_len):
                if i in selected_refs:
                    ref_list_idx = selected_refs.index(i)
                    cand_idx = selected_candidate_indices[ref_list_idx]
                    mask_prob = processor.step(imgs_cutie[i].cuda(), ref_masks[cand_idx].squeeze(0), objects=[1])
                else:
                    mask_prob = processor.step(imgs_cutie[i].cuda())
                mask = processor.output_prob_to_mask(mask_prob).float()

                # clear memory for each sequence
                if i == video_len - 1:
                    processor.clear_memory()

                # convert format
                mask = mask.detach().cpu().numpy().astype(np.float32)
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frames[i] + '.png')
                mask.save(save_file)

            # backward pass
            for i in range(earliest_ref_idx, -1, -1):
                if i == earliest_ref_idx:
                    cand_idx = earliest_candidate_idx
                    mask_prob = processor.step(imgs_cutie[i].cuda(), ref_masks[cand_idx].squeeze(0), objects=[1])
                else:
                    mask_prob = processor.step(imgs_cutie[i].cuda())
                mask = processor.output_prob_to_mask(mask_prob).float()

                # clear memory for each sequence
                if i == 0:
                    processor.clear_memory()

                # convert format
                mask = mask.detach().cpu().numpy().astype(np.float32)
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frames[i] + '.png')
                mask.save(save_file)


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        test(args)
