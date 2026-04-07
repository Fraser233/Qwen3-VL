import os
import csv
import pandas as pd
import numpy as np
import sys
from typing import Dict, Any
from common_utils import md5, toliststr, decode_base64_to_image_file

MMMU_DATASET_URL = 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv'
MMMU_DATASET_MD5 = '521afc0f3bf341e6654327792781644d'

def load_dataset(dataset_name='MMMU_DEV_VAL'):
    """Load the MMMU dataset."""
    if 'LMUData' not in os.environ:
        raise ValueError("LMUData is not set. Pass --data-dir to run_mmmu.py")
    data_root = str(os.environ['LMUData']).strip()
    if not data_root:
        raise ValueError("LMUData is empty. Pass a non-empty --data-dir to run_mmmu.py")
    data_root = os.path.abspath(data_root)
    os.makedirs(data_root, exist_ok=True)
    
    file_name = f"{dataset_name}.tsv"
    data_path = os.path.join(data_root, file_name)
    
    # Local-only mode: never auto-download
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Local dataset file not found: {data_path}. "
            "Please place MMMU_DEV_VAL.tsv under --data-dir."
        )

    local_md5 = md5(data_path)
    if local_md5 != MMMU_DATASET_MD5:
        print(
            f"Warning: local {dataset_name}.tsv md5={local_md5} "
            f"(expected {MMMU_DATASET_MD5}). Using local file without download."
        )
    
    # Load the dataset (robust to malformed local TSV quoting)
    try:
        data = pd.read_csv(data_path, sep='\t')
    except Exception:
        try:
            try:
                csv.field_size_limit(sys.maxsize)
            except Exception:
                pass
            data = pd.read_csv(data_path, sep='\t', engine='python')
            print("Warning: C parser failed for MMMU TSV; loaded with python engine.")
        except Exception:
            try:
                data = pd.read_csv(
                    data_path,
                    sep='\t',
                    engine='python',
                    quoting=csv.QUOTE_NONE,
                    on_bad_lines='skip',
                )
                print(
                    "Warning: MMMU TSV appears malformed; loaded with relaxed parsing "
                    "(QUOTE_NONE + skip bad lines)."
                )
            except Exception as e3:
                raise RuntimeError(
                    f"Failed to parse local MMMU TSV: {data_path}. "
                    f"md5={local_md5}. Please replace with a valid local copy."
                ) from e3

    # Normalize malformed quoted headers: '"index"' -> 'index'
    data.columns = [str(c).strip().strip('"').strip("'") for c in data.columns]

    # Resolve index column robustly
    if 'index' not in data.columns:
        for cand in ['id', 'question_id']:
            if cand in data.columns:
                data['index'] = data[cand]
                break
    if 'index' not in data.columns:
        raise KeyError(
            f"MMMU parsed file missing index-like column. Parsed columns: {list(data.columns)}"
        )
    
    # Process the dataset
    data['index'] = [str(x).strip().strip('"').strip("'") for x in data['index']]
    
    # Handle image data
    if 'image' in data:
        data['image'] = [str(x) for x in data['image']]
        image_map = {x: y for x, y in zip(data['index'], data['image'])}
        unresolved_refs = 0
        for k in image_map:
            if len(image_map[k]) <= 64:
                idx = image_map[k]
                if idx in image_map and image_map[idx] is not None and len(image_map[idx]) > 64:
                    image_map[k] = image_map[idx]
                else:
                    # malformed/truncated local TSV: keep as unresolved placeholder
                    image_map[k] = None
                    unresolved_refs += 1
        if unresolved_refs > 0:
            print(f"Warning: {unresolved_refs} MMMU image references unresolved; will use image_path when available.")

        images = []
        for k in data['index']:
            if image_map[k] is None:
                images.append([])
                continue
            try:
                parsed = toliststr(image_map[k])
            except Exception:
                parsed = []
            if parsed is None:
                parsed = []
            images.append(parsed)
        data['image'] = [x[0] if len(x) == 1 else (x if len(x) > 1 else None) for x in images]
    
    # Handle image paths
    if 'image_path' in data:
        paths = []
        for x in data['image_path']:
            try:
                parsed = toliststr(x)
            except Exception:
                parsed = []
            if parsed is None:
                parsed = []
            paths.append(parsed)
        data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]
    
    # Convert index to int if possible
    if np.all([isinstance(x, int) or x.isdigit() for x in data['index']]):
        data['index'] = [int(x) for x in data['index']]
    
    return data

def dump_image(line, img_root):
    """Save image data to disk and return the path."""
    os.makedirs(img_root, exist_ok=True)

    def _fallback_paths_from_row(row):
        if 'image_path' not in row:
            return None
        paths = toliststr(row['image_path'])
        if len(paths) == 0:
            return None
        abs_paths = [p if os.path.isabs(p) else os.path.join(img_root, p) for p in paths]
        existing = [p for p in abs_paths if os.path.exists(p)]
        return existing if len(existing) > 0 else abs_paths
    
    if 'image' in line and line['image'] is not None and not (isinstance(line['image'], float) and pd.isna(line['image'])):
        if isinstance(line['image'], list):
            tgt_path = []
            assert 'image_path' in line
            for img, im_name in zip(line['image'], line['image_path']):
                path = os.path.join(img_root, im_name)
                if not os.path.exists(path):
                    try:
                        decode_base64_to_image_file(img, path)
                    except Exception:
                        fallback = _fallback_paths_from_row(line)
                        if fallback is None:
                            raise
                        return fallback
                tgt_path.append(path)
        else:
            tgt_path = os.path.join(img_root, f"{line['index']}.jpg")
            if not os.path.exists(tgt_path):
                try:
                    decode_base64_to_image_file(line['image'], tgt_path)
                except Exception:
                    fallback = _fallback_paths_from_row(line)
                    if fallback is None:
                        raise
                    return fallback
            tgt_path = [tgt_path]
    else:
        assert 'image_path' in line
        tgt_path = _fallback_paths_from_row(line)
        if tgt_path is None:
            tgt_path = toliststr(line['image_path'])
    
    return tgt_path

def MMMU_preproc(data):
    """
    Preprocess MMMU dataset to reformulate open questions to multi-choice ones.
    This aligns with the implementation in multiple_choice.py
    """
    print("Preprocessing MMMU dataset...")
    cnt = 0
    As, Bs, Ans = list(data['A']), list(data['B']), list(data['answer'])
    lt = len(data)
    for i in range(lt):
        if pd.isna(As[i]):
            As[i] = Ans[i]
            Bs[i] = 'Other Answers'
            cnt += 1
    print(f'During MMMU_preproc in Evaluation, {cnt} open questions are re-formulated to multi-choice ones.')
    data['A'] = As
    data['B'] = Bs
    return data