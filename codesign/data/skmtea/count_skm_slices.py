import pathlib
import numpy as np
from tqdm import tqdm

LABEL_LIST = ['patellar_cartilage', 'femoral_cartilage', 'tibial_cartilage', 'meniscus']

def get_valid_files_in_dir(data_dir, selected_coils=8):
    files = []
    invalid_names = set()
    valid_names = set()
    wrong_coil = 0
    empty_mask = 0

    data_dir = pathlib.Path(data_dir)
    for fname in tqdm(list(data_dir.iterdir()), desc="Scanning slices"):
        patient_id = fname.stem.split('_')[1]

        # first‐time seeing this patient: check coil count
        if patient_id in invalid_names:
            continue
        elif patient_id not in valid_names:
            kspace = np.load(fname, allow_pickle=True).item()['kspace']
            if kspace.shape[-1] != selected_coils:
                wrong_coil += 1
                invalid_names.add(patient_id)
                continue
            else:
                valid_names.add(patient_id)

        # now check mask completeness
        label_path = data_dir.parent / 'label' / fname.name
        label_dict = np.load(label_path, allow_pickle=True).item()
        masks = [label_dict[label].sum() for label in LABEL_LIST]
        if np.min(masks) == 0:
            empty_mask += 1
            continue

        files.append(fname)

    print(f"\nExcluded {wrong_coil} slices due to wrong coil count")
    print(f"Excluded {empty_mask} slices due to empty masks")
    print(f"Kept {len(files)} slices from {len(valid_names)} patients\n")

    return files, list(valid_names)

if __name__ == "__main__":
    # adjust path & coil count as needed
    data_dir = "datasets/processed_skm/kspace"
    selected_coils = 8

    get_valid_files_in_dir(data_dir, selected_coils)

