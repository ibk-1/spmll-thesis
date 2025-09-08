import json
import os
import argparse
import numpy as np

pp = argparse.ArgumentParser(description='Format COCO metadata.')
pp.add_argument('--load-path', type=str, default='../data/coco', help='Path to a directory containing a copy of the COCO dataset.')
pp.add_argument('--save-path', type=str, default='../data/coco', help='Path to output directory.')
args = pp.parse_args()

def parse_categories(categories):
    category_list = []
    supercategory_list = []
    id_to_index = {}
    for i, cat in enumerate(categories):
        category_list.append(cat['name'])
        supercategory_list.append(cat['supercategory'])
        id_to_index[int(cat['id'])] = i
    return category_list, supercategory_list, id_to_index

# initialize metadata dictionary:
meta = {
    'category_id_to_index': {},
    'category_list': [],
    'supercategory_list': [],
    'supercategory_to_index': {}
}

for split in ['train', 'val']:
    with open(os.path.join(args.load_path, 'annotations', f'instances_{split}2014.json'), 'r') as f:
        D = json.load(f)

    cat_list, supercat_list, id_to_index = parse_categories(D['categories'])

    if len(meta['category_list']) == 0:
        # Save full category metadata
        meta['category_list'] = cat_list
        meta['supercategory_list'] = supercat_list
        meta['category_id_to_index'] = id_to_index

        # Build unique supercategories, excluding "person"
        unique_supers = sorted(set(supercat_list))
        if "person" in unique_supers:
            unique_supers.remove("person")
        meta['unique_supercategories'] = unique_supers
        meta['supercategory_to_index'] = {sc: i for i, sc in enumerate(unique_supers)}
    else:
        # Validate consistency across splits
        assert cat_list == meta['category_list'], "Category list mismatch between splits."
        assert supercat_list == meta['supercategory_list'], "Supercategory list mismatch between splits."
        assert id_to_index == meta['category_id_to_index'], "Category ID->index mapping mismatch between splits."

    # Image ids and indexing
    image_id_list = sorted(np.unique([str(ann['image_id']) for ann in D['annotations']]))
    image_id_list = np.array(image_id_list, dtype=int)
    image_id_to_index = {image_id_list[i]: i for i in range(len(image_id_list))}

    num_categories = len(meta['category_list'])                # 80
    num_supercats = len(meta['unique_supercategories'])        # 11 (person excluded)
    num_images = len(image_id_list)

    # label matrix: fine classes + supercategories (person supercat excluded)
    label_matrix = np.zeros((num_images, num_categories + num_supercats), dtype=np.float32)
    image_ids = np.zeros(num_images, dtype=np.int64)

    for ann in D['annotations']:
        image_id = int(ann['image_id'])
        row_index = image_id_to_index[image_id]

        category_id = int(ann['category_id'])
        category_index = meta['category_id_to_index'][category_id]

        # mark fine label
        label_matrix[row_index, category_index] = 1.0

        # mark supercategory (if not filtered out)
        supercat_name = meta['supercategory_list'][category_index]
        if supercat_name in meta['supercategory_to_index']:
            supercat_index = meta['supercategory_to_index'][supercat_name]
            label_matrix[row_index, num_categories + supercat_index] = 1.0

        image_ids[row_index] = image_id

    # convert to relative file paths
    image_ids = np.array(
        [f"{split}2014/COCO_{split}2014_{str(int(x)).zfill(12)}.jpg" for x in image_ids],
        dtype=object
    )

    # save arrays
    np.save(os.path.join(args.save_path, f'formatted_{split}_labels.npy'), label_matrix)
    np.save(os.path.join(args.save_path, f'formatted_{split}_images.npy'), image_ids)

# save metadata (includes mappings and unique supercats)
with open(os.path.join(args.save_path, 'annotations', 'formatted_metadata.json'), 'w') as f:
    json.dump(meta, f, indent=2)
