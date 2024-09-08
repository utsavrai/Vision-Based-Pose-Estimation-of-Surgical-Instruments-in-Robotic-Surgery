import mmcv
import sys
import argparse
import json

parser = argparse.ArgumentParser(description="convert det from bop format to ours")
parser.add_argument("--ipath", type=str, default="../data/dataset/lnd1/train/000001/scene_gt_info.json", help="input path")
parser.add_argument("--opath", type=str, default="output.json", help="output path")
args = parser.parse_args()

# Load the scene_gt_info.json file
scene_gt_info = mmcv.load(args.ipath)

outs = {}

# Fixed information
scene_id = 1
obj_id = 1
score = 1.0
time = 0.1

catid2obj = {
    1: "lnd1"
}
objects = [
    "lnd1"
]
obj2id = {_name: _id for _id, _name in catid2obj.items()}

# Iterate through the entries in scene_gt_info
for image_id, details in scene_gt_info.items():
    scene_im_id = f"{scene_id}/{image_id}"

    # Since we only have one object, we assume that the bbox_est is taken from the first object in the list
    bbox = details[0]["bbox_obj"]

    cur_dict = {
        "bbox_est": bbox,
        "obj_id": obj_id,
        "score": score,
        "time": time,
    }

    if scene_im_id in outs.keys():
        outs[scene_im_id].append(cur_dict)
    else:
        outs[scene_im_id] = [cur_dict]

# Function to save the output JSON
def save_json(path, content, sort=False):
    """Saves the provided content to a JSON file.

    :param path: Path to the output JSON file.
    :param content: Dictionary/list to save.
    """
    with open(path, "w") as f:

        if isinstance(content, dict):
            f.write("{\n")
            if sort:
                content_sorted = sorted(content.items(), key=lambda x: x[0])
            else:
                content_sorted = content.items()
            for elem_id, (k, v) in enumerate(content_sorted):
                f.write('  "{}": {}'.format(k, json.dumps(v, sort_keys=True)))
                if elem_id != len(content) - 1:
                    f.write(",")
                f.write("\n")
            f.write("}")

        elif isinstance(content, list):
            f.write("[\n")
            for elem_id, elem in enumerate(content):
                f.write("  {}".format(json.dumps(elem, sort_keys=True)))
                if elem_id != len(content) - 1:
                    f.write(",")
                f.write("\n")
            f.write("]")

        else:
            json.dump(content, f, sort_keys=True)

# Save the JSON output
save_json(args.opath, outs)
