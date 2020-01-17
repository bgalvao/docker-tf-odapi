import os
import json
from os.path import basename
from shutil import copyfile

import pandas
from tqdm import tqdm

from PIL import Image
import argparse


def write_labelmap_pbtxt(meta):
    label_map = [(i + 1, clss["title"]) for i, clss in enumerate(meta["classes"])]

    template = "item  {{\n  id: {}\n  name: '{}'\n}}\n\n"

    lm = ""
    for idee, name in label_map:
        lm += template.format(idee, name)

    savepath = os.path.join(_DST, "label_map.pbtxt")
    with open(savepath, "w") as f:
        f.write(lm)


def filter_train_test_anns(anns, tags=["train", "test", "val"]):
    """
    Iterate the annotation files and write csv annotations.
    If None tags, then dataset is split in .9 for training.
    """
    # i.e. if images were correctly tag with supervisely's DTL
    # with train, test, val; or other matching tagging...
    if tags is not None and len(tags) == 3:

        def is_tag_in_img(img_path, tag):
            with open(img_path, "r") as f:
                jason = json.load(f)
            tags = [taggy["name"] for taggy in jason["tags"]]
            return tag in tags

        def train_filter(path, tag=tags[0]):
            return is_tag_in_img(path, tag)

        def test_filter(path, tag=tags[1]):
            return is_tag_in_img(path, tag)

        def val_filter(path, tag=tags[2]):
            return is_tag_in_img(path, tag)

        train_anns = list(filter(train_filter, anns))
        test_anns = list(filter(test_filter, anns))
        val_anns = list(filter(val_filter, anns))

        assert (
            len(train_anns) > 0
        ), "No tag '{}' was found in images. Make sure to correct tags.".format(tags[0])
        assert (
            len(test_anns) > 0
        ), "No tag '{}' was found in images. Make sure to correct tags.".format(tags[1])
        assert (
            len(val_anns) > 0
        ), "No tag '{}' was found in images. Make sure to correct tags.".format(tags[2])

        return train_anns, test_anns, val_anns

    # otherwise split in .9/.1 ratio to train/test sets..
    else:
        l = len(anns)
        train_split_idx = int(0.9 * l)
        train_split = anns[:train_split_idx]
        test_split = anns[train_split_idx:]
        return train_split, test_split, None


def get_dict(ann_paths):

    print(ann_paths)

    def get_img_path(ann_path):
        return ann_path.replace("/ann/", "/img/")[:-5]

    def get_img_basename(ann_path):
        return basename(ann_path)[:-5]

    return {
        get_img_basename(ann_path): {"ann": ann_path, "img": get_img_path(ann_path)}
        for ann_path in ann_paths
    }


def make_ann_df(set_d):
    """
    set_d is a dataset-set dictionary from get_dict
    """

    def get_data(ann_path):
        with open(ann_path, "r") as f:
            jason = json.load(f)
        height = jason["size"]["height"]
        width = jason["size"]["width"]

        def get_obj_data(obj):
            clss = obj["classTitle"]
            coords = obj["points"]["exterior"]
            xmin = int(min(coords[0][0], coords[1][0]))
            ymin = int(min(coords[0][1], coords[1][1]))
            xmax = int(max(coords[0][0], coords[1][0]))
            ymax = int(max(coords[0][1], coords[1][1]))
            return {
                "class": clss,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }

        objs = [get_obj_data(obj) for obj in jason["objects"]]
        return {"height": height, "width": width, "objs": objs}

    # as per https://stackoverflow.com/a/17496530
    print("building rows")
    rows = []
    # for each image
    for k, v in tqdm(set_d.items()):
        filename = k
        # get 'filename', width, height...
        # from annotation
        ann_data = get_data(v["ann"])
        # for each object in image
        # build a row (dict)
        for obj in ann_data["objs"]:
            rows.append(
                {
                    "filename": filename,
                    "width": ann_data["width"],
                    "height": ann_data["height"],
                    "class": obj["class"],
                    "xmin": obj["xmin"],
                    "ymin": obj["ymin"],
                    "xmax": obj["xmax"],
                    "ymax": obj["ymax"],
                }
            )
    print("building dataframe")
    return pandas.DataFrame(rows).set_index("filename")


def remove_webps(anns, imgs):
    """
    The webp format is incompatible with PIL.
    As a prevention, these are removed from the dataset,
    so that the training session does not crash
    """
    incompatible_imgs = []
    print("checking compatibility of images")
    for img in tqdm(imgs):
        try:
            Image.open(img)
        except Exception as e:
            # print(e)
            incompatible_imgs.append(img)

    incompatible_anns = [
        img.replace("/img/", "/ann/") + ".json" for img in incompatible_imgs
    ]

    print(
        "removing {} imgs and {} annotations".format(
            len(incompatible_imgs), len(incompatible_anns)
        )
    )

    anns = [ann for ann in anns if ann not in incompatible_anns]
    imgs = [img for img in imgs if img not in incompatible_imgs]
    return anns, imgs


def load_data_paths(supervisely_dspath="./supervisely"):

    meta = None
    anns = []
    imgs = []

    for dr, subdirs, files in os.walk(supervisely_dspath):
        for f in files:
            filepath = os.path.join(dr, f)
            if "meta.json" in filepath:
                with open(os.path.join(dr, f), "r") as f:
                    meta = json.load(f)
            elif "/ann/" in filepath:
                anns.append(filepath)
            elif "/img/" in filepath:
                imgs.append(filepath)

    return meta, anns, imgs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="takes a supervisely-formatted dataset and gens a \
            tf_csv-formatted dataset"
    )
    parser.add_argument(
        "--src",
        help="path to source supervisely dataset",
        default="./input/supervisely",
    )
    parser.add_argument(
        "--dst",
        help="path to save the generated csv files to",
        default="./input/tf_csv",
    )
    parser.add_argument(
        "--no_tag_splitting",
        help="""if this flag is passed,
        then split the dataset 90/10 dataset into train and test splits; 
        otherwise, split dataset by the 'train', 'test', 'val' tags present in
        the annotation files.""",
        action="store_true",
    )
    args = parser.parse_args()
    print(args)

    _SRC = args.src
    _DST = args.dst
    _NO_TAG_SPLIT = args.no_tag_splitting

    meta, anns, imgs = load_data_paths(_SRC)

    # primeiro filtrar os webps, o colab deve usar o PIL
    anns, imgs = remove_webps(anns, imgs)

    write_labelmap_pbtxt(meta)

    if _NO_TAG_SPLIT:
        train_anns, test_anns, val_anns = filter_train_test_anns(anns, None)
    else:
        train_anns, test_anns, val_anns = filter_train_test_anns(anns)

    train_d = get_dict(train_anns)
    train_df = make_ann_df(train_d)
    train_df.to_csv(os.path.join(_DST, "train.csv"))

    test_d = get_dict(test_anns)
    test_df = make_ann_df(test_d)
    test_df.to_csv(os.path.join(_DST, "test.csv"))

    trainset_path = os.path.join(_DST, "images/train/")
    if not os.path.exists(trainset_path):
        os.makedirs(trainset_path)
    else:
        os.remove(trainset_path)
        os.makedirs(trainset_path)
    print("copying train images..")
    for d in tqdm(train_d.values()):
        img_path = d["img"]
        copyfile(img_path, os.path.join(trainset_path, basename(img_path)))

    testset_path = os.path.join(_DST, "images/test/")
    if not os.path.exists(testset_path):
        os.makedirs(testset_path)
    else:
        os.remove(testset_path)
        os.makedirs(testset_path)
    print("copying test images..")
    for d in tqdm(test_d.values()):
        img_path = d["img"]
        copyfile(img_path, os.path.join(testset_path, basename(img_path)))

    if val_anns is not None:
        valset_path = os.path.join(_DST, "test/")
        if not os.path.exists(valset_path):
            os.makedirs(valset_path)
        else:
            os.remove(valset_path)
            os.makedirs(valset_path)
        print("copying val images..")
        for d in tqdm(val_d.values()):
            img_path = d["img"]
            copyfile(img_path, os.path.join(valset_path, basename(img_path)))

