import yaml
from torchvision import transforms
from src import data
from src import dfnet, shapenet_dfnet


method_dict = {
    "dfnet": dfnet,
    "shapenet_dfnet": shapenet_dfnet,
}


# General config
def load_config(path, default_path=None):
    """Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    """
    # Load configuration from file itself
    with open(path, "r") as f:
        cfg_special = yaml.safe_load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get("inherit_from")

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, device=None, dataset=None):
    """Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    """
    method = cfg["method"]
    model = method_dict[method].config.get_model(cfg, device=device, dataset=dataset)
    return model


# Trainer
def get_trainer(model, optimizer, cfg, device):
    """Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    """
    method = cfg["method"]
    trainer = method_dict[method].config.get_trainer(model, optimizer, cfg, device)
    return trainer


# Datasets
def get_dataset(mode, cfg, return_idx=False):
    """Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    """
    print(cfg)
    method = cfg["method"]
    dataset_type = cfg["data"]["dataset"]
    dataset_folder = cfg["data"]["path"]
    categories = cfg["data"]["classes"]

    # Get split
    splits = {
        "train": cfg["data"]["train_split"],
        "val": cfg["data"]["val_split"],
        "test": cfg["data"]["test_split"],
    }

    split = splits[mode]
    # Create dataset
    if dataset_type == "Shapes3D":
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields["inputs"] = inputs_field

        if return_idx:
            fields["idx"] = data.IndexField()

        if mode == "train":
            transform = []
            if cfg["data"]["rotate"]:
                transform.append("rotate")
            if cfg["data"]["translate"]:
                transform.append("translate")
            if cfg["data"]["single_trans"]:
                transform.append("single_trans")
        else:
            transform = []

        print(f"""
              MODE: {mode}
              TRANSFORMS: {transform}
              FIELDS: {[f"{k}:{v}" for k, v in fields.items()]}
              """)
        dataset = data.Shapes3dDataset(
            dataset_folder,
            fields,
            split=split,
            categories=categories,
            cfg=cfg,
            transform=transform,
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg["data"]["dataset"])

    return dataset


def get_inputs_field(mode, cfg):
    """Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    """
    input_type = cfg["data"]["input_type"]

    if input_type is None:
        inputs_field = None
    elif input_type == "pointcloud":
        transform = transforms.Compose(
            [
                data.SubsamplePointcloud(cfg["data"]["pointcloud_n"]),
                data.PointcloudNoise(cfg["data"]["pointcloud_noise"]),
            ]
        )
        inputs_field = data.PointCloudField(
            cfg["data"]["pointcloud_file"], transform, multi_files=cfg["data"]["multi_files"]
        )
    elif input_type == "partial_pointcloud":
        transform = transforms.Compose(
            [
                data.SubsamplePointcloud(cfg["data"]["pointcloud_n"]),
                data.PointcloudNoise(cfg["data"]["pointcloud_noise"]),
            ]
        )
        inputs_field = data.PartialPointCloudField(
            cfg["data"]["pointcloud_file"],
            transform,
            multi_files=cfg["data"]["multi_files"],
            part_ratio=cfg["data"]["part_ratio"],
            partial_type=cfg["data"]["partial_type"],
        )
    elif input_type == "idx":
        inputs_field = data.IndexField()
    else:
        raise ValueError("Invalid input type (%s)" % input_type)
    return inputs_field
