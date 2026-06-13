#!/usr/bin/env python

import json
import os
import pickle
import random
import sys
import tarfile
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LIFT_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
LIFT_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_name: str
    pretrained: str | None
    tokenizer_model_name: str
    reference_clean: float | None
    reference_source: str


MODEL_SPECS = {
    "vanilla": ModelSpec(
        key="vanilla",
        label="vanilla LAION2B ViT-B/16",
        model_name="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        tokenizer_model_name="ViT-B-16",
        reference_clean=70.23,
        reference_source="OpenCLIP official classification CSV, ImageNet 1k top-1",
    ),
    "fare4": ModelSpec(
        key="fare4",
        label="FARE4 ViT-B/16 LAION2B",
        model_name="hf-hub:chs20/FARE4-ViT-B-16-laion2B-s34B-b88K",
        pretrained=None,
        tokenizer_model_name="ViT-B-16",
        reference_clean=56.6,
        reference_source="RobustVLM README Additional Models table",
    ),
    "tecoa4": ModelSpec(
        key="tecoa4",
        label="TeCoA4 ViT-B/16 LAION2B",
        model_name="hf-hub:chs20/TeCoA4-ViT-B-16-laion2B-s34B-b88K",
        pretrained=None,
        tokenizer_model_name="ViT-B-16",
        reference_clean=51.5,
        reference_source="RobustVLM README Additional Models table",
    ),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_model_keys(value):
    if value == "all":
        return ["vanilla", "fare4", "tecoa4"]
    keys = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [key for key in keys if key not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"unknown model keys: {unknown}")
    return keys


def import_open_clip():
    import open_clip

    return open_clip


def load_openclip_model(spec, device="cpu"):
    open_clip = import_open_clip()
    kwargs = {"device": device}
    if spec.pretrained is None:
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            spec.model_name,
            **kwargs,
        )
    else:
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            spec.model_name,
            pretrained=spec.pretrained,
            **kwargs,
        )
    model.eval()
    tokenizer = get_tokenizer(open_clip, spec)
    return model, preprocess_train, preprocess_val, tokenizer


def get_tokenizer(open_clip, spec):
    try:
        return open_clip.get_tokenizer(spec.model_name)
    except Exception:
        return open_clip.get_tokenizer(spec.tokenizer_model_name)


def transform_to_records(preprocess):
    records = []
    transforms_list = getattr(preprocess, "transforms", [preprocess])
    for item in transforms_list:
        record = {"type": item.__class__.__name__}
        for attr in ("size", "interpolation", "max_size", "antialias", "mean", "std"):
            if hasattr(item, attr):
                value = getattr(item, attr)
                if attr == "interpolation":
                    value = str(value)
                if isinstance(value, tuple):
                    value = list(value)
                record[attr] = value
        records.append(record)
    return records


def find_normalize(preprocess):
    for item in getattr(preprocess, "transforms", [preprocess]):
        if item.__class__.__name__ == "Normalize":
            return item
    return None


def preprocess_without_normalize(preprocess):
    transforms_list = list(getattr(preprocess, "transforms", [preprocess]))
    if transforms_list and transforms_list[-1].__class__.__name__ == "Normalize":
        transforms_list = transforms_list[:-1]
    return transforms.Compose(transforms_list)


def describe_model(spec, model, preprocess_train, preprocess_val, tokenizer):
    normalize = find_normalize(preprocess_val)
    mean = list(normalize.mean) if normalize is not None else None
    std = list(normalize.std) if normalize is not None else None
    visual = getattr(model, "visual", None)
    image_size = getattr(visual, "image_size", None)
    if isinstance(image_size, tuple):
        image_size = list(image_size)

    quickgelu_modules = [
        module.__class__.__name__
        for module in model.modules()
        if "quickgelu" in module.__class__.__name__.lower()
    ]
    tokenizer_name = tokenizer.__class__.__name__
    logit_scale = None
    if hasattr(model, "logit_scale"):
        logit_scale = float(model.logit_scale.detach().cpu().exp().item())

    return {
        "key": spec.key,
        "label": spec.label,
        "model_name": spec.model_name,
        "pretrained": spec.pretrained,
        "hf_hub": spec.model_name.startswith("hf-hub:"),
        "maps_to_non_quickgelu_vit_b16": not bool(quickgelu_modules),
        "quickgelu_modules": sorted(set(quickgelu_modules)),
        "image_size": image_size,
        "preprocess_train": transform_to_records(preprocess_train),
        "preprocess_val": transform_to_records(preprocess_val),
        "mean": mean,
        "std": std,
        "lift_mean": LIFT_CLIP_MEAN,
        "lift_std": LIFT_CLIP_STD,
        "matches_lift_normalization": mean == LIFT_CLIP_MEAN and std == LIFT_CLIP_STD,
        "tokenizer_type": tokenizer_name,
        "logit_scale_exp": logit_scale,
        "reference_clean": spec.reference_clean,
        "reference_source": spec.reference_source,
    }


def read_imagenet_templates():
    template_path = REPO_ROOT / "CLIP_eval" / "zeroshot-templates.json"
    with template_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["imagenet1k"]


def read_imagenet_classnames():
    path = REPO_ROOT / "open_flamingo" / "eval" / "classification_utils.py"
    spec = importlib_util.spec_from_file_location("robustvlm_classification_utils", path)
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.IMAGENET_1K_CLASS_ID_TO_LABEL.values())


def read_cifar100_classnames(data_root):
    meta_path = Path(data_root) / "cifar-100-python" / "meta"
    with meta_path.open("rb") as handle:
        meta = pickle.load(handle, encoding="latin1")
    return list(meta["fine_label_names"])


def build_text_features(model, tokenizer, classnames, templates, device, batch_size=256):
    embeddings = []
    with torch.no_grad():
        for name in classnames:
            texts = [template.format(c=name) if "{c}" in template else template.format(name) for template in templates]
            tokens = tokenizer(texts).to(device)
            class_embeddings = model.encode_text(tokens)
            class_embeddings = F.normalize(class_embeddings, dim=-1)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            embeddings.append(class_embedding.detach().cpu())
    return torch.stack(embeddings, dim=1).to(device)


def logits_from_pixel_images(model, normalize, text_features, images):
    image_features = model.encode_image(normalize(images), normalize=True)
    logits = image_features @ text_features
    if hasattr(model, "logit_scale"):
        logits = logits * model.logit_scale.exp()
    return logits


class CIFAR100PythonTest(Dataset):
    def __init__(self, data_root, transform):
        self.path = Path(data_root) / "cifar-100-python" / "test"
        if not self.path.exists():
            raise FileNotFoundError(f"CIFAR-100 test file not found: {self.path}")
        with self.path.open("rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        self.images = data["data"]
        self.labels = data["fine_labels"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        array = self.images[index].reshape(3, 32, 32).transpose(1, 2, 0)
        image = Image.fromarray(array)
        return self.transform(image), int(self.labels[index])


def ensure_cifar100_from_autodl(data_root, pub_root="/root/autodl-pub"):
    data_root = Path(data_root)
    target = data_root / "cifar-100-python"
    if target.exists():
        return {"status": "exists", "path": str(target)}
    source = Path(pub_root) / "cifar-100" / "cifar-100-python.tar.gz"
    if not source.exists():
        raise FileNotFoundError(f"CIFAR-100 public archive not found: {source}")
    data_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source, "r:gz") as archive:
        archive.extractall(data_root)
    return {"status": "extracted_from_autodl_pub", "path": str(target), "source": str(source)}


def pixel_preprocess_for_size(image_size=224):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])


def locate_imagenet_val(data_root, explicit=None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    root = Path(data_root)
    candidates.extend([
        root / "imagenet" / "val",
        root / "ImageNet" / "val",
        root / "ILSVRC2012" / "val",
        root / "ImageNet" / "ILSVRC2012" / "val",
        root / "val",
    ])
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            class_dirs = [item for item in candidate.iterdir() if item.is_dir()]
            if class_dirs:
                return str(candidate)
    return None
