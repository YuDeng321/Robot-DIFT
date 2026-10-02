from typing import Dict, Tuple, Union, Optional, List
import copy
import torch
import torch.nn as nn
import hydra
import torch.nn.functional as F
from agents.encoders.crop_randomizer import CropRandomizer

class TensorResize(nn.Module):
    def __init__(self, size: Tuple[int, int]):
        super().__init__()
        self.size = size
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) in [0,1]
        if x.ndim != 4:
            raise ValueError(f"TensorResize expects 4D tensor (B,C,H,W), got {x.shape}")
        return F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)

class TensorCenterCrop(nn.Module):
    def __init__(self, size: Tuple[int, int]):
        super().__init__()
        self.h, self.w = size
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        if x.ndim != 4:
            raise ValueError(f"TensorCenterCrop expects 4D tensor (B,C,H,W), got {x.shape}")
        _, _, H, W = x.shape
        th, tw = self.h, self.w
        # If requested crop larger than input, upscale first to cover
        if th > H or tw > W:
            scale = max(th / H, tw / W)
            newH, newW = max(th, int(round(H * scale))), max(tw, int(round(W * scale)))
            x = F.interpolate(x, size=(newH, newW), mode="bilinear", align_corners=False)
            H, W = newH, newW
        i = (H - th) // 2
        j = (W - tw) // 2
        return x[:, :, i:i+th, j:j+tw]

class TensorNormalize(nn.Module):
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        super().__init__()
        m = torch.tensor(mean).view(1, 3, 1, 1)
        s = torch.tensor(std).view(1, 3, 1, 1)
        self.register_buffer("mean", m)
        self.register_buffer("std", s)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(self,
                 shape_meta: dict,
                 rgb_model: Union[nn.Module, Dict[str, nn.Module]],
                 resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
                 crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
                 random_crop: bool = True,
                 # replace BatchNorm with GroupNorm
                 use_group_norm: bool = False,
                 # use single rgb model for all rgb inputs
                 share_rgb_model: bool = False,
                 # renormalize rgb input with imagenet normalization
                 # assuming input in [0,1]
                 imagenet_norm: bool = False,
                 use_dift: bool = False,
                 alignment_cfg: Optional[dict] = None,
                 alignment_views: Optional[Union[str, List[str]]] = None,
                 ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        rgb_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        self.shared_rgb_model = None
        # handle sharing vision backbone
        shared_model = None
        if share_rgb_model:
            if isinstance(rgb_model, nn.Module):
                shared_model = rgb_model
            else:
                shared_model = hydra.utils.instantiate(rgb_model)
            self.shared_rgb_model = shared_model
            key_model_map['rgb'] = shared_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    this_model = copy.deepcopy(hydra.utils.instantiate(rgb_model))
                    # if isinstance(rgb_model, DictConfig):
                    #     # have provided model for each key
                    #     this_model = rgb_model[key]
                    # # if isinstance(rgb_model, DictConfig):
                    # #     this_model = hydra.utils.instantiate(rgb_model[key])
                    # # elif isinstance(rgb_model, dict):
                    # #     this_model = rgb_model[key]
                    # else:
                    #     assert isinstance(rgb_model, nn.Module)
                    #     # have a copy of the rgb model
                    #     this_model = copy.deepcopy(rgb_model)

                if this_model is not None:
                    # if use_group_norm:
                    #     this_model = replace_submodules(
                    #         root_module=this_model,
                    #         predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    #         func=lambda x: nn.GroupNorm(
                    #             num_groups=x.num_features // 16,
                    #             num_channels=x.num_features)
                    #     )
                    key_model_map[key] = this_model

                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = TensorResize(size=(h, w))
                    input_shape = (shape[0], h, w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_randomizer = TensorCenterCrop(size=(h, w))
                this_normalizer = nn.Identity()

                if imagenet_norm:
                    this_normalizer = TensorNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

                if use_dift:
                    this_normalizer = TensorNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.key_shape_map = key_shape_map
        if alignment_cfg is not None and not isinstance(alignment_cfg, dict):
            try:
                from omegaconf import OmegaConf
                alignment_cfg = OmegaConf.to_container(alignment_cfg, resolve=True)
            except Exception:
                alignment_cfg = dict(alignment_cfg)
        self.alignment_cfg = alignment_cfg
        if alignment_views is None:
            alignment_views = "all"
        self.alignment_views = alignment_views

    def forward(self, obs_dict, lang_cond=None, alignment_context: Optional[dict] = None):
        batch_size = None
        features = list()
        if isinstance(lang_cond, tuple):
            lang_cond = list(lang_cond)
        alignment_losses: list[torch.Tensor] = []
        weight = 0.0
        sample_indices = None
        captions = None
        alignment_images = None
        log_flag = False
        if alignment_context is not None:
            weight = float(alignment_context.get("weight", 0.0) or 0.0)
            sample_indices = alignment_context.get("sample_indices")
            captions = alignment_context.get("captions")
            alignment_images = alignment_context.get("images", {})
            log_flag = bool(alignment_context.get("log", False))

        def _build_alignment_payload(key: str, transform: nn.Module):
            if weight <= 0.0 or not alignment_images or key not in alignment_images:
                return None
            align_img = alignment_images[key]
            if align_img.ndim == 5:
                align_img = align_img[:, -1]
            if sample_indices is not None:
                align_img = align_img[sample_indices]
                if captions is not None:
                    per_captions = [captions[i] for i in sample_indices]
                else:
                    per_captions = None
            else:
                per_captions = captions
            align_img = transform(align_img)
            payload = {
                "images": align_img,
                "captions": per_captions,
            }
            if sample_indices is not None:
                payload["sample_indices"] = sample_indices
            if log_flag:
                payload["log"] = True
            return payload
        # process rgb input
        def _forward_single_view(model: nn.Module, key: str):
            nonlocal batch_size
            img = obs_dict[key]
            if batch_size is None:
                batch_size = img.shape[0]
            else:
                assert batch_size == img.shape[0]
            try:
                assert img.shape[1:] == self.key_shape_map[key]
            except AssertionError as e:
                print(f"key: {key}, shape: {img.shape[1:]}, expected: {self.key_shape_map[key]}")
                raise e
            img = self.key_transform_map[key](img)

            # Skip alignment when the backbone is explicitly frozen to avoid extra compute.
            if getattr(model, "freeze_backbone", False):
                per_camera_alignment = None
            else:
                per_camera_alignment = _build_alignment_payload(key, self.key_transform_map[key])

            if lang_cond is None:
                if per_camera_alignment is None:
                    output = model(img)
                else:
                    output = model(img, alignment_context=per_camera_alignment)
            else:
                if per_camera_alignment is None:
                    output = model(img, lang_cond)
                else:
                    output = model(img, lang_cond, alignment_context=per_camera_alignment)

            if isinstance(output, tuple) and len(output) == 2:
                feature, alignment_loss = output
            else:
                feature, alignment_loss = output, None
            return feature, alignment_loss

        if self.share_rgb_model:
            shared_model = self.key_model_map['rgb']
            for key in self.rgb_keys:
                feature, alignment_loss = _forward_single_view(shared_model, key)
                if feature.dim() == 2:  # [B, C] -> [B, 1, C] for uniform downstream handling
                    feature = feature.unsqueeze(1)
                elif feature.dim() != 3:
                    raise ValueError(f"Unexpected feature shape {feature.shape} for key {key}")
                features.append(feature)
                if alignment_loss is not None:
                    alignment_losses.append(alignment_loss)
        else:
            for key in self.rgb_keys:
                feature, alignment_loss = _forward_single_view(self.key_model_map[key], key)
                if feature.dim() == 2:
                    feature = feature.unsqueeze(1)
                elif feature.dim() != 3:
                    raise ValueError(f"Unexpected feature shape {feature.shape} for key {key}")
                features.append(feature)
                if alignment_loss is not None:
                    alignment_losses.append(alignment_loss)

        # concatenate all features -> [B, num_cam, num_tokens_per_cam, dim]
        result = torch.stack(features, dim=1)

        # result = torch.cat(features, dim=-1)
        alignment_loss = None
        if alignment_losses:
            alignment_loss = torch.stack([loss for loss in alignment_losses]).mean()

        return result, alignment_loss  # shape-> [B, f1 + f2] -> [B, 64 + 8]

    @staticmethod
    def _trainable_params(module: nn.Module) -> List[nn.Parameter]:
        return [param for param in module.parameters() if param.requires_grad]

    def get_parameter_groups(
            self,
            base_lr: float,
            adapter_lr_multiplier: float = 1.0,
            backbone_lr_multiplier: float = 1.0,
            **kwargs,
    ) -> List[dict]:
        groups: List[dict] = []

        def _groups_for_model(model: nn.Module, prefix: str) -> List[dict]:
            if hasattr(model, "get_parameter_groups"):
                model_groups = model.get_parameter_groups(
                    base_lr=base_lr,
                    adapter_lr_multiplier=adapter_lr_multiplier,
                    backbone_lr_multiplier=backbone_lr_multiplier,
                    **kwargs,
                )
                for group in model_groups:
                    if "name" in group:
                        group = dict(group)
                        group["name"] = f"{prefix}.{group['name']}"
                    groups.append(group)
                return groups

            params = self._trainable_params(model)
            if params:
                groups.append(
                    {
                        "params": params,
                        "lr": float(base_lr) * float(adapter_lr_multiplier),
                        "name": prefix,
                    }
                )
            return groups

        if self.share_rgb_model:
            _groups_for_model(self.key_model_map["rgb"], "rgb")
        else:
            for key in self.rgb_keys:
                _groups_for_model(self.key_model_map[key], key)

        return groups

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape,
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape
