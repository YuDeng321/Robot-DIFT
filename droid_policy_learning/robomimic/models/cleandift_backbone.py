import torch
import torch.nn as nn
import sys
import os

try:
    from omegaconf import ListConfig
except ImportError:
    ListConfig = None

# The repository root is three directories above ``models``. Going one level
# farther can silently import an unrelated checkout of ``agents`` when this
# repository is nested inside a research workspace.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Use the S2-FPN CleanDIFT encoder
from agents.encoders.cleandift_img_encoder import CleanDIFTImgEncoder
import robomimic.models.base_nets as BaseNets


class CleanDIFTConv(BaseNets.ConvBase):
    """
    CleanDIFT backbone adapter for robomimic.

    Architecture: CleanDIFT + S2-FPN Bidirectional Fusion + Queries Attention

    This wraps CleanDIFTImgEncoder to be compatible with robomimic's ConvBase interface.
    The encoder uses:
        - Multi-scale feature extraction from SD2.1 UNet
        - S2-FPN bidirectional fusion (top-down + bottom-up)
        - Learnable queries with cross-attention pooling
        - Layer Scale for stable training
    """
    def __init__(
        self,
        input_channel=3,
        pretrained=False,  # Ignored, CleanDIFT always uses pretrained weights
        input_coord_conv=False,  # Not supported for CleanDIFT
        sd_version="sd21",  # "sd15" or "sd21"
        feature_key=None,  # Multi-scale feature keys (default: ["us3", "us6", "us8"])
        freeze_backbone=False,  # Whether to freeze CleanDIFT backbone
        freeze_head=False,  # Whether to freeze the S2-FPN/query readout
        use_text_condition=False,  # Enable language conditioning
        map_out_dim=512,  # Output dimension
        yaml_file=None,
        use_fp32=False,  # Use float32 instead of bfloat16
        custom_checkpoint=None,
        load_full_encoder_checkpoint=False,
        strict_full_encoder_checkpoint=False,
        # S2-FPN parameters
        fpn_dim=256,  # FPN intermediate dimension
        fpn_num_queries=8,  # Number of learnable queries
        fpn_dropout=0.1,  # Dropout rate
        layer_scale_init=0.1,  # Layer scale initial value
        fusion_mode="s2fpn",
        output_mode="pooled",
        # Legacy parameters (kept for backward compatibility)
        fusion_pooling=None,  # Ignored
        alignment_weight: float = 0.0,
        alignment_with_inputs: bool = True,
        num_t_stratification_bins=None,
        t_min=None,
        t_max=None,
        vae_latent_mode="sample",
        apply_feature_adapter=False,
        feature_adapter_timestep=None,
        alignment_apply_feature_adapter=None,
        alignment_feature_keys=None,
        alignment_layer_reduction="mean",
        student_init="cleandift",
    ):
        super(CleanDIFTConv, self).__init__()

        print(f"\n{'='*60}")
        print(f"[CleanDIFTConv] Initializing S2-FPN encoder")
        print(f"{'='*60}")
        print(f"  freeze_backbone = {freeze_backbone}")
        print(f"  freeze_head = {freeze_head}")
        print(f"  alignment_weight = {alignment_weight}")
        print(f"  feature_key = {feature_key}")
        print(f"  fpn_dim = {fpn_dim}")
        print(f"  fpn_num_queries = {fpn_num_queries}")
        print(f"  map_out_dim = {map_out_dim}")
        print(f"{'='*60}\n")

        assert input_channel == 3, "CleanDIFT only supports RGB images (3 channels)"
        assert not input_coord_conv, "CleanDIFT does not support CoordConv"

        # Default multi-scale feature keys for S2-FPN
        if feature_key is None:
            feature_key = ["us3", "us6", "us8"]
        if ListConfig is not None and isinstance(feature_key, ListConfig):
            feature_key = list(feature_key)
        elif isinstance(feature_key, tuple):
            feature_key = list(feature_key)

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create CleanDIFT encoder with S2-FPN
        self.encoder = CleanDIFTImgEncoder(
            sd_version=sd_version,
            feature_key=feature_key,
            freeze_backbone=freeze_backbone,
            freeze_head=freeze_head,
            device=device,
            camera_names=None,
            use_text_condition=use_text_condition,
            yaml_file=yaml_file,
            map_out_dim=map_out_dim,
            use_fp32=use_fp32,
            custom_checkpoint=custom_checkpoint,
            load_full_encoder_checkpoint=load_full_encoder_checkpoint,
            strict_full_encoder_checkpoint=strict_full_encoder_checkpoint,
            num_t_stratification_bins=num_t_stratification_bins,
            t_min=t_min,
            t_max=t_max,
            vae_latent_mode=vae_latent_mode,
            fusion_mode=fusion_mode,
            output_mode=output_mode,
            apply_feature_adapter=apply_feature_adapter,
            feature_adapter_timestep=feature_adapter_timestep,
            alignment_apply_feature_adapter=alignment_apply_feature_adapter,
            alignment_feature_keys=alignment_feature_keys,
            alignment_layer_reduction=alignment_layer_reduction,
            student_init=student_init,
            # S2-FPN parameters
            fpn_dim=fpn_dim,
            fpn_num_queries=fpn_num_queries,
            fpn_dropout=fpn_dropout,
            layer_scale_init=layer_scale_init,
        )

        self._output_dim = map_out_dim
        self._output_mode = str(output_mode).lower()
        self._num_queries = int(fpn_num_queries)
        self.use_text_condition = use_text_condition
        self.alignment_weight = float(alignment_weight)
        self.alignment_with_inputs = bool(alignment_with_inputs)
        self._freeze_backbone = freeze_backbone
        self._freeze_head = bool(freeze_head)
        self._logged_input_range = False

        # Sync freeze_backbone state with encoder
        # This ensures the encoder's freeze_backbone matches even after checkpoint loading
        self._sync_freeze_state()

    def _sync_freeze_state(self):
        """
        Synchronize freeze_backbone state between wrapper and encoder.
        This is important when resuming from checkpoint with different freeze settings.
        """
        def _set_trainable(module, trainable: bool):
            if isinstance(module, nn.Module):
                module.train(trainable)
                module.requires_grad_(trainable)

        def _set_ref_frozen(module):
            if isinstance(module, nn.Module):
                module.eval()
                module.requires_grad_(False)

        if hasattr(self.encoder, 'freeze_backbone'):
            if self.encoder.freeze_backbone != self._freeze_backbone:
                print(f"[CleanDIFTConv] Syncing freeze_backbone: encoder={self.encoder.freeze_backbone} -> {self._freeze_backbone}")
                self.encoder.freeze_backbone = self._freeze_backbone

                # Update backbone training mode and requires_grad
                if hasattr(self.encoder, 'model'):
                    model = self.encoder.model
                    if self._freeze_backbone:
                        # Freeze backbone
                        model.eval()
                        model.requires_grad_(False)
                        print(f"[CleanDIFTConv] Backbone FROZEN (eval mode, requires_grad=False)")
                    else:
                        # Only the clean student path is trainable. Frozen references
                        # must stay fixed for valid teacher anchoring.
                        model.train()
                        _set_trainable(getattr(model, "unet_feature_extractor_cleandift", None), True)
                        for attr in ("adapters", "mapping", "time_emb", "time_in_proj"):
                            _set_trainable(getattr(model, attr, None), True)
                        if hasattr(model, "timestep") and isinstance(model.timestep, torch.nn.Parameter):
                            model.timestep.requires_grad_(True)

                        _set_ref_frozen(getattr(model, "ae", None))
                        _set_ref_frozen(getattr(model, "unet_feature_extractor_base", None))
                        pipe = getattr(model, "pipe", None)
                        _set_ref_frozen(getattr(pipe, "text_encoder", None) if pipe is not None else None)
                        if hasattr(self.encoder, "_keep_frozen_reference_eval"):
                            self.encoder._keep_frozen_reference_eval()
                        print(f"[CleanDIFTConv] Backbone UNFROZEN (student/adapters trainable; AE/base/text frozen)")
            else:
                print(f"[CleanDIFTConv] freeze_backbone already synced: {self._freeze_backbone}")

        if hasattr(self.encoder, 'freeze_head'):
            if self.encoder.freeze_head != self._freeze_head:
                print(f"[CleanDIFTConv] Syncing freeze_head: encoder={self.encoder.freeze_head} -> {self._freeze_head}")
                self.encoder.freeze_head = self._freeze_head
            if hasattr(self.encoder, "_set_head_trainable"):
                self.encoder._set_head_trainable(not self._freeze_head)
            print(f"[CleanDIFTConv] Readout head {'FROZEN' if self._freeze_head else 'trainable'}")

    def forward(self, inputs, lang_cond=None):
        """
        Forward pass.

        Args:
            inputs: [B, 3, H, W] RGB images (typically 256x256)
            lang_cond: Optional language condition (str or list of str)

        Returns:
            features: [B, 1, map_out_dim] global features
        """
        if lang_cond is not None and not isinstance(lang_cond, (list, tuple)):
            lang_cond = [str(lang_cond)]
        if isinstance(lang_cond, tuple):
            lang_cond = list(lang_cond)

        if (
            not self._logged_input_range
            and os.environ.get("ROBOT_DIFT_LOG_CLEANDIFT_INPUT_RANGE", "").lower()
            in {"1", "true", "yes"}
        ):
            with torch.no_grad():
                stats = inputs.detach().float()
                print(
                    "[CleanDIFTConv] input range "
                    f"shape={tuple(inputs.shape)} "
                    f"min={stats.min().item():.6f} "
                    f"max={stats.max().item():.6f} "
                    f"mean={stats.mean().item():.6f}",
                    flush=True,
                )
            self._logged_input_range = True

        # Alignment for DROID finetuning is computed once in diffusion_policy.py,
        # where it can be scheduled and subsampled. Computing it here would build
        # an unused loss graph because ConvBase.forward returns features only.
        features, _ = self.encoder(inputs, lang_cond=lang_cond, alignment_context=None)
        return features

    def output_shape(self, input_shape):
        """Return the shape produced by the CleanDIFT policy interface."""
        if self._output_mode == "queries":
            return [self._num_queries, self._output_dim]
        return [self._output_dim]

    def get_parameter_groups(
        self,
        base_lr: float,
        backbone_lr_multiplier: float = 0.01,
        head_lr_multiplier: float = 1.0,
        backbone_weight_decay=None,
        head_weight_decay=None,
        student_lr_multiplier=None,
    ):
        """
        Get parameter groups with different learning rates for optimizer.

        Recommended for DROID pre-training:
            backbone_lr_multiplier: 0.01 (very slow adaptation of SD backbone)
            head_lr_multiplier: 1.0 (full learning for S2-FPN + Queries)

        Example:
            param_groups = model.get_parameter_groups(base_lr=1e-4)
            optimizer = torch.optim.AdamW(param_groups)
        """
        return self.encoder.get_parameter_groups(
            base_lr=base_lr,
            backbone_lr_multiplier=backbone_lr_multiplier,
            head_lr_multiplier=head_lr_multiplier,
            backbone_weight_decay=backbone_weight_decay,
            head_weight_decay=head_weight_decay,
            student_lr_multiplier=student_lr_multiplier,
        )


if __name__ != "__main__":
    import robomimic.models.base_nets as BaseNets
    # Add to base_nets module namespace
    BaseNets.CleanDIFTConv = CleanDIFTConv
