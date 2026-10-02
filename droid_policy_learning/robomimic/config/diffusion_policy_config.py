"""
Config for Diffusion Policy algorithm.
"""

from robomimic.config.base_config import BaseConfig

class DiffusionPolicyConfig(BaseConfig):
    ALGO_NAME = "diffusion_policy"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config`
        argument to the constructor. Any parameter that an algorithm needs to determine its
        training and test-time behavior should be populated here.
        """

        # optimization parameters (paper Table S1: Adam, lr 1e-4, linear schedule)
        self.algo.optim_params.policy.optimizer_type = "adam"           # "adam", "adamw", or "muon"
        self.algo.optim_params.policy.betas = (0.9, 0.999)              # Adam/AdamW betas (momentum coefficients)
        self.algo.optim_params.policy.eps = 1e-8                        # Adam/AdamW epsilon for numerical stability
        self.algo.optim_params.policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # final/initial LR for the linear schedule
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs

        # Learning rate scheduler type: "linear" (paper), "multistep", or "cosine_warmup"
        self.algo.optim_params.policy.learning_rate.scheduler_type = "linear"
        self.algo.optim_params.policy.learning_rate.warmup_epochs = 0   # number of warmup epochs (0 = no warmup)
        self.algo.optim_params.policy.learning_rate.warmup_type = "linear"  # "linear" or "constant"
        self.algo.optim_params.policy.learning_rate.total_epochs = 1000 # total training epochs (for cosine scheduler)
        self.algo.optim_params.policy.learning_rate.min_lr_ratio = 0.01 # minimum lr = initial_lr * min_lr_ratio

        self.algo.optim_params.policy.regularization.L2 = 0.0           # weight decay (the paper uses plain Adam)

        # Optional Student LR schedule (0 and 0 in the paper protocol): the Student
        # U-Net and its timestep keep their weights for the first student_freeze_steps
        # optimizer steps while the readout, policy, and alignment adapters train,
        # then their LR ramps up linearly over student_lr_warmup_steps.
        self.algo.optim_params.policy.student_freeze_steps = 0
        self.algo.optim_params.policy.student_lr_warmup_steps = 0

        # horizon parameters
        self.algo.horizon.observation_horizon = 2
        self.algo.horizon.action_horizon = 8
        self.algo.horizon.prediction_horizon = 16

        # UNet parameters
        self.algo.unet.enabled = True
        self.algo.unet.diffusion_step_embed_dim = 256
        self.algo.unet.down_dims = [256,512,1024]
        self.algo.unet.kernel_size = 5
        self.algo.unet.n_groups = 8

        # EMA parameters: decay = min(max_decay, 1 - (1 + step / inv_gamma) ** -power)
        self.algo.ema.enabled = True
        self.algo.ema.power = 0.75
        self.algo.ema.inv_gamma = 1.0
        self.algo.ema.max_decay = 1.0

        # Noise Scheduler
        ## DDPM
        self.algo.ddpm.enabled = True
        self.algo.ddpm.num_train_timesteps = 100
        self.algo.ddpm.num_inference_timesteps = 100
        self.algo.ddpm.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddpm.clip_sample = True
        self.algo.ddpm.prediction_type = 'epsilon'
        self.algo.noise_samples = 1

        ## DDIM
        self.algo.ddim.enabled = False
        self.algo.ddim.num_train_timesteps = 50 #100
        self.algo.ddim.num_inference_timesteps = 10
        self.algo.ddim.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddim.clip_sample = True
        self.algo.ddim.set_alpha_to_one = True
        self.algo.ddim.steps_offset = 0
        self.algo.ddim.prediction_type = 'epsilon'

        # Robot-DIFT visual interface: "paper" (S2-FPN + frozen CLIP readout shared
        # with Stage II) or "legacy" (per-camera CleanDIFT S2-FPN/query encoder).
        self.algo.robot_dift_readout = "paper"
        self.algo.robot_dift_paper_readout.clip_model_path = None
        self.algo.robot_dift_paper_readout.output_dim = 512
        self.algo.robot_dift_paper_readout.fpn_dim = 256
        self.algo.robot_dift_paper_readout.model_dim = 256
        self.algo.robot_dift_paper_readout.num_heads = 8
        self.algo.robot_dift_paper_readout.transformer_layers = 1
        self.algo.robot_dift_paper_readout.mlp_hidden_dims = [1024, 512]
        self.algo.robot_dift_paper_readout.dropout = 0.0

        # Teacher alignment: lambda(s) anneals from weight to weight * min_decay_factor
        # over warmdown_steps (paper Eq. 4: 0.1 -> 0.001, linear, 150k steps).
        self.algo.cleandift_alignment_weight = 0.0  # Alignment loss weight (0.0 = disabled)
        self.algo.cleandift_alignment_warmdown_frac = 0.5        # Used when warmdown_steps is None
        self.algo.cleandift_alignment_warmdown_steps = None      # Fixed warmdown steps
        self.algo.cleandift_alignment_min_decay_factor = 0.01    # Final weight multiplier after warmdown
        self.algo.cleandift_alignment_decay_power = 1.0          # Decay exponent (1.0 = linear)
        self.algo.cleandift_alignment_schedule_origin = "absolute"  # "absolute" preserves annealing on resume

        # Language dropout for robustness (per-sample)
        self.algo.language_dropout_prob = 0.0

        # Disable low_dim inputs (pure RGB policy)
        self.algo.rgb_only = False

        # Optional optimizer steps at which to record Student gradient/update audits
        self.algo.representation_audit_steps = []
