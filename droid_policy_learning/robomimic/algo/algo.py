"""
This file contains base classes that other algorithm classes subclass.
Each algorithm file also implements a algorithm factory function that
takes in an algorithm config (`config.algo`) and returns the particular
Algo subclass that should be instantiated, along with any extra kwargs.
These factory functions are registered into a global dictionary with the
@register_algo_factory_func function decorator. This makes it easy for
@algo_factory to instantiate the correct `Algo` subclass.
"""
import textwrap
from copy import deepcopy
from collections import OrderedDict
import random

import torch.nn as nn
import torch
import os
import io
import imageio
import numpy as np
import matplotlib.pyplot as plt

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.action_utils as AcUtils
import robomimic.utils.vis_utils as VisUtils

from torch.utils.data import DataLoader

# mapping from algo name to factory functions that map algo configs to algo class names
REGISTERED_ALGO_FACTORY_FUNCS = OrderedDict()


def register_algo_factory_func(algo_name):
    """
    Function decorator to register algo factory functions that map algo configs to algo class names.
    Each algorithm implements such a function, and decorates it with this decorator.

    Args:
        algo_name (str): the algorithm name to register the algorithm under
    """
    def decorator(factory_func):
        REGISTERED_ALGO_FACTORY_FUNCS[algo_name] = factory_func
    return decorator


def algo_name_to_factory_func(algo_name):
    """
    Uses registry to retrieve algo factory function from algo name.

    Args:
        algo_name (str): the algorithm name
    """
    return REGISTERED_ALGO_FACTORY_FUNCS[algo_name]


def algo_factory(algo_name, config, obs_key_shapes, ac_dim, device):
    """
    Factory function for creating algorithms based on the algorithm name and config.

    Args:
        algo_name (str): the algorithm name

        config (BaseConfig instance): config object

        obs_key_shapes (OrderedDict): dictionary that maps observation keys to shapes

        ac_dim (int): dimension of action space

        device (torch.Device): where the algo should live (i.e. cpu, gpu)
    """

    # @algo_name is included as an arg to be explicit, but make sure it matches the config
    assert algo_name == config.algo_name

    # use algo factory func to get algo class and kwargs from algo config
    factory_func = algo_name_to_factory_func(algo_name)
    algo_cls, algo_kwargs = factory_func(config.algo)

    # create algo instance
    return algo_cls(
        algo_config=config.algo,
        obs_config=config.observation,
        global_config=config,
        obs_key_shapes=obs_key_shapes,
        ac_dim=ac_dim,
        device=device,
        **algo_kwargs
    )


class Algo(object):
    """
    Base algorithm class that all other algorithms subclass. Defines several
    functions that should be overriden by subclasses, in order to provide
    a standard API to be used by training functions such as @run_epoch in
    utils/train_utils.py.
    """
    def __init__(
        self,
        algo_config,
        obs_config,
        global_config,
        obs_key_shapes,
        ac_dim,
        device
    ):
        """
        Args:
            algo_config (Config object): instance of Config corresponding to the algo section
                of the config

            obs_config (Config object): instance of Config corresponding to the observation
                section of the config

            global_config (Config object): global training config

            obs_key_shapes (OrderedDict): dictionary that maps observation keys to shapes

            ac_dim (int): dimension of action space

            device (torch.Device): where the algo should live (i.e. cpu, gpu)
        """
        self.optim_params = deepcopy(algo_config.optim_params)
        self.algo_config = algo_config
        self.obs_config = obs_config
        self.global_config = global_config

        self.ac_dim = ac_dim
        self.device = device
        self.obs_key_shapes = obs_key_shapes

        self.nets = nn.ModuleDict()
        self._create_shapes(obs_config.modalities, obs_key_shapes)
        self._create_networks()
        self._create_optimizers()
        assert isinstance(self.nets, nn.ModuleDict)

    def _create_shapes(self, obs_keys, obs_key_shapes):
        """
        Create obs_shapes, goal_shapes, and subgoal_shapes dictionaries, to make it
        easy for this algorithm object to keep track of observation key shapes. Each dictionary
        maps observation key to shape.

        Args:
            obs_keys (dict): dict of required observation keys for this training run (usually
                specified by the obs config), e.g., {"obs": ["rgb", "proprio"], "goal": ["proprio"]}
            obs_key_shapes (dict): dict of observation key shapes, e.g., {"rgb": [3, 224, 224]}
        """
        # determine shapes
        self.obs_shapes = OrderedDict()
        self.goal_shapes = OrderedDict()
        self.subgoal_shapes = OrderedDict()

        # We check across all modality groups (obs, goal, subgoal), and see if the inputted observation key exists
        # across all modalitie specified in the config. If so, we store its corresponding shape internally
        for k in obs_key_shapes:
            if "obs" in self.obs_config.modalities and k in [obs_key for modality in self.obs_config.modalities.obs.values() for obs_key in modality]:
                self.obs_shapes[k] = obs_key_shapes[k]
            if "goal" in self.obs_config.modalities and k in [obs_key for modality in self.obs_config.modalities.goal.values() for obs_key in modality]:
                self.goal_shapes[k] = obs_key_shapes[k]
            if "subgoal" in self.obs_config.modalities and k in [obs_key for modality in self.obs_config.modalities.subgoal.values() for obs_key in modality]:
                self.subgoal_shapes[k] = obs_key_shapes[k]

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        @self.nets should be a ModuleDict.
        """
        raise NotImplementedError

    def _create_optimizers(self):
        """
        Creates optimizers using @self.optim_params and places them into @self.optimizers.
        """
        self.optimizers = dict()
        self.lr_schedulers = dict()

        for k in self.optim_params:
            # only make optimizers for networks that have been created - @optim_params may have more
            # settings for unused networks
            if k in self.nets:
                if isinstance(self.nets[k], nn.ModuleList):
                    self.optimizers[k] = [
                        TorchUtils.optimizer_from_optim_params(net_optim_params=self.optim_params[k], net=self.nets[k][i])
                        for i in range(len(self.nets[k]))
                    ]
                    self.lr_schedulers[k] = [
                        TorchUtils.lr_scheduler_from_optim_params(net_optim_params=self.optim_params[k], net=self.nets[k][i], optimizer=self.optimizers[k][i])
                        for i in range(len(self.nets[k]))
                    ]
                else:
                    self.optimizers[k] = TorchUtils.optimizer_from_optim_params(
                        net_optim_params=self.optim_params[k], net=self.nets[k])
                    self.lr_schedulers[k] = TorchUtils.lr_scheduler_from_optim_params(
                        net_optim_params=self.optim_params[k], net=self.nets[k], optimizer=self.optimizers[k])

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        return batch

    def postprocess_batch_for_training(self, batch, obs_normalization_stats):
        """
        Does some operations (like channel swap, uint8 to float conversion, normalization)
        after @process_batch_for_training is called, in order to ensure these operations
        take place on GPU.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader. Assumed to be on the device where
                training will occur (after @process_batch_for_training
                is called)

            obs_normalization_stats (dict or None): if provided, this should map observation
                keys to dicts with a "mean" and "std" of shape (1, ...) where ... is the
                default shape for the observation.

        Returns:
            batch (dict): postproceesed batch
        """
        obs_keys = ["obs", "next_obs", "goal_obs"]
        for k in obs_keys:
            if k in batch and batch[k] is not None:
                batch[k] = ObsUtils.process_obs_dict(batch[k])
                if obs_normalization_stats is not None:
                    batch[k] = ObsUtils.normalize_dict(batch[k], obs_normalization_stats=obs_normalization_stats)
        return batch

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        assert validate or self.nets.training
        return OrderedDict()

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss log (dict): name -> summary statistic
        """
        log = OrderedDict()

        # record current optimizer learning rates
        for k in self.optimizers:
            for i, param_group in enumerate(self.optimizers[k].param_groups):
                log["Optimizer/{}{}_lr".format(k, i)] = param_group["lr"]

        return log

    def on_epoch_end(self, epoch):
        """
        Called at the end of each epoch.
        """

        # LR scheduling updates
        for k in self.lr_schedulers:
            if self.lr_schedulers[k] is not None:
                self.lr_schedulers[k].step()

    def set_eval(self):
        """
        Prepare networks for evaluation.
        """
        self.nets.eval()

    def set_train(self):
        """
        Prepare networks for training.
        """
        self.nets.train()

    def _get_nets(self):
        """
        Get the actual nets ModuleDict, unwrapping DDP if necessary.
        This is useful for accessing nets like self.nets["policy"] when DDP is enabled.
        """
        from torch.nn.parallel import DistributedDataParallel as DDP
        if isinstance(self.nets, DDP):
            return self.nets.module
        return self.nets

    @staticmethod
    def _strip_ddp_from_key(key):
        if "module" not in key:
            return key
        parts = [p for p in key.split('.') if p != 'module']
        return '.'.join(parts)

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        state = self.nets.state_dict()
        cleaned_state = OrderedDict()
        for k, v in state.items():
            cleaned_state[self._strip_ddp_from_key(k)] = v
        if hasattr(state, "_metadata"):
            cleaned_state._metadata = state._metadata
        return cleaned_state

    def get_optimizer_states(self):
        """
        Get optimizer state dicts for checkpoint saving.
        """
        return self.optimizers

    def get_scheduler_states(self):
        """
        Get learning rate scheduler state dicts for checkpoint saving.
        """
        return self.lr_schedulers

    def load_optimizer_states(self, optimizer_states):
        """
        Load optimizer states from checkpoint.

        Args:
            optimizer_states (dict): dictionary of optimizer state dicts
        """
        for k, state in optimizer_states.items():
            if k in self.optimizers:
                if isinstance(self.optimizers[k], list):
                    for i, s in enumerate(state):
                        self.optimizers[k][i].load_state_dict(s)
                else:
                    self.optimizers[k].load_state_dict(state)

    def load_scheduler_states(self, scheduler_states):
        """
        Load learning rate scheduler states from checkpoint.

        Args:
            scheduler_states (dict): dictionary of scheduler state dicts
        """
        for k, state in scheduler_states.items():
            if k in self.lr_schedulers and state is not None:
                if isinstance(self.lr_schedulers[k], list):
                    for i, s in enumerate(state):
                        if s is not None and self.lr_schedulers[k][i] is not None:
                            self.lr_schedulers[k][i].load_state_dict(s)
                else:
                    if self.lr_schedulers[k] is not None:
                        self.lr_schedulers[k].load_state_dict(state)

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        try:
            self.nets.load_state_dict(model_dict)
        except RuntimeError:
            if not any("module" in k for k in model_dict.keys()):
                raise
            cleaned_state = OrderedDict()
            for k, v in model_dict.items():
                cleaned_state[self._strip_ddp_from_key(k)] = v
            self.nets.load_state_dict(cleaned_state)

    def __repr__(self):
        """
        Pretty print algorithm and network description.
        """
        return "{} (\n".format(self.__class__.__name__) + \
               textwrap.indent(self.nets.__repr__(), '  ') + "\n)"

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        pass


class PolicyAlgo(Algo):
    """
    Base class for all algorithms that can be used as policies.
    """
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        raise NotImplementedError

    def compute_traj_pred_actual_actions(self, traj, return_images=False):
        """
        traj is an DROIDDataset object representing one trajectory
        This function is slow (>1s per trajectory) because there is no batching
        and instead loops through all timesteps one by one
        TODO: documentation
        """
        if return_images:
            image_keys = [item for item in traj.__getitem__(0)['obs'].keys() if "image" in item]
            images = {key: [] for key in image_keys}
        else:
            images = None

        dataloader = DataLoader(
            dataset=traj,
            sampler=None,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )

        self.reset()
        actual_actions = []
        predicted_actions = []

        # loop through each timestep
        for batch in iter(dataloader):
            batch = self.process_batch_for_training(batch)

            if return_images:
                for image_key in image_keys:
                    im = batch["obs"][image_key][0][-1]
                    im = TensorUtils.to_numpy(im).astype(np.uint32)
                    images[image_key].append(im)

            batch = self.postprocess_batch_for_training(batch, obs_normalization_stats=None) # ignore obs_normalization for now

            model_output = self.get_action(batch["obs"])

            actual_action = TensorUtils.to_numpy(
                batch["actions"][0][0]
            )
            predicted_action = TensorUtils.to_numpy(
                model_output[0]
            )

            actual_actions.append(actual_action)
            predicted_actions.append(predicted_action)

        actual_actions = np.array(actual_actions)
        predicted_actions = np.array(predicted_actions)
        return actual_actions, predicted_actions, images

    def compute_batch_visualize(self, batch, num_samples, savedir=None):
        visualize = savedir is not None

        # Visualize whichever cameras this run trains on (two or three views).
        image_keys = sorted(
            key for key, value in batch["obs"].items()
            if key.startswith("camera/image/") and torch.is_tensor(value)
        )
        images = {
            key.split("/")[-1]: batch["obs"][key][:num_samples]
            for key in image_keys
        }

        if visualize:
            print("Saving batch visualization plots to {}".format(savedir))

        vis_log = {}
        preds = None
        actual_actions = None
        action_plot_img = None
        plot_samples = 0
        try:
            plot_samples = min(num_samples, batch["actions"].shape[0])
            if plot_samples > 0:
                batch_clone = TensorUtils.clone(batch)
                batch_device = TensorUtils.to_device(batch_clone, self.device)
                with torch.no_grad():
                    was_training = self.nets.training
                    if was_training:
                        self.set_eval()
                    processed = self.process_batch_for_training(batch_device)
                    processed = self.postprocess_batch_for_training(processed, obs_normalization_stats=None)
                    actual_actions = processed["actions"][:plot_samples].detach().cpu().numpy()
                    preds = []
                    self.reset()
                    lang_prompts_batch = processed.get("lang_prompts", None)
                    for i in range(plot_samples):
                        obs_input = {k: processed["obs"][k][i:i+1].clone() for k in processed["obs"]}
                        sample_prompts = None
                        if lang_prompts_batch is not None:
                            if isinstance(lang_prompts_batch, (list, tuple)):
                                if i < len(lang_prompts_batch):
                                    sample_prompts = [lang_prompts_batch[i]]
                            elif isinstance(lang_prompts_batch, torch.Tensor):
                                sample_prompts = lang_prompts_batch[i:i+1]
                        traj_pred = self._get_action_trajectory(obs_dict=obs_input, lang_prompts=sample_prompts)
                        preds.append(traj_pred[0].detach().cpu().numpy())
                    self.reset()
                    if was_training:
                        self.set_train()
                if len(preds) > 0:
                    preds = np.stack(preds, axis=0)
                if len(preds) > 0:
                    action_horizon = preds.shape[1]
                    if actual_actions.shape[1] != action_horizon:
                        actual_actions = actual_actions[:, :action_horizon]
                if len(preds) > 0 and preds.shape == actual_actions.shape and preds.size > 0:
                    time_axis = np.arange(actual_actions.shape[1])
                    action_keys = getattr(self.global_config.train, "action_keys", None)
                    action_shapes = getattr(self.global_config.train, "action_shapes", None)
                    if action_keys is None or action_shapes is None:
                        action_keys = ["action"]
                        action_shapes = [(self.ac_dim,)]
                    rows = len(action_keys)
                    fig, axes = plt.subplots(rows, 1, figsize=(10, max(3, 3 * rows)), sharex=True)
                    if rows == 1:
                        axes = [axes]
                    start_idx = 0
                    for ax, key, shape in zip(axes, action_keys, action_shapes):
                        dim = int(np.prod(shape))
                        end_idx = start_idx + dim
                        actual_block = actual_actions[0, :, start_idx:end_idx]
                        pred_block = preds[0, :, start_idx:end_idx]
                        for j in range(dim):
                            ax.plot(time_axis, actual_block[:, j], label="gt" if j == 0 else None, color="tab:blue")
                            ax.plot(time_axis, pred_block[:, j], label="pred" if j == 0 else None, color="tab:orange", linestyle="--")
                        ax.set_title(f"{key} (sample 0)", fontsize=12)
                        ax.tick_params(labelsize=9)
                        start_idx = end_idx
                    axes[0].legend(loc="upper right", fontsize=9, frameon=False)
                    axes[-1].set_xlabel("Timestep", fontsize=10)
                    fig.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
                    buf.seek(0)
                    action_plot_img = imageio.imread(buf)
                    buf.close()
                    plt.close(fig)
        except Exception as e:
            print(f"[BatchVis] Failed to compute action visualization: {e}")

        primary_key = next(iter(images), None)

        if visualize:
            save_path = os.path.join(savedir, "batch_images.png")
            VisUtils.make_batch_vis_plot(
                save_path=save_path,
                images=images,
            )
            try:
                vis_array = imageio.imread(save_path)
                if vis_array.ndim == 3 and vis_array.shape[2] == 4:
                    vis_array = vis_array[..., :3]
                vis_log["BatchVis/batch_images"] = vis_array.astype(np.uint8)
            except Exception as e:
                print(f"[BatchVis] Failed to load saved visualization: {e}")

        if action_plot_img is not None:
            vis_log["BatchVis/action_predictions"] = action_plot_img

        try:
            if primary_key is not None and preds is not None and actual_actions is not None:
                primary_imgs = images[primary_key][:plot_samples].cpu()
                num_cols = primary_imgs.shape[0]
                fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
                if num_cols == 1:
                    axes = [axes]
                for idx, ax in enumerate(axes):
                    img_seq = primary_imgs[idx]
                    img = img_seq[-1] if img_seq.ndim == 4 else img_seq
                    img = img.permute(1, 2, 0).numpy()
                    if img.min() < 0.0:
                        img = (img + 1.0) * 0.5
                    img = np.clip(img, 0.0, 1.0)
                    ax.imshow(img)
                    ax.axis("off")
                    if idx < preds.shape[0]:
                        actual_vec = actual_actions[idx, 0]
                        pred_vec = preds[idx, 0]
                        diff = pred_vec - actual_vec
                        def fmt(v):
                            return ", ".join(f"{x:.2f}" for x in v[:3])
                        text = f"GT: [{fmt(actual_vec)}]\nPred: [{fmt(pred_vec)}]\nDiff: [{fmt(diff)}]"
                        ax.text(0.02, 0.98, text, color="white", fontsize=9, va="top",
                                ha="left", bbox=dict(facecolor="black", alpha=0.45, pad=4))
                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
                buf.seek(0)
                overlay_img = imageio.imread(buf)
                buf.close()
                plt.close(fig)
                vis_log["BatchVis/image_overlays"] = overlay_img
        except Exception as e:
            print(f"[BatchVis] Failed to create overlay visualization: {e}")

        return vis_log

    def compute_mse_visualize(self, trainset, validset, num_samples, savedir=None):
        """If savedir is not None, then also visualize the model predictions and save them to savedir"""
        visualize = savedir is not None

        # set model into eval mode
        self.set_eval()
        eval_data = [d for d in trainset.datasets if "eval" in d.hdf5_path]
        broad_data = [d for d in trainset.datasets if "eval" not in d.hdf5_path]
        if len(eval_data) < int(num_samples / 2):
            training_sampled_data = random.sample(broad_data, int(num_samples))
        elif len(broad_data) < int(num_samples / 2):
            training_sampled_data = random.sample(eval_data, int(num_samples))
        else:
            training_sampled_data = random.sample(eval_data, int(num_samples / 2)) + random.sample(broad_data, int(num_samples / 2))


        if validset is not None:
            random_state = np.random.RandomState(0)
            valid_indices = random_state.choice(
                len(validset.datasets),
                min(len(validset.datasets), num_samples)
            ).astype(int)
            validation_sampled_data = [validset.datasets[idx] for idx in valid_indices]

            inference_datasets_mapping = {"Train": training_sampled_data, "Valid": validation_sampled_data}
        else:
            inference_datasets_mapping = {"Train": training_sampled_data}

        # extract action name for visualization
        action_keys = self.global_config.train.action_keys
        training_sample=training_sampled_data[0][0]
        modified_action_keys = [element.replace("action/", "") for element in action_keys]
        action_names = []

        for i, action_key in enumerate(action_keys):
            if isinstance(training_sample[action_key][0], np.ndarray):
                action_names.extend([f'{modified_action_keys[i]}_{j+1}' for j in range(len(training_sample[action_key][0]))])
            else:
                action_names.append(modified_action_keys[i])

        if visualize:
            print("Saving model prediction plots to {}".format(savedir))

        mse_log = {}
        vis_log = {}
        # loop through training and validation sets
        for inference_key in inference_datasets_mapping:
            actual_actions_all_traj = [] # (NxT, D)
            predicted_actions_all_traj = [] # (NxT, D)

            # loop through each trajectory
            traj_num = 1
            for d in inference_datasets_mapping[inference_key]:
                actual_actions, predicted_actions, images = self.compute_traj_pred_actual_actions(d, return_images=visualize)
                actual_actions_all_traj.append(actual_actions)
                predicted_actions_all_traj.append(predicted_actions)
                if visualize:
                    traj_key = "{}_traj_{}".format(inference_key.lower(), traj_num)
                    save_path = os.path.join(savedir, traj_key + ".png")
                    VisUtils.make_model_prediction_plot(
                        hdf5_path=d.hdf5_path,
                        save_path=save_path,
                        images=images,
                        action_names=action_names,
                        actual_actions=actual_actions,
                        predicted_actions=predicted_actions,
                    )
                    try:
                        vis_log[traj_key] = imageio.imread(save_path)
                    except:
                        pass
                traj_num += 1

            actual_actions_all_traj = np.concatenate(actual_actions_all_traj, axis=0)
            predicted_actions_all_traj = np.concatenate(predicted_actions_all_traj, axis=0)
            accuracy_thresholds = np.logspace(-3,-5, num=3).tolist()
            mse = torch.nn.functional.mse_loss(
                torch.tensor(predicted_actions_all_traj),
                torch.tensor(actual_actions_all_traj),
                reduction='none'
            ) # (NxT, D)
            mse_log[f'{inference_key}/action_mse_error'] = mse.mean().item() # average MSE across all timesteps averaged across all action dimensions (D,)

            # compute percentage of timesteps that have MSE less than the accuracy thresholds
            for accuracy_threshold in accuracy_thresholds:
                mse_log[f'{inference_key}/action_accuracy@{accuracy_threshold}'] = (torch.less(mse,accuracy_threshold).float().mean().item())

        return mse_log, vis_log


class ValueAlgo(Algo):
    """
    Base class for all algorithms that can learn a value function.
    """
    def get_state_value(self, obs_dict, goal_dict=None):
        """
        Get state value outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            value (torch.Tensor): value tensor
        """
        raise NotImplementedError

    def get_state_action_value(self, obs_dict, actions, goal_dict=None):
        """
        Get state-action value outputs.

        Args:
            obs_dict (dict): current observation
            actions (torch.Tensor): action
            goal_dict (dict): (optional) goal

        Returns:
            value (torch.Tensor): value tensor
        """
        raise NotImplementedError


class PlannerAlgo(Algo):
    """
    Base class for all algorithms that can be used for planning subgoals
    conditioned on current observations and potential goal observations.
    """
    def get_subgoal_predictions(self, obs_dict, goal_dict=None):
        """
        Get predicted subgoal outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoal prediction (dict): name -> Tensor [batch_size, ...]
        """
        raise NotImplementedError

    def sample_subgoals(self, obs_dict, goal_dict, num_samples=1):
        """
        For planners that rely on sampling subgoals.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoals (dict): name -> Tensor [batch_size, num_samples, ...]
        """
        raise NotImplementedError


class HierarchicalAlgo(Algo):
    """
    Base class for all hierarchical algorithms that consist of (1) subgoal planning
    and (2) subgoal-conditioned policy learning.
    """
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        raise NotImplementedError

    def get_subgoal_predictions(self, obs_dict, goal_dict=None):
        """
        Get subgoal predictions from high-level subgoal planner.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoal (dict): predicted subgoal
        """
        raise NotImplementedError

    @property
    def current_subgoal(self):
        """
        Get the current subgoal for conditioning the low-level policy

        Returns:
            current subgoal (dict): predicted subgoal
        """
        raise NotImplementedError


class RolloutPolicy(object):
    """
    Wraps @Algo object to make it easy to run policies in a rollout loop.
    """
    def __init__(self, policy, obs_normalization_stats=None, action_normalization_stats=None):
        """
        Args:
            policy (Algo instance): @Algo object to wrap to prepare for rollouts

            obs_normalization_stats (dict): optionally pass a dictionary for observation
                normalization. This should map observation keys to dicts
                with a "mean" and "std" of shape (1, ...) where ... is the default
                shape for the observation.
        """
        self.policy = policy
        self.obs_normalization_stats = obs_normalization_stats
        self.action_normalization_stats = action_normalization_stats

    def start_episode(self):
        """
        Prepare the policy to start a new rollout.
        """
        self.policy.set_eval()
        self.policy.reset()

    def _prepare_observation(self, ob):
        """
        Prepare raw observation dict from environment for policy.

        Args:
            ob (dict): single observation dictionary from environment (no batch dimension,
                and np.array values for each key)
        """
        if self.obs_normalization_stats is not None:
            ob = ObsUtils.normalize_dict(ob, obs_normalization_stats=self.obs_normalization_stats)
        ob = TensorUtils.to_tensor(ob)
        ob = TensorUtils.to_batch(ob)
        ob = TensorUtils.to_device(ob, self.policy.device)
        ob = TensorUtils.to_float(ob)
        return ob

    def __repr__(self):
        """Pretty print network description"""
        return self.policy.__repr__()

    def __call__(self, ob, goal=None):
        """
        Produce action from raw observation dict (and maybe goal dict) from environment.

        Args:
            ob (dict): single observation dictionary from environment (no batch dimension,
                and np.array values for each key)
            goal (dict): goal observation
        """
        ob = self._prepare_observation(ob)
        if goal is not None:
            goal = self._prepare_observation(goal)
        ac = self.policy.get_action(obs_dict=ob, goal_mode=self.goal_mode, eval_mode=self.eval_mode)
        ac = TensorUtils.to_numpy(ac[0])
        if self.action_normalization_stats is not None:
            action_keys = self.policy.global_config.train.action_keys
            action_shapes = {k: self.action_normalization_stats[k]["offset"].shape[1:] for k in self.action_normalization_stats}
            ac_dict = AcUtils.vector_to_action_dict(ac, action_shapes=action_shapes, action_keys=action_keys)
            ac_dict = ObsUtils.unnormalize_dict(ac_dict, normalization_stats=self.action_normalization_stats)
            action_config = self.policy.global_config.train.action_config
            for key, value in ac_dict.items():
                this_format = action_config[key].get("format", None)
                if this_format == "rot_6d":
                    rot_6d = torch.from_numpy(value).unsqueeze(0)
                    conversion_format = action_config[key].get("convert_at_runtime", "rot_axis_angle")
                    if conversion_format == "rot_axis_angle":
                        rot = TorchUtils.rot_6d_to_axis_angle(rot_6d=rot_6d).squeeze().numpy()
                    elif conversion_format == "rot_euler":
                        rot = TorchUtils.rot_6d_to_euler_angles(rot_6d=rot_6d, convention="XYZ").squeeze().numpy()
                    else:
                        raise ValueError
                    ac_dict[key] = rot
            ac = AcUtils.action_dict_to_vector(ac_dict, action_keys=action_keys)
        return ac
