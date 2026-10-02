import logging

import hydra
import torch
from omegaconf import DictConfig

from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class DroidDiffusionAgent(BaseAgent):
    def __init__(
        self,
        model: DictConfig,
        obs_encoders: DictConfig,
        language_encoders: DictConfig,
        optimization: DictConfig,
        obs_seq_len: int,
        act_seq_len: int,
        cam_names: list[str],
        if_robot_states: bool = False,
        if_film_condition: bool = False,
        if_dift_language: bool = False,
        device: str = "cpu",
        state_dim: int = 7,
        latent_dim: int = 64,
    ):
        super().__init__(
            model=model,
            obs_encoders=obs_encoders,
            language_encoders=language_encoders,
            device=device,
            state_dim=state_dim,
            latent_dim=latent_dim,
            obs_seq_len=obs_seq_len,
            act_seq_len=act_seq_len,
            temporal_obs_len=obs_seq_len,
            cam_names=cam_names,
            if_robot_states=if_robot_states,
            if_film_condition=if_film_condition,
            if_dift_language=if_dift_language,
        )

        self.optimizer_config = optimization

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        return optimizer

    def forward(self, obs_dict, actions=None, alignment_context=None):
        perceptual_emb, latent_goal = self.compute_input_embeddings(obs_dict, alignment_context=alignment_context)
        alignment_info = getattr(self, "_last_alignment_loss", None)
        self._last_alignment_loss = None

        raw_alignment_loss = None
        weighted_alignment_loss = None
        if alignment_info is not None:
            if isinstance(alignment_info, tuple):
                raw_alignment_loss, weighted_alignment_loss = alignment_info
            else:
                raw_alignment_loss = alignment_info
                weighted_alignment_loss = alignment_info

        if self.training and actions is not None:
            loss = self.model(perceptual_emb, latent_goal, action=actions, if_train=True)

            total_loss = loss
            raw_aux = None
            weighted_aux = None
            if weighted_alignment_loss is not None:
                total_loss = total_loss + weighted_alignment_loss
                weighted_aux = weighted_alignment_loss.detach()
            if raw_alignment_loss is not None:
                raw_aux = raw_alignment_loss.detach()

            aux = None
            if raw_aux is not None or weighted_aux is not None:
                aux = (raw_aux, weighted_aux)

            return total_loss, aux

        predicted_action = self.model(perceptual_emb, latent_goal, if_train=False)
        return predicted_action
