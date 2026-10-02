import logging
import abc
import os
from pathlib import Path
from typing import Optional

from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class BaseSim(abc.ABC):

    def __init__(
            self,
            seed: int,
            device: str,
            render: bool = True,
            n_cores: int = 1,
            if_vision: bool = False
    ):
        self.seed = seed
        self.device = device
        self.render = render
        self.n_cores = n_cores

        self.if_vision = if_vision

        self.working_dir = os.getcwd()
        self.env_name = 'BaseEnvironment'

    def _finalize_episode_video(
        self,
        temp_path: Optional[Path],
        episode_success: bool,
    ) -> Optional[Path]:
        if temp_path is None:
            return None
        status = "success" if episode_success else "fail"
        final_path = temp_path.with_name(f"{temp_path.stem}_{status}{temp_path.suffix}")
        if final_path.exists():
            final_path.unlink(missing_ok=True)
        temp_path.rename(final_path)
        return final_path

    @abc.abstractmethod
    def test_agent(self, agent: BaseAgent, cpu_set):
        pass
