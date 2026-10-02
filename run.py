import logging
import random
import os
import warnings
import inspect
import hydra
import numpy as np
import multiprocessing as mp
mp.set_start_method("spawn", force=True)


from datetime import timedelta
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist

from agents.utils.sim_path import sim_framework_path

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def init_distributed():
    """Initialize torch.distributed if launched with torchrun.

    Returns:
        is_distributed (bool), rank (int), local_rank (int), world_size (int)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        # Extend timeout so long-running eval phases do not trip the NCCL watchdog.
        timeout_minutes_str = os.environ.get("TORCH_DIST_TIMEOUT_MINUTES", "360")
        try:
            timeout_minutes = float(timeout_minutes_str)
        except ValueError:
            timeout_minutes = 60.0
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=timeout_minutes),
        )
        return True, rank, local_rank, world_size
    return False, 0, 0, 1


@hydra.main(config_path="configs", config_name="libero_config.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Initialize distributed and set per-rank behavior
    is_dist, rank, local_rank, world_size = init_distributed()

    # Ensure non-master disables wandb before importing it
    if is_dist and rank != 0:
        os.environ.setdefault("WANDB_MODE", "disabled")

    import wandb  # import after possibly setting WANDB_MODE

    # Seed per rank for determinism and diversity
    set_seed_everywhere(int(cfg.seed) + int(rank))

    # init wandb logger and config from hydra path
    if (not is_dist) or rank == 0:
        wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        print(f"directory: {os.getcwd()}")
        wandb_name = None
        try:
            wandb_cfg = cfg.get("wandb")
            if wandb_cfg and "name" in wandb_cfg and wandb_cfg.name is not None:
                wandb_name = wandb_cfg.name
        except Exception:
            wandb_name = None
        wandb_id = None
        wandb_resume = None
        try:
            wandb_cfg = cfg.get("wandb")
            if wandb_cfg and "id" in wandb_cfg and wandb_cfg.id is not None:
                wandb_id = wandb_cfg.id
            if wandb_cfg and "resume" in wandb_cfg and wandb_cfg.resume is not None:
                wandb_resume = wandb_cfg.resume
        except Exception:
            wandb_id = None
            wandb_resume = None

        run = wandb.init(
            dir=os.getcwd(),
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.group,
            name=wandb_name,
            id=wandb_id,
            resume=wandb_resume,
            config=wandb.config,
            reinit=True,
        )
    else:
        run = None

    # train the agent
    agent = hydra.utils.instantiate(cfg.agents)
    trainer = hydra.utils.instantiate(cfg.trainers)

    # Visualization is now handled automatically by the trainer
    # See trainers/base_trainer.py _init_visualizer and _run_visualization methods

    # Only main process logs params to wandb
    agent.get_params(wandb_log=((not is_dist) or rank == 0))
    trainer.main(agent)

    # After training, sync all ranks to ensure no more collectives will run.
    if is_dist:
        dist.barrier()
        # Non-master ranks exit immediately to avoid waiting during long eval/sim
        if rank != 0:
            dist.destroy_process_group()
            return

    skip_final_sim = cfg.get("skip_final_sim", False)
    if isinstance(skip_final_sim, str):
        skip_final_sim = skip_final_sim.lower() in {"1", "true", "yes", "y"}

    # Only main process runs simulation and final logging
    if (not is_dist) or rank == 0:
        if not skip_final_sim:
            env_sim = hydra.utils.instantiate(cfg.simulation)
            sim_kwargs = {}
            final_step = getattr(trainer, "epoch", None)
            try:
                sim_signature = inspect.signature(env_sim.test_agent)
            except (TypeError, ValueError):
                sim_signature = None
            save_sim_videos = cfg.get("save_sim_videos", True)
            if sim_signature and "save_videos" in sim_signature.parameters:
                sim_kwargs["save_videos"] = bool(save_sim_videos)
                if save_sim_videos:
                    sim_video_root = os.environ.get("ROBOT_DIFT_POLICY_OUTPUT_DIR") or getattr(agent, "working_dir", os.getcwd())
                    sim_kwargs["video_dir"] = sim_video_root
            if sim_signature and "agent_config" in sim_signature.parameters:
                sim_kwargs["agent_config"] = cfg.agents
            if sim_signature and "epoch" in sim_signature.parameters and final_step is not None:
                sim_kwargs["epoch"] = final_step
            if hasattr(env_sim, "get_task_embs"):
                trainset = getattr(trainer, "trainset", None)
                task_embs = getattr(trainset, "tasks", None) if trainset is not None else None
                if task_embs is not None:
                    env_sim.get_task_embs(task_embs)
            sim_metrics = env_sim.test_agent(agent, **sim_kwargs)
            if sim_metrics:
                log.info("Final simulation metrics: %s", sim_metrics)
                if final_step is not None:
                    sim_metrics.setdefault("simulation_epoch", final_step)
                    sim_metrics.setdefault("epoch", final_step)
                wandb.log(sim_metrics)
        else:
            log.info("Skipping final simulation because skip_final_sim=True")

        log.info("Training done")
        log.info("state_dict saved in {}".format(agent.working_dir))

        wandb.finish()
        wandb.run = None

    # Clean up process group (rank 0 only at this point)
    if is_dist:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
