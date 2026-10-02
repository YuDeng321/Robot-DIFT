import logging
import os
import cv2
import random
import numpy as np
import torch
import wandb
import hydra
import multiprocessing as mp
import re
import sys
import types
from .base_sim import BaseSim
# from libero.libero.envs import *
from tqdm import tqdm


def _install_robosuite_single_arm_compat() -> None:
    """Provide the robosuite<=1.4 SingleArmEnv import path for LIBERO."""
    import robosuite

    if not hasattr(robosuite, "load_controller_config") and hasattr(
        robosuite, "load_composite_controller_config"
    ):
        def _load_controller_config(*, default_controller=None, custom_fpath=None):
            del default_controller
            if custom_fpath is not None:
                return robosuite.load_composite_controller_config(controller=custom_fpath)
            return robosuite.load_composite_controller_config(controller=None, robot="Panda")

        robosuite.load_controller_config = _load_controller_config

    robot_module_name = "robosuite.robots.single_arm"
    if robot_module_name not in sys.modules:
        try:
            __import__(robot_module_name, fromlist=["SingleArm"])
        except ModuleNotFoundError:
            from robosuite.robots.fixed_base_robot import FixedBaseRobot

            class SingleArm(FixedBaseRobot):
                """robosuite>=1.5 replacement for the legacy SingleArm robot class."""

            robot_module = types.ModuleType(robot_module_name)
            robot_module.SingleArm = SingleArm
            sys.modules[robot_module_name] = robot_module

    module_name = "robosuite.environments.manipulation.single_arm_env"
    if module_name in sys.modules:
        return
    try:
        __import__(module_name, fromlist=["SingleArmEnv"])
        return
    except ModuleNotFoundError:
        pass

    from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
    from robosuite.utils.transform_utils import mat2quat

    class SingleArmEnv(ManipulationEnv):
        """Compatibility layer for LIBERO on robosuite>=1.5."""

        def __init__(self, *args, mount_types=None, base_types=None, **kwargs):
            if base_types is None and mount_types is not None:
                base_types = mount_types
            super().__init__(*args, base_types=base_types or "default", **kwargs)
            self._compat_seed_value = getattr(self, "seed", None)
            self.seed = self._compat_seed

        def _load_model(self):
            super()._load_model()
            robot = self.robots[0]
            arms = getattr(robot, "arms", ["right"])
            assert len(arms) == 1, f"Error: Expected one single-armed robot, got arms={arms}"

        def _check_robot_configuration(self, robots):
            super()._check_robot_configuration(robots)
            if isinstance(robots, list):
                assert len(robots) == 1, "Error: Only one robot should be inputted for this task!"

        def _compat_seed(self, seed=None):
            self._compat_seed_value = seed
            self.rng = np.random.default_rng(seed)
            np.random.seed(seed)
            random.seed(seed)
            return [seed] if seed is not None else None

        def _eef_site_id(self):
            robot = self.robots[0]
            site_id = robot.eef_site_id
            if isinstance(site_id, dict):
                arm = getattr(robot, "arms", ["right"])[0]
                site_id = site_id[arm]
            return site_id

        @property
        def _eef_xpos(self):
            return np.array(self.sim.data.site_xpos[self._eef_site_id()])

        @property
        def _eef_xmat(self):
            return np.array(self.sim.data.site_xmat[self._eef_site_id()]).reshape(3, 3)

        @property
        def _eef_xquat(self):
            return mat2quat(self._eef_xmat)

    compat_module = types.ModuleType(module_name)
    compat_module.SingleArmEnv = SingleArmEnv
    sys.modules[module_name] = compat_module


_install_robosuite_single_arm_compat()

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.robots.mounted_panda import MountedPanda
from libero.libero.envs.robots.on_the_ground_panda import OnTheGroundPanda

log = logging.getLogger(__name__)


def _patch_libero_robot_models() -> None:
    """Adapt LIBERO's robosuite<=1.4 robot models to robosuite>=1.5."""
    for robot_cls in (MountedPanda, OnTheGroundPanda):
        robot_cls.arms = ["right"]

        old_mount = getattr(robot_cls, "default_mount", None)
        old_gripper = getattr(robot_cls, "default_gripper", None)
        old_controller = getattr(robot_cls, "default_controller_config", None)

        def _default_base(self, old=old_mount):
            value = old.fget(self) if isinstance(old, property) else "RethinkMount"
            return "NullMount" if value is None else value

        def _default_gripper(self, old=old_gripper):
            value = old.fget(self) if isinstance(old, property) else "PandaGripper"
            return value if isinstance(value, dict) else {"right": value}

        def _default_controller_config(self, old=old_controller):
            value = old.fget(self) if isinstance(old, property) else "default_panda"
            return value if isinstance(value, dict) else {"right": value}

        robot_cls.default_base = property(_default_base)
        robot_cls.default_gripper = property(_default_gripper)
        robot_cls.default_controller_config = property(_default_controller_config)


_patch_libero_robot_models()


def _load_libero_init_states(task_suite, context):
    """Load LIBERO init states under torch>=2.6, which changed torch.load defaults."""
    original_load = torch.load

    def _compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = _compat_load
    try:
        return task_suite.get_task_init_states(context)
    finally:
        torch.load = original_load


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


def process_image_input(img_tensor):
    # return (img_tensor / 255. - 0.5) * 2.
    return img_tensor / 255.


class MultiTaskSim(BaseSim):
    def __init__(self,
                 num_episode,
                 max_step_per_episode,
                 task_suite: str,
                 use_eye_in_hand: bool,
                 seed,
                 device,
                 render,
                 n_cores,
                 use_task_emb: bool = True,
                 use_multiprocessing=True):
        super().__init__(seed, device, render, n_cores)

        # according to the task_id, load the corresponding bddl file
        self.task_suite = task_suite

        self.use_eye_in_hand = use_eye_in_hand
        self.render = render

        self.num_episode = num_episode
        self.max_step_per_episode = max_step_per_episode

        self.success_rate = 0
        self.use_multiprocessing = use_multiprocessing
        self.use_task_emb = use_task_emb
        self._task_lang_cache: dict[str, str] = {}

    def reverse_rgb_channels(self, test_img):

        test_img = test_img[::-1, ::-1, :]
        # cv2.imshow("test_img", test_img)
        # cv2.waitKey(0)

        return np.ascontiguousarray(test_img)

    def _get_task_language(self, bddl_path: str) -> str:
        cached = self._task_lang_cache.get(bddl_path)
        if cached is not None:
            return cached

        fallback = os.path.splitext(os.path.basename(bddl_path))[0].replace("_", " ")
        try:
            with open(bddl_path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            log.warning("Failed to read BDDL file %s: %s", bddl_path, exc)
            self._task_lang_cache[bddl_path] = fallback
            return fallback

        match = re.search(r"\(:language\s+(.+?)\)", text)
        if match:
            lang = match.group(1).strip()
        else:
            log.warning("No language entry found in BDDL file: %s", bddl_path)
            lang = fallback

        self._task_lang_cache[bddl_path] = lang
        return lang

    def eval_agent(self,
                   contexts,
                   context_ind,
                   success,
                   episode_lengths,
                   pid,
                   cpu_set,
                   counter,
                   agent=None,
                   agent_config=None,
                   model_states=None):
        # Only set CPU affinity if using multiprocessing
        # if self.use_multiprocessing:
        #     print(os.getpid(), cpu_set)
        #     assign_process_to_cpu(os.getpid(), cpu_set)

        # Handle agent initialization based on input type
        if agent_config is not None:
            # Case 1: Initialize agent from config and states
            assert model_states is not None, "model_states must be provided when using agent_config"
            agent = hydra.utils.instantiate(agent_config)
            agent.recover_model_state(
                model_states['model'],
                model_states['scaler']
            )
        else:
            # Case 2: Use provided agent directly
            assert agent is not None, "Either agent or (agent_config + states) must be provided"

        # print(contexts)

        for i, context in enumerate(contexts):

            task_suite = benchmark.get_benchmark_dict()[self.task_suite]()

            task_bddl_file = task_suite.get_task_bddl_file_path(context)

            file_name = os.path.basename(task_bddl_file).split('.')[0]

            task_emb = None
            if self.use_task_emb and self.task_embs is not None:
                task_emb = self.task_embs[file_name].to(self.device).unsqueeze(0)
            task_lang = self._get_task_language(task_bddl_file)

            # goal_images = self.goal_dicts[file_name]
            # goal_image = random.choice(goal_images)

            init_states = _load_libero_init_states(task_suite, context)

            env_args = {
                "bddl_file_name": task_bddl_file,
                "camera_heights": 128,
                "camera_widths": 128
            }

            env = OffScreenRenderEnv(**env_args)

            agent.reset()
            env.seed(self.seed)
            env.reset()
            obs = env.set_init_state(init_state=init_states[context_ind[i]])

            # dummy actions all zeros for initial physics simulation
            dummy = np.zeros(7)
            dummy[-1] = -1.0  # set the last action to -1 to open the gripper
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)

            # multiprocessing simulation
            for j in range(self.max_step_per_episode):
                agentview_rgb = torch.from_numpy(obs["agentview_image"]).to(self.device).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.
                eye_in_hand_rgb = torch.from_numpy(obs["robot0_eye_in_hand_image"]).to(self.device).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.

                joint_state = obs["robot0_joint_pos"]
                gripper_state = obs["robot0_gripper_qpos"]

                robot_states = torch.from_numpy(np.concatenate([joint_state, gripper_state], axis=-1)).to(self.device).float().unsqueeze(0).unsqueeze(0)

                # save_path = os.path.join("debug_images", f"{self.task_suite}", "images")
                # img = env.sim.render(camera_name="frontview", width=1280, height=800)[..., ::-1]
                # img = np.flip(img, axis=0)
                # cv2.imwrite(os.path.join(save_path, f"agentview_{context}_{context_ind[i]}_{j}.png"), img)

                # agentview_rgb = self.reverse_rgb_channels(agentview_rgb)
                # eye_in_hand_rgb = self.reverse_rgb_channels(eye_in_hand_rgb)

                obs_dict = {"agentview_image": agentview_rgb,
                            "eye_in_hand_image": eye_in_hand_rgb,
                            "lang": [task_lang],
                            "robot_states": robot_states}
                if task_emb is not None:
                    obs_dict["lang_emb"] = task_emb

                action = agent.predict(obs_dict).cpu().numpy()
                obs, r, done, _ = env.step(action)

                # if self.render:
                # env.render()

                if r == 1:
                    success[context, context_ind[i]] = r
                    episode_lengths[context, context_ind[i]] = j + 1
                    break

            if success[context, context_ind[i]] == 0:
                episode_lengths[context, context_ind[i]] = self.max_step_per_episode

            if hasattr(counter, 'get_lock'):  # If it's a multiprocessing Value
                with counter.get_lock():
                    counter.value += 1
                    current_count = counter.value
            else:  # If it's a simple object with value attribute (single process)
                counter.value += 1
                current_count = counter.value
                counter.update()

            mask = episode_lengths.flatten() != 0
            completed_success = success.flatten()[mask]
            completed_lengths = episode_lengths.flatten()[mask]
            average_success = torch.mean(completed_success).item()
            average_episode_length = torch.mean(completed_lengths).item()
            log.info(f'completed_success {completed_success}')
            log.info(f'completed_lengths {completed_lengths}')
            log.info(f'average success rate: {average_success}')
            log.info(f'average episode length: {average_episode_length}')

            env.close()

    def get_task_embs(self, task_embs):
        self.task_embs = task_embs

    def test_agent(
        self,
        agent,
        agent_config,
        cpu_set=None,
        epoch=None,
        save_videos: bool = False,
        video_dir: str | None = None,
        video_fps: int = 20,
    ):
        logging.info("Start testing agent")

        if save_videos:
            log.warning("MultiTaskSim does not implement video recording yet; ignoring save_videos request.")

        if cpu_set is None:
            num_cpu = self.n_cores
            cpu_set = [i for i in range(num_cpu)]
        else:
            num_cpu = len(cpu_set)

        if self.use_multiprocessing:
            log.info("there is {} cpus".format(num_cpu))
        else:
            log.info("not using multiprocessing, run on 1 cpu")

        if self.task_suite == "libero_90":
            num_tasks = 90
        else:
            num_tasks = 10

        success = torch.zeros([num_tasks, self.num_episode]).share_memory_()
        episode_lengths = torch.zeros([num_tasks, self.num_episode]).share_memory_()
        all_runs = num_tasks * self.num_episode

        contexts = np.arange(num_tasks)
        contexts = np.repeat(contexts, self.num_episode)

        context_ind = np.arange(self.num_episode)
        context_ind = np.tile(context_ind, num_tasks)

        if not self.use_multiprocessing:
            # Single process execution
            pbar = tqdm(total=all_runs, desc="Testing agent")
            counter = type('Counter', (), {'value': 0})()  # Simple counter object

            def update_pbar():
                pbar.update(1)

            counter.update = update_pbar  # Add update method to counter

            self.eval_agent(
                contexts=contexts,
                context_ind=context_ind,
                success=success,
                episode_lengths=episode_lengths,
                pid=0,
                cpu_set=set(cpu_set),
                counter=counter,
                agent=agent
            )
            pbar.close()
        else:
            repeat_num = all_runs // num_cpu
            repeat_res = all_runs % num_cpu

            workload_array = np.ones([num_cpu], dtype=int)
            workload_array[:repeat_res] += repeat_num
            workload_array[repeat_res:] = repeat_num

            assert np.sum(workload_array) == all_runs

            ind_workload = np.cumsum(workload_array)
            ind_workload = np.concatenate([[0], ind_workload])
            ###################################################################
            ctx = mp.get_context('spawn')
            processes_list = []

            all_runs = num_tasks * self.num_episode
            counter = ctx.Value('i', 0) #create a shared counter for progress bar
            pbar = tqdm(total=all_runs, desc="Testing agent")

            # Create shared memory state dictionaries for all models
            model_states = agent.get_model_state
            shared_states = {
                'model': {},
                'scaler': model_states[1]  # Assuming scaler is the 4th element
            }

            # Share memory for each state dictionary
            for key, tensor in model_states[0].items():
                shared_states['model'][key] = tensor.share_memory_()

            for i in range(self.n_cores):
                p = ctx.Process(target=self.eval_agent,
                                kwargs={  # Now passing single parameter
                                    "contexts": contexts[ind_workload[i]:ind_workload[i + 1]],
                                    "context_ind": context_ind[ind_workload[i]:ind_workload[i + 1]],
                                    "success": success,
                                    "episode_lengths": episode_lengths,
                                    "pid": i,
                                    "cpu_set": set(cpu_set[i:i + 1]),
                                    "counter": counter,
                                    "agent": None,
                                    "agent_config": agent_config,
                                    "model_states": shared_states,
                                },
                                )
                p.start()
                processes_list.append(p)

            # Monitor progress and update bar
            last_counter = 0
            while any(p.is_alive() for p in processes_list):
                if counter.value > last_counter:
                    pbar.update(counter.value - last_counter)
                    last_counter = counter.value

            [p.join() for p in processes_list]
            pbar.close()

        success_rate = torch.mean(success, dim=-1)
        average_success = torch.mean(success_rate).item()

        print(f'success array {success.detach()}')

        custom_step = f"{epoch}_custom_step"
        wandb.define_metric(custom_step)
        wandb.define_metric(f"{epoch}_tasks_success", step_metric=custom_step)

        for num in range(num_tasks):
            log.info(f"Task {num}: {success_rate[num].item()}")

            wandb.log({custom_step: num,
                       f"{epoch}_tasks_success": success_rate[num].item()
                       })

        wandb.log({f"epoch{epoch}_average_success": average_success})
        log.info(f"Average success rate: {average_success}")
