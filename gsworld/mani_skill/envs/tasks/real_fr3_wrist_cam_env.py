from typing import Dict

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from gsworld.constants import rs_d435i_rgb_k


@register_env("RealFr3WristCam-v1", max_episode_steps=200000)
class RealFr3WristCam(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none", "dense", "sparse"]
    """
    This is just a dummy environment for showcasing robots in a empty scene
    """

    def __init__(self, *args, robot_uids="fr3_umi_wrist435_modified", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig("wrist_cam", sapien.Pose(), width=640, height=480, intrinsic=rs_d435i_rgb_k, near=0.01, far=100, mount=self.agent.robot.find_link_by_name("camera_link")),
        ]

    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at([1.25, -1.25, 1.5], [0.0, 0.0, 0.2])
        # return CameraConfig("render_camera", pose, 2048, 2048, 1, 0.01, 100)
        pose = sapien_utils.look_at([1, 0.2, 0.5], [0.0, 0.0, 0.15])
        return CameraConfig(
            "render_camera", pose=pose, width=640, height=480, intrinsic=rs_d435i_rgb_k, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        # default Pose([0, 0, 0], [1, 0, 0, 0])
        super()._load_agent(options, sapien.Pose())

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene)
        self.ground.set_collision_group_bit(group=2, bit_idx=30, bit=1)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        pass

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: Dict):
        return dict()