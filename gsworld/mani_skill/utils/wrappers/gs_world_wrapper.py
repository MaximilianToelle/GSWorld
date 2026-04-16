import copy
from typing import Union
import gymnasium as gym
import torch
import os
import time
import matplotlib.pyplot as plt
import open3d as o3d
from functools import partial

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs import Actor, Link
from mani_skill.utils.structs.pose import Pose

from scene.cameras import Camera
from gaussian_renderer import render
from arguments import PipelineParams

from gsworld.utils.pcd_utils import extract_rigid_transform
from gsworld.utils.gs_utils import transform_gaussians
from gsworld.constants import *
from gsworld.utils.gaussian_merger import GaussianModelMerger

import viser
from gsworld.mani_skill.utils.gsplat_viewer.gsplat_viewer import GsplatViewer
from gsworld.mani_skill.utils.gsplat_viewer.utils_rasterize_render import _viewer_render_fn, _on_connect

from PIL import Image

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


class GSWorldWrapper(gym.Wrapper):
    """This wrapper wraps any maniskill env created via gym.make to add gaussian splattings.
    Also, this wrapper returns the gaussian observation. Code is adapted from original gymnasium wrapper.
    """

    def __init__(
        self,
        env: gym.Env,
        robot_pipe: PipelineParams,
        scene_gs_cfg_name: str = "franka_merger",
        device: Union[str, torch.device] = "cuda",
        log_state: bool = False,
        state_log_path: str = "./exp_log",
        cam_randomization: bool = False,
        use_gsplat_viewer: bool = False,
    ):
        super().__init__(env)
        self.num_envs = env.num_envs
        self.robot_pipe = robot_pipe
        self.device = device
        self.scene_gs_cfg_name = scene_gs_cfg_name
        self.log_state = log_state
        self.cam_randomization = cam_randomization
        ###########################################################
        if "xarm" in self.scene_gs_cfg_name:
            self.gs_semantics = xarm_gs_semantics
            self.sim2gs_arm_trans = sim2gs_xarm_trans
        elif "fr3" in self.scene_gs_cfg_name or "franka" in self.scene_gs_cfg_name:
            self.gs_semantics = fr3_gs_semantics
            self.sim2gs_arm_trans = sim2gs_arm_trans
        elif "r1" in self.scene_gs_cfg_name:
            self.gs_semantics = r1_gs_semantics
            self.sim2gs_arm_trans = sim2gs_r1_trans
        else:
            raise NotImplementedError
        self.sim2gs_arm_trans = torch.tensor(self.sim2gs_arm_trans, dtype=torch.float32, device=self.device)
        self.obj_gs_semantics = obj_gs_semantics
        ###########################################################
        
        # Extract rigid transformation
        self.rigid_sim2real, self.scale_sim2real, R_sim2real, t_sim2real = extract_rigid_transform(self.sim2gs_arm_trans)
        
        # Setup camera
        self.human_cam_config = dict(render_camera=self.base_env.scene.human_render_cameras['render_camera'].get_params())
        self.state_log_path_prefix = state_log_path

        ###########################################################################
        merger = GaussianModelMerger()
        path = os.path.join(CFG_DIR, f"{scene_gs_cfg_name}.json")
        indices = merger.load_models_from_config(path)
        merged_model = merger.merge_models()
        self.initial_merger_robot = merged_model
        ###########################################################################

        env4moving = gym.make(
            "Empty-v1",
            obs_mode="none",
            reward_mode="none",
            enable_shadow=False,
            robot_uids=self.base_env.agent.robot.name,
            render_mode=None,
            sim_config=dict(sim_freq=100, control_freq=20),
            sim_backend="auto",
        )
        env4moving.reset(seed=0)
        self.env4moving: BaseEnv = env4moving.unwrapped

        self.gs_initial_qpos = robot_scan_qpos[self.base_env.agent.robot.name]
        self.task_init_qpos = robot_task_init_qpos[self.base_env.agent.robot.name]
        # ########################################
        self.env4moving.agent.robot.set_qpos(self.gs_initial_qpos)
        self.gs_link_pose_mats = []
        for link in self.env4moving.agent.robot.get_links():
            link_mat = link.pose.to_transformation_matrix().to(self.device)
            self.gs_link_pose_mats.append(link_mat)
        env4moving.close()
        print("finished storing initial qpos and link pose mats")
        # ######################

        self.gs_movable_pts = dict()

        # Precompute semantic masks once — these never change
        semantics = self.initial_merger_robot._semantics.long().squeeze(-1)
        self._semantic_masks = {}   # key -> bool mask (N,)
        self._semantic_indices = {} # key -> index tensor for transform_gaussians
        for link in self.env.unwrapped.agent.robot.get_links():
            if link.name in self.gs_semantics:
                target = torch.tensor(self.gs_semantics[link.name], device=semantics.device).long()
                mask = torch.isin(semantics, target)
                self._semantic_masks[link.name] = mask
                self._semantic_indices[link.name] = torch.where(mask)[0]
        for actor_key, sem_id in self.obj_gs_semantics.items():
            mask = (semantics == sem_id)
            self._semantic_masks[actor_key] = mask
            self._semantic_indices[actor_key] = torch.where(mask)[0]

        # Build Sapien segmentation ID → GS semantic ID lookup table
        max_sapien_id = max(self.base_env.segmentation_id_map.keys()) + 1
        self._sapien_id_to_gs_id = torch.full((max_sapien_id,), -1, dtype=torch.int16, device=self.device)
        for sapien_id, obj in self.base_env.segmentation_id_map.items():
            name = obj.name
            if name in self.gs_semantics:
                gs_id = self.gs_semantics[name]
                if isinstance(gs_id, list):
                    gs_id = gs_id[0]
                self._sapien_id_to_gs_id[sapien_id] = gs_id
            elif name in self.obj_gs_semantics:
                self._sapien_id_to_gs_id[sapien_id] = self.obj_gs_semantics[name]

        # Init Gsplat Viewer
        self.use_gsplat_viewer = use_gsplat_viewer
        if self.use_gsplat_viewer:
            self.gs4viewer = {
                "means": self.initial_merger_robot._xyz.view(-1, 3),
                "quats": self.initial_merger_robot._rotation.view(-1, 4),
                "scales": self.initial_merger_robot._scaling.view(-1, 3),
                "rgb_colors": self.initial_merger_robot._features_dc.view(-1, 3),
                "opacities": self.initial_merger_robot._opacity.view(-1),
            }
            server = viser.ViserServer(port=8081, verbose=False)
            self.viewer = GsplatViewer(
                        server=server,
                        render_fn=lambda camera_state, render_tab_state: _viewer_render_fn(
                            camera_state, 
                            render_tab_state, 
                            self.gs4viewer, 
                            "3dgs", 
                            device
                        ),
                        output_dir=None,
                        mode="training",
                    )
            scene_center = (self.gs4viewer["means"].data
                    .mean(dim=0)
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
            time.sleep(1)
            server.on_client_connect(partial(_on_connect, server=server, scene_center=scene_center))

    def transform_gs_perlink(self, splats, gs_init_qpos=None, target_mat=None, eef_pos=None):
        """Transform gaussian splatting model based on robot joint states"""
        transformed_splats = copy.deepcopy(splats)

        ######################
        for idx, link in enumerate(self.env.unwrapped.agent.robot.get_links()):
            link_mat = link.pose.to_transformation_matrix().to(self.device)
            if "xarm" in self.env.unwrapped.agent.uid:
                for i in range(3):
                    link_mat[:, i, 3] += object_offset["xarm_arm"][i]
            link_trans = self.sim2gs_arm_trans @ link_mat @ torch.linalg.inv(self.gs_link_pose_mats[idx]) @ torch.linalg.inv(self.sim2gs_arm_trans)
            link_xyz, link_scaling, link_rotation, link_opacity = transform_gaussians(
                transformed_splats,
                selected_indices=self._semantic_indices[link.name],
                scale=None,
                rot_mat=link_trans[:, :3, :3],
                translation=link_trans[:, :3, 3],
                new_opacity=None,
                )
            self.gs_movable_pts[link.name] = (link_xyz, link_scaling, link_rotation, link_opacity)

        #######################

        if "actors" in self.env.base_env.get_state_dict():
            for actor_key in self.env.base_env.get_state_dict()['actors']:
                if actor_key not in sim2gs_object_transforms:
                    continue
                pose_tensor = self.env.base_env.get_state_dict()['actors'][actor_key][:, :7]
                mat = Pose.create(pose=pose_tensor).to_transformation_matrix().to(self.device)
                if actor_key in object_offset:
                    for i in range(3):
                        mat[:, i, 3] += object_offset[actor_key][i]

                # Get sim2gs transform for this object
                sim2gs_obj_trans = torch.tensor(sim2gs_object_transforms[actor_key], device=self.device, dtype=torch.float32)
                full_transform = self.sim2gs_arm_trans @ mat @ torch.linalg.inv(sim2gs_obj_trans)

                # Extract and apply rigid transform
                transform, scale, _, _ = extract_rigid_transform(full_transform)

                obj_xyz, obj_scaling, obj_rotation, obj_opacity = transform_gaussians(
                    transformed_splats,
                    selected_indices=self._semantic_indices[actor_key],
                    scale=scale * object_scale[actor_key],
                    rot_mat=transform[:, :3, :3],
                    translation=transform[:, :3, 3],
                    new_opacity=None,
                )
                self.gs_movable_pts[actor_key] = (obj_xyz, obj_scaling, obj_rotation, obj_opacity)
                self.gs_semantics[actor_key] = self.obj_gs_semantics[actor_key]

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped
    
    def get_seg_id2obj_mapping(self):
        for obj_id, obj in sorted(self.base_env.segmentation_id_map.items()):
            if isinstance(obj, Actor):
                print(f"{obj_id}: Actor, name - {obj.name}")
            elif isinstance(obj, Link):
                print(f"{obj_id}: Link, name - {obj.name}")

    def step(self, action):
        if self.use_gsplat_viewer:
            while self.viewer.state == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            tic = time.time()

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.transform_gs_perlink(self.initial_merger_robot)
        #######################################################################
        
        gs_renders = self._render_gsworld()
        # overwriting the env observation with obtained gs renders
        for cam_name, gs_render in gs_renders.items():
            obs['sensor_data'][cam_name]['rgb'] = gs_render['rgb']
            # Remap Sapien segmentation IDs to GS semantic IDs (pixel-perfect, no blending artifacts from GS)
            sapien_seg = obs['sensor_data'][cam_name]['segmentation'].long()
            obs['sensor_data'][cam_name]['segmentation'] = self._sapien_id_to_gs_id[sapien_seg]
            # NOTE: taking perfect depth from sapien instead of worse gs_render depth (assumption: good alignment of gs on sapien objects)
            
            # saving model_matrix to simplify coordinate transformation hell later
            obs["sensor_param"][cam_name]["gl_cam2sapien_world"] = self.env.unwrapped.scene.sensors[cam_name].camera.get_model_matrix()

            # == Debug ==
            #depth = obs['sensor_data'][cam_name]['depth'] / 1000.0  
            #intrinsic = obs["sensor_param"][cam_name]["intrinsic_cv"]
            #pts_world = _debug_depth_to_sapien_world_pcd(depth, intrinsic, obs["sensor_param"][cam_name]["gl_cam2sapien_world"])
            
            #if cam_name == "right_cam":
            #    _debug_visualize_rgb_depth(gs_render['rgb'][0], obs['sensor_data'][cam_name]['depth'][0])
            #    _debug_visualize_pointcloud_tensors(pts_world.reshape(-1, 3), gs_render['rgb'].reshape(-1, 3))
            # === === ===

        # Update gaussians for gsplatviewer 
        if self.use_gsplat_viewer:
            for key in self.gs_movable_pts:
                mask = self._semantic_masks[key]
                self.gs4viewer["means"][mask] = self.gs_movable_pts[key][0].view(-1, 3)
                self.gs4viewer["scales"][mask] = self.gs_movable_pts[key][1].view(-1, 3)
                self.gs4viewer["quats"][mask] = self.gs_movable_pts[key][2].view(-1, 4)
                self.gs4viewer["opacities"][mask] = self.gs_movable_pts[key][3].view(-1)

            self.viewer.rerender(None)
            self.viewer.lock.release()
            num_train_rays_per_step = (3 * 540 * 960)
            num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
            num_train_rays_per_sec = (num_train_rays_per_step * num_train_steps_per_sec)
            self.viewer.render_tab_state.num_train_rays_per_sec = (num_train_rays_per_sec)
            self.viewer.update(info['elapsed_steps'].item(), num_train_rays_per_step)  

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        self.transform_gs_perlink(self.initial_merger_robot)
        ##############################################
        
        gs_renders = self._render_gsworld()
        # overwriting the env observation with obtained gs renders
        for cam_name, gs_render in gs_renders.items():
            obs['sensor_data'][cam_name]['rgb'] = gs_render['rgb']
            # Remap Sapien segmentation IDs to GS semantic IDs (pixel-perfect, no blending artifacts from GS)
            sapien_seg = obs['sensor_data'][cam_name]['segmentation'].long()
            obs['sensor_data'][cam_name]['segmentation'] = self._sapien_id_to_gs_id[sapien_seg]
            
            obs["sensor_param"][cam_name]["gl_cam2sapien_world"] = self.env.unwrapped.scene.sensors[cam_name].camera.get_model_matrix()

        return obs, info

    def render(self):
        ret = self.env.render()
        return ret
    
    def render_current_step(self):
        # code adapted from sapien base env
        info = self.env.unwrapped.get_info()
        obs = self.env.unwrapped.get_obs(info)
        reward = self.unwrapped.get_reward(obs=obs, action=None, info=info)
        if "success" in info:
            if "fail" in info:
                terminated = torch.logical_or(info["success"], info["fail"])
            else:
                terminated = info["success"].clone()
        else:
            if "fail" in info:
                terminated = info["fail"].clone()
            else:
                terminated = torch.zeros(self.num_envs, dtype=bool, device=self.device)

        gs_renders = self._render_gsworld()
        for cam_name, gs_render in gs_renders.items():
            obs['sensor_data'][cam_name]['rgb'] = gs_render['rgb']
            # Remap Sapien segmentation IDs to GS semantic IDs (pixel-perfect, no blending artifacts from GS)
            sapien_seg = obs['sensor_data'][cam_name]['segmentation'].long()
            obs['sensor_data'][cam_name]['segmentation'] = self._sapien_id_to_gs_id[sapien_seg]

            obs["sensor_param"][cam_name]["gl_cam2sapien_world"] = self.env.unwrapped.scene.sensors[cam_name].camera.get_model_matrix()
        
        return (
            obs,
            reward,
            terminated,
            torch.zeros(self.num_envs, dtype=bool, device=self.device),
            info,
        )
        
    def _render_gsworld(self, renderer='gaussian-splatting'):
        # get gs rendering
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        gs_renders = {}
        self.gs_cam = self.cam_maniskill2gs(self.base_env.get_sensor_params(), self.device, cam_names=list(self.base_env.get_sensor_params().keys()))
        for cam_name, cam_param in self.gs_cam.items():
            if renderer == 'gaussian-splatting':
                cam_name_rendering = []
                cam_name_depth = []
                for i in range(self.num_envs):
                    # prepare gaussian model
                    gs4render = copy.deepcopy(self.initial_merger_robot)
                    for key in self.gs_movable_pts:
                        mask = self._semantic_masks[key]
                        if self.gs_movable_pts[key][0].shape[0] == self.num_envs:
                            gs4render._xyz[mask] = self.gs_movable_pts[key][0][i]
                        if self.gs_movable_pts[key][1].shape[0] == self.num_envs:
                            gs4render._scaling[mask] = self.gs_movable_pts[key][1][i]
                        if self.gs_movable_pts[key][2].shape[0] == self.num_envs:
                            gs4render._rotation[mask] = self.gs_movable_pts[key][2][i]
                        if self.gs_movable_pts[key][3].shape[0] == self.num_envs:
                            gs4render._opacity[mask] = self.gs_movable_pts[key][3][i]
                    output = render(cam_param, gs4render, self.robot_pipe, background, 
                                    use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE)
                    rendering = output["render"].detach()
                    gs_render = rendering.permute(1, 2, 0).unsqueeze(dim=0)
                    # Convert gs_render from [0,1] float32 to [0,255] uint8
                    cam_name_rendering.append((gs_render * 255).clamp(0, 255).to(torch.uint8))

                    # Depth 
                    depth = output["depth"].detach()
                    depth = depth / self.scale_sim2real   # NOTE: important to get depth back in SAPIEN metrics!!
                    depth = depth.permute(1, 2, 0).unsqueeze(dim=0)
                    cam_name_depth.append(depth)
            else:
                return ValueError(f"{renderer} is not supported. Official renderer also supports gsplat.")
            gs_renders.update({
                cam_name: {
                "rgb": torch.vstack(cam_name_rendering),   # (num_envs, H, W, 3)
                "depth": torch.vstack(cam_name_depth),     # (num_envs, H, W, 1)
                }
            })    

        return gs_renders
        
    def cam_maniskill2gs(self, config_maniskill: dict, device: Union[str, torch.device], cam_names: list) -> Camera:
        cam_params = {}
        for cam_name in cam_names:
            extrinsic_rt_cv = config_maniskill[cam_name]['extrinsic_cv'][0].to(device)
            intrinsic_k = config_maniskill[cam_name]['intrinsic_cv'][0].to(device)
            
            sim_world2cam = torch.vstack([extrinsic_rt_cv, torch.tensor([0, 0, 0, 1], dtype=extrinsic_rt_cv.dtype, device=extrinsic_rt_cv.device)])
            sim_cam2world = torch.linalg.inv(sim_world2cam)

            fx = intrinsic_k[0, 0]
            fy = intrinsic_k[1, 1]
            img_w = self.base_env.get_sensor_images()[cam_name]['rgb'].shape[2]
            img_h = self.base_env.get_sensor_images()[cam_name]['rgb'].shape[1]

            resolution = (img_w, img_h)

            fovx = 2 * torch.arctan(img_w / (2*fx))
            fovy = 2 * torch.arctan(img_h / (2*fy))

            # Transform the camera pose to real-world frame
            real_cam2world = sim_cam2world
            real_cam2world[:3, 3] = real_cam2world[:3, 3] * self.scale_sim2real
            real_world2cam = torch.linalg.inv(self.rigid_sim2real @ real_cam2world)
            R = real_world2cam[:3, :3].T
            T = real_world2cam[:3, 3]

            invdepthmap = None
            img = Image.new('RGB', (img_w, img_h), color='white')
            img_name = cam_name
            colmap_id = 0
            uid = 0

            cam_param = Camera(
                resolution=resolution,
                colmap_id=colmap_id,
                R=R.cpu().numpy(),
                T=T.cpu().numpy(),
                FoVx=float(fovx),
                FoVy=float(fovy),
                depth_params=None,
                image=img,
                invdepthmap=invdepthmap,
                image_name=img_name,
                uid=uid,
                data_device=device,
            )
            cam_params.update({cam_name: cam_param})

        return cam_params


def _debug_depth_to_sapien_world_pcd(depth, intrinsic, gl_cam2sapien_world):
    """
    Converts batched depth to pcd in sapien world coordinates assuming graphics convention (x-axis right, y-axis upward, z-axis backward) in extrinsics. 
    All inputs/outputs are torch tensors on the same device.
    """
    
    B, H, W = depth.shape[:3]
    device = depth.device

    u, v = torch.meshgrid(torch.arange(W, device=device, dtype=depth.dtype) + 0.5,  # half-pixel offset to model pixel center depth
                          torch.arange(H, device=device, dtype=depth.dtype) + 0.5,
                          indexing='xy')

    fx, fy = intrinsic[:, 0, 0].view(B, 1, 1), intrinsic[:, 1, 1].view(B, 1, 1)
    cx, cy = intrinsic[:, 0, 2].view(B, 1, 1), intrinsic[:, 1, 2].view(B, 1, 1)

    d = depth[..., 0]
    
    z = - d     # assuming positive depth values
    x = (u - cx) * d / fx   # X is right -> (u - cx) is positive on the right
    y = -(v - cy) * d / fy   # Y is upward -> (v - cy) is positive downward!

    pts_cam = torch.stack([x, y, z], dim=-1).reshape(B, -1, 3)

    rotation_mats = gl_cam2sapien_world[:, :3, :3]
    translation_vecs = gl_cam2sapien_world[:, :3, 3].unsqueeze(1)
    
    pts_world = torch.matmul(pts_cam, rotation_mats.transpose(-2, -1)) + translation_vecs

    return pts_world


def _debug_visualize_pointcloud_tensors(merged_pts: torch.Tensor, merged_rgb: torch.Tensor) -> None:
    """
    Quickly visualizes GPU point cloud tensors using Open3D.
    """
    # 1. Detach from graph, move to CPU, and convert to contiguous NumPy arrays
    #    Using .contiguous() ensures memory layout compatibility with Open3D
    pts_np = merged_pts.detach().cpu().contiguous().numpy()
    rgb_np = merged_rgb.detach().cpu().contiguous().numpy()

    # 2. Normalize colors to [0.0, 1.0] if they are in [0, 255]
    #    Open3D expects float64 colors in the [0, 1] range.
    if rgb_np.max() > 1.0 or rgb_np.dtype == np.uint8:
        rgb_np = rgb_np.astype(np.float64) / 255.0
    else:
        rgb_np = rgb_np.astype(np.float64)

    # 3. Cast points to float64 (Open3D Vector3dVector requirement)
    pts_np = pts_np.astype(np.float64)

    # 4. Instantiate the Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    
    # 5. Assign vertices and colors via Open3D's utility vectors
    pcd.points = o3d.utility.Vector3dVector(pts_np)
    # pcd.colors = o3d.utility.Vector3dVector(rgb_np)

    # 6. Launch the interactive visualizer (this will block execution until closed)
    print(f"Visualizing {pts_np.shape[0]} points. Close the window to continue execution.")
    o3d.visualization.draw_geometries([pcd])


def _debug_visualize_rgb_depth(
    rgb: torch.Tensor | np.ndarray, 
    depth: torch.Tensor | np.ndarray
) -> None:
    """
    Visualizes RGB and Depth images side-by-side.
    Safely handles PyTorch tensors (GPU/CPU) and NumPy arrays.
    """
    # 1. Convert PyTorch tensors to NumPy arrays safely
    if isinstance(rgb, torch.Tensor):
        rgb_np = rgb.detach().cpu().numpy()
    else:
        rgb_np = rgb
        
    if isinstance(depth, torch.Tensor):
        depth_np = depth.detach().cpu().numpy()
    else:
        depth_np = depth

    # 2. Handle shapes
    # Squeeze depth from (H, W, 1) to (H, W) for matplotlib
    if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
        depth_np = depth_np.squeeze(-1)
        
    # Ensure RGB is (H, W, 3) just in case it was passed as (C, H, W)
    if rgb_np.ndim == 3 and rgb_np.shape[0] == 3: 
        rgb_np = np.transpose(rgb_np, (1, 2, 0))
        
    # 3. Handle data ranges for RGB
    # Matplotlib expects float arrays to be strictly in [0.0, 1.0]
    if rgb_np.dtype != np.uint8:
        rgb_np = np.clip(rgb_np, 0.0, 1.0)

    # 4. Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot RGB
    axes[0].imshow(rgb_np)
    axes[0].set_title("RGB Image")
    axes[0].axis('off')
    
    # Plot Depth
    # Using 'plasma' or 'viridis' helps differentiate subtle depth changes
    # where the "smearing" artifacts usually hide.
    im_depth = axes[1].imshow(depth_np, cmap='plasma') 
    axes[1].set_title("Depth Map")
    axes[1].axis('off')
    
    # Add a colorbar to read the exact depth values
    cbar = fig.colorbar(im_depth, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Depth')
    
    plt.tight_layout()
    plt.show() 