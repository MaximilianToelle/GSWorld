from typing import Tuple

import torch

import viser
from nerfview import CameraState, RenderTabState, apply_float_colormap
from gsplat.rendering import rasterization, rasterization_2dgs

from .gsplat_viewer import GsplatRenderTabState


def rasterize_splats(
    splats,
    world_to_cam: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    model_type: str,        # "3dgs" or "2dgs"
    rasterize_mode = "classic",
    camera_model = "pinhole",
    render_mode = "RGB+ED",     # depth loss included
    **kwargs,
) -> Tuple:
    """
    Info-Dict has keys:
        ['batch_ids', 
        'camera_ids', 
        'gaussian_ids', 
        'radii',
        'means2d', 
        'depths', 
        'conics', 
        'opacities', 
        'tile_width', 
        'tile_height', 
        'tiles_per_gauss', 
        'isect_ids', 
        'flatten_ids', 
        'isect_offsets', 
        'width', 
        'height', 
        'tile_size', 
        'n_batches', 
        'n_cameras'])

    """
    means = splats["means"]  # [N, 3]
    # rasterization does normalization internally
    quats = splats["quats"]  # [N, 4]
    scales = torch.exp(splats["scales"])  # [N, 3]
    opacities = torch.sigmoid(splats["opacities"])  # [N,]
    colors = splats["rgb_colors"] # [N, 3]

    if model_type == "3dgs":
        (
        render_colors, 
        render_alphas, 
        info
        ) = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=world_to_cam,    # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=False,
            render_mode=render_mode,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode=rasterize_mode,
            distributed=False,
            camera_model=camera_model,
            with_ut=False,
            with_eval3d=False,
            **kwargs,
        )
    elif model_type == "2dgs":
        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=world_to_cam,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            render_mode=render_mode,
            packed=False,
            absgrad=False,
            sparse_grad=False,
            distloss=False,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type == "3dgs":
        return render_colors, render_alphas, info, None, None, None, None
    elif model_type == "2dgs":
        return render_colors, render_alphas, info, render_normals, normals_from_depth, render_distort, render_median


@torch.no_grad()
def _viewer_render_fn(
    camera_state: CameraState, 
    render_tab_state: RenderTabState,
    splats, 
    model_type,
    device,
):
    assert isinstance(render_tab_state, GsplatRenderTabState)
    if render_tab_state.preview_render:
        width = render_tab_state.render_width
        height = render_tab_state.render_height
    else:
        width = render_tab_state.viewer_width
        height = render_tab_state.viewer_height
    c2w = camera_state.c2w
    K = camera_state.get_K((width, height))
    c2w = torch.from_numpy(c2w).float().to(device)
    K = torch.from_numpy(K).float().to(device)

    RENDER_MODE_MAP = {
        "rgb": "RGB",
        "depth(accumulated)": "D",
        "depth(expected)": "ED",
        "alpha": "RGB",
    }

    render_colors, render_alphas, info, _, _, _, _ = rasterize_splats(
        splats,
        world_to_cam=torch.linalg.inv(c2w)[None],
        Ks=K[None],
        width=width,
        height=height,
        model_type=model_type,
        near_plane=render_tab_state.near_plane,
        far_plane=render_tab_state.far_plane,
        radius_clip=render_tab_state.radius_clip,
        eps2d=render_tab_state.eps2d,
        backgrounds=torch.tensor([render_tab_state.backgrounds], device=device)
        / 255.0,
        render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
        rasterize_mode=render_tab_state.rasterize_mode,
        camera_model=render_tab_state.camera_model,
    )  # [1, H, W, 3]
    render_tab_state.total_gs_count = len(splats["means"])
    render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

    if render_tab_state.render_mode == "rgb":
        # colors represented with sh are not guranteed to be in [0, 1]
        render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
        renders = render_colors.cpu().numpy()
    elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
        # normalize depth to [0, 1]
        depth = render_colors[0, ..., 0:1]
        if render_tab_state.normalize_nearfar:
            near_plane = render_tab_state.near_plane
            far_plane = render_tab_state.far_plane
        else:
            near_plane = depth.min()
            far_plane = depth.max()
        depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
        depth_norm = torch.clip(depth_norm, 0, 1)
        if render_tab_state.inverse:
            depth_norm = 1 - depth_norm
        renders = (
            apply_float_colormap(depth_norm, render_tab_state.colormap)
            .cpu()
            .numpy()
        )
    elif render_tab_state.render_mode == "alpha":
        alpha = render_alphas[0, ..., 0:1]
        if render_tab_state.inverse:
            alpha = 1 - alpha
        renders = (
            apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
        )
    return renders


def _on_connect(
    client: viser.ClientHandle, 
    server: viser.ViserServer, 
    scene_center
) -> None:
    # server.scene.world_axes.visible = True
    server.scene.set_up_direction("+z")
    client.camera.look_at = tuple(scene_center)
