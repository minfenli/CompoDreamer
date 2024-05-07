import argparse
import os
import os.path as osp
import time

import numpy as np
import pytorch3d
import torch
from PIL import Image
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
)
from tqdm import tqdm
from utils import (
    get_cosine_schedule_with_warmup,
    get_mesh_renderer_soft,
    init_mesh,
    clone_mesh,
    prepare_embeddings,
    prepare_clip_embeddings,
    render_360_views,
    seed_everything,
    save_mesh_as_ply,
    normalize_mesh_longest_axis,
    random_mesh_initiailization,
    random_mesh_initiailization_queue
)
import random
from SDS import SDS, CLIP
import torchvision.transforms as T
from differentiable_object import DifferentiableObject
from pytorch3d.structures import (
    join_meshes_as_batch, 
    join_meshes_as_scene
)

def optimize_mesh_texture(
    sds: SDS,
    clip: CLIP,
    mesh_paths,
    output_dir,
    prompt,
    neg_prompt="",
    device="cpu",
    log_interval=100,
    save_mesh=True,
    args=None,
):
    """
    Optimize the texture map of a mesh to match the prompt.
    """
    # Step 1. Create text embeddings from prompt
    sds_embeddings = prepare_embeddings(sds, prompt, neg_prompt) if args.use_sds else None
    clip_embeddings = prepare_clip_embeddings(clip, prompt, neg_prompt) if args.use_clip else None
    # sds.text_encoder.to("cpu")  # free up GPU memory
    # torch.cuda.empty_cache()

    # Step 3.1 Initialize the renderer
    renderer = get_mesh_renderer_soft(image_size=512, device=device)
    renderer.shader.lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    # Step 2. Load the mesh

    mesh_list = []
    for mesh_path, mesh_init_orientation in zip(mesh_paths, args.mesh_init_orientations):
        mesh, _, _, _ = init_mesh(mesh_path, device=device)
        mesh = normalize_mesh_longest_axis(mesh, rotation_degrees=mesh_init_orientation)
        mesh_list.append(mesh.to(device))


    for i in range(len(mesh_list)):
        mesh_list[i] = clone_mesh(mesh_list[i], shift=args.mesh_configs[i]["transition"], scale=args.mesh_configs[i]["scale"])

    if args.use_rand_init:
        mesh_list = random_mesh_initiailization_queue(args, mesh_list, renderer, clip, clip_embeddings["default"], rand_scale=0.3)

    # Step 2.1 Initialize a randome texture map (optimizable parameter)
    # create a texture field with implicit function
    mesh = join_meshes_as_batch(mesh_list)

    diff_objects = DifferentiableObject(mesh, device)

    mesh = join_meshes_as_scene(diff_objects())


    # For logging purpose, render 360 views of the initial mesh
    if save_mesh:
        for i, m in enumerate(mesh_list):
            save_mesh_as_ply(m.detach(), osp.join(output_dir, f"mesh_{i}.ply"))

        render_360_views(
            mesh.detach(),
            renderer,
            dist=args.dist,
            device=device,
            output_path=osp.join(output_dir, "initial_mesh.gif"),
        )
        save_mesh_as_ply(mesh.detach(), osp.join(output_dir, f"initial_mesh.ply"))


    # generate rendering viewpoints
    Rs = []
    Ts = []
    for elev in np.linspace(0, 15, 5, endpoint=True):
        R, T = look_at_view_transform(dist=args.dist, elev=elev, azim=np.linspace(-180, 180, 18, endpoint=False))
        Rs.append(R)
        Ts.append(T)
    Rs = torch.cat(Rs)
    Ts = torch.cat(Ts)

    query_cameras = FoVPerspectiveCameras(R=Rs, T=Ts, device=mesh.device)

    # generate rendering viewpoints
    R, T = look_at_view_transform(dist=args.dist, elev=0, azim=np.linspace(-180, 180, 12, endpoint=False))
    testing_cameras = FoVPerspectiveCameras(R=R, T=T, device=mesh.device)


    # Step 4. Create optimizer training parameters
    parameters = [
        {'params': [diff_objects.scale], 'lr': 1e-3, "name": "scale"},
        {'params': [diff_objects.rotation], 'lr': 1e-2, "name": "rotation"},
        {'params': [diff_objects.transition], 'lr': 1e-2, "name": "transition"},
    ]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0)
    total_iter = 20000
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(total_iter * 1.5))
    
    # Step 5. Training loop to optimize the mesh positions
    loss_dict = {}
    for i in tqdm(range(total_iter)):
        # Initialize optimizer
        optimizer.zero_grad()

        # Update the textures
        mesh = join_meshes_as_scene(diff_objects())

        # Forward pass
        # Render a randomly sampled camera view to optimize in this iteration
        sampled_cameras = query_cameras[random.choices(range(len(query_cameras)), k=args.views_per_iter)]
        rend = torch.permute(renderer(join_meshes_as_batch([mesh]*args.views_per_iter), cameras=sampled_cameras)[..., :3], (0, 3, 1, 2))
        loss = 0.
        if args.use_sds:
            latents = sds.encode_imgs(rend)
            loss += sds.sds_loss_batch(latents, sds_embeddings["default"], sds_embeddings["uncond"])
        if args.use_clip:
            loss += clip.clip_loss(rend, clip_embeddings["default"], clip_embeddings["uncond"])
            # loss += clip.clip_loss(rend, clip_embeddings["default"])

        print(f"Iter {i}, Loss: {loss.item()}")


        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameter['params'][0] for parameter in parameters], max_norm=1.0)
        optimizer.step()
        scheduler.step()


        # clamping the latents to avoid over saturation
        if i % log_interval == 0 or i == total_iter - 1:
            with torch.no_grad():
                mesh = join_meshes_as_scene(diff_objects())

            # save the loss
            loss_dict[i] = loss.item()

            # save the image
            with torch.no_grad():
                renderer.shader.lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
                img = renderer(mesh, cameras=sampled_cameras[0])[0, ..., :3]
                img = (img.clamp(0, 1) * 255).round().cpu().numpy()
            output_im = Image.fromarray(img.astype("uint8"))
            output_path = os.path.join(
                output_dir,
                f"output_{prompt[0].replace(' ', '_')}_iter_{i}.png",
            )
            output_im.save(output_path)
            mesh_path = os.path.join(
                output_dir,
                f"mesh_iter_{i}.ply",
            )
            save_mesh_as_ply(mesh, mesh_path)

            log_path = os.path.join(
                output_dir,
                f"logs.txt",
            )
            diff_objects.write_log_to_file(log_path)

            # validation
            loss = 0.
            for camera in testing_cameras:
                rend = torch.permute(renderer(mesh, cameras=camera)[..., :3], (0, 3, 1, 2))
                if args.use_sds:
                    latents = sds.encode_imgs(rend)
                    loss -= sds.sds_loss_batch(latents, sds_embeddings["default"], sds_embeddings["uncond"])
                if args.use_clip:
                    loss -= clip.clip_loss(rend, clip_embeddings["default"])
                    # loss += clip.clip_loss(rend, clip_embeddings["default"])

            diff_objects.write_log_to_file(log_path, log=f"Iter {i}, Score: {loss.item()/len(testing_cameras)}")
            print(f"Iter {i}, Score: {loss.item()/len(testing_cameras)}")
            

    if save_mesh:
        render_360_views(
            mesh.detach(),
            renderer,
            dist=args.dist,
            device=device,
            output_path=osp.join(output_dir, f"final_mesh.gif"),
        )
        save_mesh_as_ply(mesh.detach(), osp.join(output_dir, f"final_mesh.ply"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A chair and a table with a toy dinosaur on it")
    parser.add_argument("--seed", type=int, default=42) # 42
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument(
        "--postfix",
        type=str,
        default="_test",
        help="postfix for the output directory to differentiate multiple runs",
    )

    parser.add_argument(
        "-m",
        "--mesh_paths",
        type=list,
        # default=["data/10213_Armed_ Dinner_Chair_v2_iterations-2.obj", "data/10241_Outdoor_table_w_hole_for_umbrella_v1_L3.obj", "data/11678_dinosaur_v1_L3.obj"],
        default=["data/dg/chair_sai.obj", "data/dg/table_sai.obj", "data/dg/toy_dinosaur_sai.obj"],
        help="Path to the input image",
    )
    parser.add_argument(
        "--dist",
        type=int,
        default=4
    )
    parser.add_argument(
        "--views_per_iter",
        type=int,
        default=3    # viewpoints sampled from calculated the loss in a single iteration
    )
    parser.add_argument(
        "--use_sds",
        type=int,
        default=0    # use SDS loss when != 0
    )
    parser.add_argument(
        "--use_clip",
        type=int,
        default=1    # use CLIP loss when != 0
    )
    parser.add_argument(
        "--use_rand_init",
        type=int,
        default=0    # use sampling-base initialization for the initial positions of meshes
    )
    args = parser.parse_args()

    #### rotation for meshes to face forward (edit it for the meshes downloaded from Internet) ####

    # args.mesh_init_orientations = [(90, 180, 0), (90, 0, 0), (90, -90, 0)]
    args.mesh_init_orientations = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]  # meshes from DreamGaussian need not to be rotated

    ##########################

    #### configs from LLM ####

    # manually adjustthe initial positions of meshes (could use the suggestion from LLM instead)

    args.mesh_configs = [{
                    "transition": (-1,0,0), 
                    "rotation": (0,0,0), 
                    "scale": 1  # chair
                },
                {
                    "transition": (0,0,0), 
                    "rotation": (0,0,0), 
                    "scale": 1  # table
                },
                {
                    "transition": (0,1,0), 
                    "rotation": (0,0,0), 
                    "scale": 0.5  # dinosaur
                }]
    
    ##########################

    seed_everything(args.seed)

    # create output directory
    args.output_dir = osp.join(args.output_dir, "mesh")
    output_dir = os.path.join(
        args.output_dir, args.prompt.replace(" ", "_") + args.postfix + ("_sds" if args.use_sds else "") + ("_clip" if args.use_clip else "") + f"_{args.views_per_iter}view"
    )
    os.makedirs(output_dir, exist_ok=True)

    # initialize SDS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sds = SDS(sd_version="2.1", device=device, output_dir=output_dir) if args.use_sds else None
    clip = CLIP(device=device, output_dir=output_dir) if args.use_clip else None

    # optimize the texture map of a mesh
    start_time = time.time()
    assert (
        args.mesh_paths is not None
    ), "mesh_path should be provided for optimizing the texture map for a mesh"
    
    neg_prompt = [""] # ["", "distortion", "blur"]
    
    optimize_mesh_texture(
        sds, clip, mesh_paths=args.mesh_paths, output_dir=output_dir, prompt=args.prompt, neg_prompt=neg_prompt, device=device, args=args
    )
    print(f"Optimization took {time.time() - start_time:.2f} seconds")
