import torch
import kaolin as kal
import kaolin.ops.mesh
import clip
import numpy as np
from torchvision import transforms
from pathlib import Path
from collections import Counter
from Normalization import MeshNormalizer
from pytorch3d.structures import Pointclouds
import open3d as o3d
import trimesh
import copy
import random

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def get_camera_from_view2(elev, azim, r=3.0):
    x = r * torch.cos(elev) * torch.cos(azim)
    y = r * torch.sin(elev)
    z = r * torch.cos(elev) * torch.sin(azim)
    # print(elev,azim,x,y,z)

    pos = torch.tensor([x, y, z]).unsqueeze(0)
    look_at = -pos
    direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

    camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
    return camera_proj

def get_texture_map_from_color(mesh, color, H=224, W=224):
    num_faces = mesh.faces.shape[0]
    texture_map = torch.zeros(1, H, W, 3).to(device)
    texture_map[:, :, :] = color
    return texture_map.permute(0, 3, 1, 2)


def get_face_attributes_from_color(mesh, color):
    num_faces = mesh.faces.shape[0]
    face_attributes = torch.zeros(1, num_faces, 3, 3).to(device)
    face_attributes[:, :, :] = color
    return face_attributes

# mesh coloring helpers
def color_mesh(pred_class, sampled_mesh, colors):
    pred_rgb = segment2rgb(pred_class, colors)
    sampled_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
        pred_rgb.unsqueeze(0),
        sampled_mesh.faces)
    MeshNormalizer(sampled_mesh)()

def color_points_cloud(pred_class, points, colors):
    pred_rgb = segment2rgb(pred_class, colors)
    return Pointclouds(points=[points], features=[pred_rgb])

def segment2rgb(pred_class, colors):
    pred_rgb = torch.zeros(pred_class.shape[0], 3).to(device)
    for class_idx, color in enumerate(colors):
        pred_rgb += torch.matmul(pred_class[:,class_idx].unsqueeze(1), color.unsqueeze(0))
        
    return pred_rgb

# function that add random background (random uniform color or stripes)
def random_background(image: torch.Tensor, resolution: int, background_color: torch.Tensor = torch.tensor([1., 1., 1.]).to(device), threshold: float = 0.9) -> torch.Tensor:
    image = image[0]  # Remove the batch dimension
    image_np = image.permute(1, 2, 0).detach().cpu().numpy()

    # Convert to tensor and move to device
    resized_image = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).to(device)

    if random.random() > 0.5:
        # Uniform background
        bg_color = torch.tensor(np.random.rand(3), dtype=torch.float32).view(3, 1, 1).to(device)
        background = bg_color.expand(3, resolution, resolution)
    else:
        # Striped background
        background = torch.zeros(3, resolution, resolution, dtype=torch.float32, device=device)
        stripe_width = resolution // 10
        for i in range(0, resolution, stripe_width):
            color = torch.tensor(np.random.rand(3), dtype=torch.float32).view(3, 1, 1).to(device)
            stripe_end = min(i + stripe_width, resolution)
            background[:, :, i:stripe_end] = color.expand(3, resolution, stripe_end - i)

    # Pixels near the background_color are considered as the background
    mask = (torch.abs(resized_image - background_color.view(3, 1, 1)) < (1.0 - threshold)).all(dim=0).float().to(device)

    # Blend the resized image with the generated background
    blended_image = background * mask + resized_image * (1 - mask)

    return blended_image.unsqueeze(0)

def simplify_mesh(mesh):
    mesh_copy= copy.deepcopy(mesh)
    trimesh_obj = trimesh.Trimesh(vertices=mesh.vertices.cpu().numpy(), faces=mesh.faces.cpu().numpy())
    simplified_trimesh = trimesh_obj.simplify_quadric_decimation(face_count=5000, aggression=1)
    mesh_copy.vertices = torch.tensor(simplified_trimesh.vertices).to(device)
    mesh_copy.faces = torch.tensor(simplified_trimesh.faces).to(device)
    return mesh_copy

def intersection_over_union(pred_class: torch.Tensor, ground_truth: torch.Tensor, treshold: float) -> float:
    prediction_binary = (pred_class.detach().T[0] >= treshold).int()
    affordance_binary = (ground_truth.squeeze(1) >= treshold).int()

    intersection = torch.sum((prediction_binary & affordance_binary).float())
    union = torch.sum((prediction_binary | affordance_binary).float())

    return intersection/union

def down_sample(point_list, strength):
    # Pass the point list into open3D for downsampling
    o3d_points_cloud = o3d.geometry.PointCloud()
    o3d_points_cloud.points = o3d.utility.Vector3dVector(point_list.cpu().numpy())
    
    # Perform the downsampling
    o3d_point_cloud_down_sampled = o3d_points_cloud.voxel_down_sample(strength)

    print(f"Downsampled from {len(o3d_points_cloud.points)} to {len(o3d_point_cloud_down_sampled.points)}")

    return torch.tensor(np.array(o3d_point_cloud_down_sampled.points), dtype=torch.float32).cuda()