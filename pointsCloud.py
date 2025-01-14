import kaolin as kal
import torch
import utils
from utils import device
import copy
import numpy as np
import PIL
import open3d as o3d
from pytorch3d.io import IO
from pytorch3d.structures import Pointclouds
import numpy as np
from utils import device

class PointsCloud():
    def __init__(self, data: str | torch.Tensor, number_of_points: int = 5000):
        if isinstance(data, torch.Tensor):
            colors = torch.fill(data, 0.5).to(device) # Each points are gray for the first iteration

            # Create pytorch3d pointcloud class
            self.points_cloud = Pointclouds(points=[data], features=[colors])
        else:
            if ".obj" in data:
                # Load mesh and trsnform into point clouds
                mesh = o3d.io.read_triangle_mesh(data)
                o3d_points_cloud = mesh.sample_points_uniformly(number_of_points=number_of_points)

                # Pass point cloud into gpu
                points = torch.tensor(np.array(o3d_points_cloud.points), dtype=torch.float32).to(device)
                colors = torch.fill(points, 0.5).to(device) # Each points are gray for the first iteration

                # Create pytorch3d pointcloud class
                self.points_cloud = Pointclouds(points=[points], features=[colors])
            elif ".ply" in data:
                self.points_cloud = IO().load_pointcloud(data).to(device)
            else:
                raise ValueError(f"{data} extension not implemented in pointsCloud reader.")

    def save(self, path: str):
        IO().save_pointcloud(self.points_cloud, path)