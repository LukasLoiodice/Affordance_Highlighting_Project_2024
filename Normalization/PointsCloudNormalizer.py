import copy
from . import Normalizer
from pytorch3d.structures import Pointclouds

class PointsCloudNormalizer:
    def __init__(self, points_clouds):
        self._points_cloud = points_clouds  # original copy of the mesh
        self._original_points = self._points_cloud.points_cloud.points_list()[0]
        self._original_features = self._points_cloud.points_cloud.features_list()[0]

        self.normalizer = Normalizer.get_bounding_sphere_normalizer(self._original_points)

    def __call__(self):
        points_normalized = self.normalizer(self._original_points)

        self._points_cloud.points_cloud = Pointclouds(points=[points_normalized], features=[self._original_features])
        