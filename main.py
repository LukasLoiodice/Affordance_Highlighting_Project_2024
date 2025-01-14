import clip
import copy
import json
import clip.model
import kaolin as kal
import kaolin.ops.mesh
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torchvision
import pickle

from itertools import permutations, product
from Normalization import MeshNormalizer, PointsCloudNormalizer
from mesh import Mesh
from pointsCloud import PointsCloud
from pathlib import Path
from render import PointsCloudRenderer
from tqdm import tqdm
from torch.autograd import grad
from torchvision import transforms
from utils import device, color_mesh, color_points_cloud, intersection_over_union, down_sample

class NeuralHighlighter(nn.Module):
    def __init__(self, depth, width, output_dim=2, input_dim=3):
        super(NeuralHighlighter, self).__init__()

        # Adapt input dim to the width of the model
        moduleList = nn.ModuleList([
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.LayerNorm([width])
        ])

        # Append all the remaining layers
        for _ in range(depth):
            moduleList.extend([
                nn.Linear(width, width),
                nn.ReLU(),
                nn.LayerNorm([width])
            ])

        # Append last layer with softmax
        moduleList.extend([
            nn.Linear(width, output_dim),
            nn.Softmax(dim=1)
        ])

        self.mlp = moduleList
        print(self.mlp)
    
    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x

def get_clip_model(clipmodel):
    return clip.load(clipmodel, device=device)
        

def clip_loss(clip_text: torch.Tensor, rendered_images: torch.Tensor, clip_model: clip.model.CLIP, n_augs: int, augment_transform: transforms.Compose, clip_transform: transforms.Compose) -> torch.Tensor:
    if n_augs == 0:
        transformed_images = clip_transform(rendered_images)
        clip_images: torch.Tensor = clip_model.encode_image(transformed_images)
        return -(torch.cosine_similarity(clip_text, torch.mean(clip_images)))
    
    loss = .0
    for _ in range(n_augs):
        augmented_images = augment_transform(rendered_images)
        clip_images: torch.Tensor = clip_model.encode_image(augmented_images)
        loss -= torch.cosine_similarity(clip_text, torch.mean(clip_images, dim=0, keepdim=True))

    return loss

    
def save_renders(dir, i, rendered_images, name=None):
    if name is not None:
        torchvision.utils.save_image(rendered_images, os.path.join(dir, name))
    else:
        torchvision.utils.save_image(rendered_images, os.path.join(dir, 'renders/iter_{}.jpg'.format(i)))


# Constrain most sources of randomness
# (some torch backwards functions within CLIP are non-determinstic)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


render_res = 224
learning_rate = 0.0001
n_iter = 3000
res = 224
affordanceNet_path = 'AffordanceNet/full_shape_train_data.pkl' 
selected_index = 9447 # first mug index 9447
n_views = 5
n_augs = 2
output_dir = './output/'
clip_model_name = 'ViT-L/14'
depth = 5
width = 256
sample_points = 2048
render_point_radius = 0.03
iou_treshold = 0.4
augment_extention = False
downsample_extention = False
downsample_strength = 0.08

Path(os.path.join(output_dir, 'renders')).mkdir(parents=True, exist_ok=True)

# Get data from AffordanceNet
try:
    file = open(affordanceNet_path, "rb")
except FileNotFoundError:
    raise Exception("You need to specify/download the affordanceNet full shape data")
else:
    with file:
        data = pickle.load(file)
        selected_shape = data[selected_index]
        points = torch.tensor(selected_shape['full_shape']["coordinate"]).cuda()
        ground_truth = torch.tensor(selected_shape["full_shape"]["label"]["grasp"]).cuda()

if downsample_extention:
    points = down_sample(points, downsample_strength)

# Initialize variables
background = torch.tensor((1., 1., 1.)).to(device)

render = PointsCloudRenderer(background, dim=(render_res, render_res), radius=render_point_radius)
points_cloud = PointsCloud(points, sample_points)
PointsCloudNormalizer.PointsCloudNormalizer(points_cloud)()

log_dir = output_dir


# MLP Settings
mlp = NeuralHighlighter(depth, width).to(device)
optim = torch.optim.Adam(mlp.parameters(), learning_rate)

# list of possible colors
rgb_to_color = {(204/255, 1., 0.): "highlighter", (180/255, 180/255, 180/255): "gray"}
color_to_rgb = {"highlighter": [204/255, 1., 0.], "gray": [180/255, 180/255, 180/255]}
full_colors = [[204/255, 1., 0.], [180/255, 180/255, 180/255]]
colors = torch.tensor(full_colors).to(device)

# Transformations for images augmentations
clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
clip_transform = transforms.Compose([
    transforms.Resize((res, res)),
    clip_normalizer
])
if augment_extention:
    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(res, scale=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
        clip_normalizer
    ])
else:
    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(res, scale=(1, 1)),
        transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
        clip_normalizer
    ])


# --- Prompt ---
# encode prompt with CLIP
clip_model, _ = get_clip_model(clip_model_name)
prompt = f"A 3D point cloud representation of a gray mug with the precise area intented for gripping shown in green." # A 3D point cloud representation of a gray mug with the precise area intented for gripping shown in green.
print("prompt : ", prompt)
with torch.no_grad():
    prompt_tokenize = clip.tokenize(prompt).to(device)
    clip_text = clip_model.encode_text(prompt_tokenize)
    clip_text = clip_text / clip_text.norm(dim=1, keepdim=True)

points = copy.deepcopy(points_cloud.points_cloud.points_list()[0])

losses = []

# Optimization loop
for i in tqdm(range(n_iter)):
    optim.zero_grad()

    # predict highlight probabilities
    pred_class = mlp(points)

    # color and render mesh
    sample_points_cloud = points_cloud
    sample_points_cloud.points_cloud = color_points_cloud(pred_class, points, colors)

    rendered_images = render.render_views(sample_points_cloud.points_cloud,
                                               num_views=n_views,
                                               std=1)

    # Calculate CLIP Loss
    loss = clip_loss(clip_text, rendered_images, clip_model, n_augs, augment_transform, clip_transform)
    loss.backward(retain_graph=True)

    optim.step()

    # update variables + record loss
    with torch.no_grad():
        losses.append(loss.item())

    # report results
    if i % 100 == 0:
        print("Last 100 CLIP score: {}".format(np.mean(losses[-100:])))
        save_renders(log_dir, i, rendered_images)
        with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
            f.write(f"For iteration {i}... Prompt: {prompt}, Last 100 avg CLIP score: {np.mean(losses[-100:])}, CLIP score {losses[-1]}\n")

torch.save(pred_class.detach(), f"{output_dir}{selected_index}.pth")

# Compute intersection over union
if not downsample_extention:
    iou = intersection_over_union(pred_class, ground_truth, iou_treshold)
    print("Intersection over Union = ", iou.item())

# save results
sample_points_cloud.save(f"{output_dir}{selected_index}.ply")