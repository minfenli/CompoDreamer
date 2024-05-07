import torch
import pytorch3d
from pytorch3d.structures import Meshes
import torch.nn.functional as F

def quaternion_to_rotation_matrix(quaternion):
    """Convert a quaternion to a differentiable rotation matrix."""
    # Ensure quaternion is normalized
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    w, x, y, z = quaternion.unbind(-1)

    # Compute rotation matrix components
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    # Construct rotation matrix
    rot_matrix = torch.stack([
        1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy),
        2 * (xy + wz),     1 - 2 * (xx + zz),     2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)
    ], dim=-1).reshape(-1, 3, 3)  # Reshape to 3x3 matrix

    return rot_matrix

def quaternion_to_y_rotation_matrix(quaternion):
    """Convert a quaternion to a differentiable rotation matrix that only rotates about the y-axis."""
    # Ensure quaternion is normalized
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    w, _, y, _ = quaternion.unbind(-1)  # Ignore x and z components for y-axis rotation

    # Compute rotation matrix components based on y-axis rotation
    cos_theta = 1 - 2 * y * y
    sin_theta = 2 * w * y

    # Construct rotation matrix only with y-axis rotation
    rot_matrix = torch.stack([
        cos_theta, torch.zeros_like(cos_theta), sin_theta,
        torch.zeros_like(cos_theta), torch.ones_like(cos_theta), torch.zeros_like(cos_theta),
        -sin_theta, torch.zeros_like(cos_theta), cos_theta
    ], dim=-1).reshape(-1, 3, 3)  # Reshape to 3x3 matrix

    return rot_matrix

class  DifferentiableObject(torch.nn.Module):
    def __init__(self, meshes: Meshes, device="cpu"):
        super().__init__()

        self.meshes = meshes
        self.n_meshes = len(meshes)
        self.scale = torch.autograd.Variable(torch.ones(self.n_meshes, device=device), requires_grad=True)
        self.scale_limit = 0.5
        # Initialize the quaternion for rotation (w, x, y, z) with w=1 and x=y=z=0 for no rotation
        self.rotation = torch.autograd.Variable(torch.tensor(self.n_meshes*[[1.0, 0.0, 0.0, 0.0]], device=device), requires_grad=True)
        self.rotate_along_y_axis = True
        # Initialize the transition vector (x, y, z) with 0s
        # self.transition = torch.autograd.Variable(torch.zeros(3, device=device), requires_grad=True)
        self.transition = torch.autograd.Variable(torch.tensor(self.n_meshes*[[0.0, 0.0, 0.0]], device=device), requires_grad=True)

    def set_scale(self, scale):
        assert scale.shape == self.scale.shape
        self.scale = torch.autograd.Variable(scale.to(self.scale.device), requires_grad=True)
    
    def set_rotation(self, rotation):
        assert rotation.shape == self.rotation.shape
        self.rotation = torch.autograd.Variable(rotation.to(self.rotation.device), requires_grad=True)

    def set_transition(self, transition):
        assert transition.shape == self.transition.shape
        self.transition = torch.autograd.Variable(transition.to(self.transition.device), requires_grad=True)

    def forward(self):
        verts = []

        for i, vert in enumerate(self.meshes.verts_list()):
            
            vert = vert.detach().unsqueeze(0).clone() # vert position will be change even if .detach(), so add .clone()

            # Scale
            vert *= torch.clip(self.scale[i:i+1, None, None], min=1-self.scale_limit, max=1+self.scale_limit)
            # Rotate
            vert = self.rotate_vertices(vert, self.rotation[i:i+1], self.rotate_along_y_axis)
            # Translate
            vert += self.transition[i:i+1, None, :]

            verts.append(vert.squeeze(0))

        print(self.info())

        return Meshes(verts=[vert for vert in verts], faces=self.meshes.faces_list(), textures=self.meshes.textures)

    def rotate_vertices(self, vertices, quaternion, rotate_along_y_axis):
        """
        Rotate vertices by a quaternion.
        vertices: Tensor of shape (B, N, 3), N vertices with (x, y, z) coordinates.
        quaternion: Tensor of shape (B, 4), represents (w, x, y, z) of the quaternion.
        Returns rotated vertices of shape (N, 3).
        """
        if rotate_along_y_axis:
            rot_matrix = quaternion_to_y_rotation_matrix(quaternion)
        else:
            rot_matrix = quaternion_to_rotation_matrix(quaternion)
       
        return vertices @ torch.transpose(rot_matrix, -1, -2)  # Rotate vertices

    def info(self):
        info = ""
        for i in range(self.n_meshes):
            scale = torch.clip(self.scale[i], min=1-self.scale_limit, max=1+self.scale_limit)
            info += f"Mesh {i}: scale = {scale.item()}, rotation = {self.rotation[i].tolist()}, transition = {self.transition[i].tolist()}\n"
        return info

    def write_log_to_file(self, filename, log=None):
        with open(filename, 'a+') as file:
            file.write("======================\n")
            if log is None:
                file.write(self.info())
            else:
                file.write(log+"\n")


class DifferentiableObjectV2(torch.nn.Module):
    def __init__(self, meshes: Meshes, device="cpu"):
        super().__init__()

        self.meshes = meshes
        self.n_meshes = len(meshes)
        self.scale = torch.autograd.Variable(torch.zeros(self.n_meshes, device=device), requires_grad=True)
        # Initialize the quaternion for rotation (w, x, y, z) with w=1 and x=y=z=0 for no rotation
        self.rotation = torch.autograd.Variable(torch.tensor(self.n_meshes*[[1.0, 0.0, 0.0, 0.0]], device=device), requires_grad=True)
        # Initialize the transition vector (x, y, z) with 0s
        # self.transition = torch.autograd.Variable(torch.zeros(3, device=device), requires_grad=True)
        self.transition = torch.autograd.Variable(torch.tensor(self.n_meshes*[[0.0, 0.0, 0.0]], device=device), requires_grad=True)

    def forward(self):
        verts = []

        for i, vert in enumerate(self.meshes.verts_list()):

            vert = vert.detach().unsqueeze(0)

            # Scale
            vert *= 1 + (F.sigmoid(self.scale[i:i+1, None, None]) - 0.5)*0.4 # (0.8, 1.2)
            # Rotate
            vert = self.rotate_vertices(vert, self.rotation[i:i+1])
            # Translate
            # vert += self.transition[i:i+1, None, :]
            vert += 2 * (F.sigmoid(self.transition[i:i+1, None, :]) - 0.5) # (-1, 1)
            
            verts.append(vert.squeeze(0))

        print(self.info())

        return Meshes(verts=[vert for vert in verts], faces=self.meshes.faces_list(), textures=self.meshes.textures)

    def rotate_vertices(self, vertices, quaternion):
        """
        Rotate vertices by a quaternion.
        vertices: Tensor of shape (B, N, 3), N vertices with (x, y, z) coordinates.
        quaternion: Tensor of shape (B, 4), represents (w, x, y, z) of the quaternion.
        Returns rotated vertices of shape (N, 3).
        """
        rot_matrix = quaternion_to_rotation_matrix(quaternion)
       
        return vertices @ torch.transpose(rot_matrix, -1, -2)  # Rotate vertices

    def info(self):
        info = ""
        for i in range(self.n_meshes):
            scale = 1 + (F.sigmoid(self.scale[i]) - 0.5) * 0.4
            info += f"Mesh {i}: scale = {scale.item()}, rotation = {self.rotation[i].tolist()}, transition = {self.transition[i].tolist()}\n"
        return info