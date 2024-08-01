import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, LayerNorm, Linear, GELU, Dropout
from transformers import LlamaModel, LlamaConfig, AutoTokenizer
from utils import *
from torchsummary import summary

# Logger setup
import logging
logger = logging.getLogger(__name__)


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        B, N, C = xyz.shape
        if C > 3:
            data = xyz
            xyz = data[:,:,:3]
            rgb = data[:, :, 3:]
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group)  # B G 3

        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        neighborhood_xyz = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood_xyz = neighborhood_xyz.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        if C > 3:
            neighborhood_rgb = rgb.view(batch_size * num_points, -1)[idx, :]
            neighborhood_rgb = neighborhood_rgb.view(batch_size, self.num_group, self.group_size, -1).contiguous()

        # normalize xyz 
        neighborhood_xyz = neighborhood_xyz - center.unsqueeze(2)
        if C > 3:
            neighborhood = torch.cat((neighborhood_xyz, neighborhood_rgb), dim=-1)
        else:
            neighborhood = neighborhood_xyz
        return neighborhood, center

class Encoder(nn.Module):
    def __init__(self, encoder_channel, point_input_dims=3):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.point_input_dims = point_input_dims
        self.first_conv = nn.Sequential(
            nn.Conv1d(self.point_input_dims, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)
    

# Configuration for PointLLM
class PointLLMConfig(LlamaConfig):
    model_type = "pointllm"

# MLP module used within Transformer
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super(Mlp, self).__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)

# Transformer Block containing Attention and MLP
class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.):
        super(Block, self).__init__()
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=GELU, drop=drop)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.drop_path = Dropout(drop_path)

    def forward(self, x):
        # print("x in block:", x.shape)
        x_norm1 = self.norm1(x)
        # print("x_norm1 in block:", x_norm1.shape)
        x = x + self.drop_path(self.attn(x_norm1))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x

# Point Transformer Model
class PointTransformer(nn.Module):
    def __init__(self, config, use_max_pool=True):
        super(PointTransformer, self).__init__()

        self.trans_dim = config["trans_dim"]
        self.depth = config["depth"]
        self.drop_path_rate = config["drop_path_rate"]
        self.cls_dim = config["cls_dim"]
        self.num_heads = config["num_heads"]
        self.use_max_pool = config["use_max_pool"]

        self.group_size = config["group_size"]
        self.num_group = config["num_group"]
        self.point_dims = config["point_dims"]

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = config["encoder_dims"]
        self.encoder = Encoder(encoder_channel=self.encoder_dims, point_input_dims=self.point_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        # Position embedding for center
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

    def forward(self, pts):
        # knn farthest sampling -> knn+
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # add pos embedding
        pos = self.pos_embed(center)
        # print("Pos Embedding:",pos.shape)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        # print("x:",x.shape)
        pos = torch.cat((cls_pos, pos), dim=1)
        # print("pos:",pos.shape)
        x = self.blocks(x, pos)
        x = self.norm(x) # * B, G + 1(cls token)(513), C(384)
        # print(x.shape)
        if not self.use_max_pool:
            return x
        
        # final input
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1).unsqueeze(1) # * concat the cls token and max pool the features of different tokens, make it B, 1, C
        return concat_f

model_config = {
  "NAME": PointTransformer,
  "trans_dim": 384, 
  "depth": 12, 
  "drop_path_rate": 0.1, 
  "cls_dim": 40, 
  "num_heads": 6,
  "group_size": 32, 
  "num_group": 512,
  "encoder_dims": 256,
  "point_dims": 6,
  "projection_hidden_layer": 2,
  "projection_hidden_dim": [1024, 2048],
  "use_max_pool": False
}
npoints = 8192

def testGroup_Encoder():
    xyz = torch.randn(32, npoints, 3)
    group_module = Group(num_group=512, group_size=32)
    neighborhood, centers = group_module(xyz)
    print("Neighborhood shape:", neighborhood.shape)  # Should be [B, num_group, group_size, C]
    print("Centers shape:", centers.shape)  # Should be [B, num_group, C]

    # Test Encoder
    neighborhood = neighborhood.to("mps")
    encoder = Encoder(encoder_channel=512, point_input_dims=3).to("mps")
    out = encoder(neighborhood)
    print("Output shape:", out.shape)


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def test_PointTransformer():
    xyz = torch.randn(32, npoints, 6).to("mps")
    ptv1 = PointTransformer(model_config).to("mps")
    
    out = ptv1(xyz)
    print(out.shape)
    print("Model Architecture:")
    pp = get_n_params(ptv1)
    print(pp)



# test_PointTransformer()
