import torch
import torch.nn as nn


def wasserstein_distance(features_real, features_fake):
    """Calculate the Wasserstein distance"""
    w_distance = torch.mean(torch.abs(features_real - features_fake))
    return w_distance


def l2_normalize(tensor, dim=-1, eps=1e-12):
    """L2 normalization is applied to a given tensor"""
    min_val = tensor.min()
    if min_val < 0:
        tensor = tensor - min_val + eps

    norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
    return tensor / (norm + eps)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop_ratio=0.):
        super().__init__()
        out_features = in_features
        hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_ratio)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, patch_size=3, patch_grid=3, attn_drop_ratio=0.):
        super().__init__()
        self.patch_size = patch_size
        self.patch_grid = patch_grid
        self.num_heads = num_heads
        self.attn_drop_ratio = nn.Dropout(attn_drop_ratio)
        self.softmax = nn.Softmax(dim=-1)
        self.hidden_dim = dim // num_heads
        self.proj = nn.Linear(dim, dim, bias=True)
        self.scale = (self.hidden_dim * self.patch_size ** 2) ** -0.5
        self.N = self.patch_grid * self.patch_grid
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape  # 64, 15, 15, 64
        N = self.N  # 25
        P = self.patch_size ** 2  # 9

        # Attention calculation
        x_view = x.view(B, self.patch_grid, self.patch_size, self.patch_grid, self.patch_size, C)  # b, 5, 3, 5, 3, c
        x_fc = x_view.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, N, P, C)  # b, 5, 5, 3, 3, c --> b, 25, 9, c
        qkv = self.qkv(x_fc).reshape(B, N, 3, self.num_heads, P, C // self.num_heads).permute(2, 0, 3, 1, 4, 5)
        qkv = qkv.view(3, B, self.num_heads, N, -1)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, 4, 25, 9*16
        k = k.transpose(-2, -1)
        attn = q @ k  # b, 4, 25, 25
        attn = attn * self.scale
        attn = self.softmax(attn)
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, P, C // self.num_heads, self.num_heads)
        x_attn = x_attn.view(B, N, P, C)
        x_attn = self.proj(x_attn)
        x_back = x_attn.view(B, self.patch_grid, self.patch_grid, self.patch_size, self.patch_size, C)

        x = x_back.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

        return x


# Realize background reconstruction
class Transformer(nn.Module):
    def __init__(self, dim, patch_size=3, patch_grid=3, mlp_ratio=4., num_heads=4,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_drop_ratio=0., drop_ratio=0.):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, patch_size=patch_size, patch_grid=patch_grid,
                              attn_drop_ratio=attn_drop_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_ratio=drop_ratio)

    def forward(self, x):
        B, H, W, C = x.shape
        x1 = x.view(B, H * W, C)
        x_norm1 = self.norm1(x1)
        x_norm1 = x_norm1.view(B, H, W, C)
        x_attn = self.attn(x_norm1)

        # FFN
        x = x_attn.view(B, H * W, C)
        x_norm2 = self.norm2(x)
        x_mlp = x + self.mlp(x_norm2)
        x = x_mlp.view(B, H, W, C)

        return x


class Net(nn.Module):
    def __init__(self, input_channels=3, embedding_dim=96, patch_size=3, patch_grid=3,
                 num_heads=4, mlp_ratio=2., attn_drop_ratio=0., drop_ratio=0.):
        super(Net, self).__init__()
        self.conv_head = nn.Conv2d(input_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.attn_layer = Transformer(embedding_dim, patch_size=patch_size,
                                      patch_grid=patch_grid, mlp_ratio=mlp_ratio, num_heads=num_heads,
                                      attn_drop_ratio=attn_drop_ratio,
                                      drop_ratio=drop_ratio)
        self.conv_middle = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.conv_tail = nn.Conv2d(embedding_dim, input_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_conv = self.conv_head(x)  # (b, c, h, w)
        x_conv_reshaped = x_conv.permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)
        x_attn = self.attn_layer(x_conv_reshaped)
        x_attn_reshaped = x_attn.permute(0, 3, 1, 2).contiguous()  # b, c, h, w
        x_middle = self.conv_middle(x_attn_reshaped)  # b, c, h, w
        x_reconstructed = self.conv_tail(x_middle)

        # Normalize
        x_normalized = l2_normalize(x.view(x.size(0), -1))
        x_reconstructed_normalized = l2_normalize(x_reconstructed.view(x_reconstructed.size(0), -1))

        x_conv_normalized = l2_normalize(x_conv.view(x_conv.size(0), -1))
        x_middle_normalized = l2_normalize(x_middle.view(x_middle.size(0), -1))

        # Calculate the Wasserstein distance
        w_distance1 = wasserstein_distance(x_normalized, x_reconstructed_normalized)
        w_distance2 = wasserstein_distance(x_conv_normalized, x_middle_normalized)

        w = w_distance1 + w_distance2

        return x_reconstructed, w
