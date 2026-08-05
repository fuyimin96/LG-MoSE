from torch import nn
from torch.nn.functional import interpolate
import torch
import numpy as np
from model_ss2d.vmamba import SS2D
from .softmoe_spectral import SoftMoE_Spectral


class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels):
        super(Bottleneck, self).__init__()

        self.layers = nn.Sequential(*[
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, 3, 1, 1),
        ])

    def forward(self, x):
        x = self.layers(x)
        return x


class Down(nn.Module):

    def __init__(self,
                 down_rate,):
        super(Down, self).__init__()

        self.layers = nn.Sequential(*[
            nn.ReLU(),
            nn.AvgPool2d(down_rate)
        ])

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):

    def __init__(self,
                 shape,
                 need_relu):
        super(Up, self).__init__()

        self.need_relu = need_relu
        self.rows, self.cols, bands = shape

        self.conv = nn.Conv2d(bands, bands, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = interpolate(x, size=(self.rows, self.cols), mode='bilinear')
        if self.need_relu:
            x = self.relu(x)
        return x


class Encoder(nn.Module):

    def __init__(self,
                 shape,
                 expert,
                 slot,
                 down_rate):
        super(Encoder, self).__init__()

        rows, cols, bands = shape

        self.bottleneck = Bottleneck(bands, 16)
        self.softmoe_spectral = SoftMoE_Spectral(dim=bands, rows=rows, cols=cols, num_experts=expert,num_slots=slot,use_layernorm=True)
        self.down_block = Down(down_rate)
        self.up_block = Up(shape, need_relu=False)

    def forward(self, x, y):
        output = self.bottleneck(x)
        output = output + self.softmoe_spectral(x, y)
        output = self.down_block(output)
        sm = self.up_block(output)
        return output, sm
    

class Exchange(nn.Module):
    def __init__(self, dim, top_percent=0.1):
        super().__init__()
        self.dim = dim               # bands
        self.top_percent = top_percent

    def forward(self, x):
        B, C, H, W = x.shape

        if C < 2:
            return x

        block_size = 3
        H_blocks = H // block_size
        W_blocks = W // block_size
        if H_blocks == 0 or W_blocks == 0:
            return x

        H_valid = H_blocks * block_size
        W_valid = W_blocks * block_size
        x_valid = x[:, :, :H_valid, :W_valid]   


        x_blocks = x_valid.reshape(B, C, H_blocks, block_size, W_blocks, block_size)
        x_blocks = x_blocks.permute(0, 2, 4, 1, 3, 5)  # (B, H_blocks, W_blocks, C, 3, 3)

        # (B, H_blocks, W_blocks, C)
        block_means = x_blocks.mean(dim=(-2, -1))

        # (B, C)
        global_mean = x.mean(dim=[2, 3])

        with torch.no_grad():
            global_avg = global_mean.mean(dim=1, keepdim=True)        # (B, 1)
            global_std = global_mean.std(dim=1, keepdim=True, unbiased=False) + 1e-8
            
            block_avg = block_means.mean(dim=-1, keepdim=True)        # (B, H_blocks, W_blocks, 1)
            block_std = block_means.std(dim=-1, keepdim=True, unbiased=False) + 1e-8

            g_centered = global_mean.view(B, 1, 1, C) - global_avg.view(B, 1, 1, 1)
            b_centered = block_means - block_avg

            cov = (b_centered * g_centered).sum(dim=-1) / (C - 1)

            # r = cov / (std_x * std_y)
            pearson_r = cov / (block_std.squeeze(-1) * global_std.view(B, 1, 1) + 1e-8)  # (B, H_blocks, W_blocks)

            # top_percent
            num_blocks = H_blocks * W_blocks
            k = max(1, int(num_blocks * self.top_percent))

            out_blocks = x_blocks.clone()
            for b in range(B):
                r_map = pearson_r[b]                     # (H_blocks, W_blocks)
                flat_r = r_map.view(-1)
                _, top_idx = torch.topk(flat_r, k)
                top_h = top_idx // W_blocks
                top_w = top_idx % W_blocks

                perm = torch.randperm(k, device=x.device)
                shuffled_h = top_h[perm]
                shuffled_w = top_w[perm]

                selected = out_blocks[b, top_h, top_w].clone()
                out_blocks[b, top_h, top_w] = selected[perm]

        # (B, H_blocks, W_blocks, C, 3, 3) -> (B, C, H_blocks, 3, W_blocks, 3)
        out_blocks = out_blocks.permute(0, 3, 1, 4, 2, 5).contiguous()
        out_valid = out_blocks.reshape(B, C, H_valid, W_valid)

        if H != H_valid or W != W_valid:
            out = x.clone()
            out[:, :, :H_valid, :W_valid] = out_valid
        else:
            out = out_valid
        return out


class Decoder(nn.Module):

    def __init__(self,
                 shape):
        super(Decoder, self).__init__()

        rows, cols, bands = shape

        self.up_block = Up(shape, need_relu=True)
        self.conv = nn.Conv2d(bands, bands, 1)
        self.conv2 = nn.Conv2d(bands, bands, 1)
        self.exchange = Exchange(dim=bands)
        self.ss2d = SS2D(
                        d_model=bands, 
                        d_state=1, 
                        ssm_ratio=2.0,
                        dt_rank="auto",
                        act_layer=nn.SiLU,
                        # ==========================
                        d_conv=3,
                        conv_bias=True,
                        # ==========================
                        dropout=0.0,
                        initialize="v0",
                        # ==========================
                        forward_type="v2",
                        channel_first=True,
                        )

    def forward(self, encoder_output, sm):
        output = self.up_block(encoder_output)
        output = self.ss2d(self.exchange(output)) + self.conv2(output)
        output = output + sm
        output = self.conv(output)
        return output


class Text_layer(nn.Module):
    def __init__(self,
                 in_channels = 512):
        super(Text_layer, self).__init__()
        self.layer0 = nn.Linear(in_features = in_channels, out_features = in_channels)
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(in_features = in_channels, out_features = in_channels)
        
    def forward(self, x):
        x = self.layer0(x)
        x = self.relu(x)
        x = self.layer1(x)
        return x


class LG_MoSE(nn.Module):

    def __init__(self,
                 **kwargs,):
        super(LG_MoSE, self).__init__()

        self.name = 'LG_MoSE'

        self.num_layers = kwargs['num_layers']
        rows, cols, bands = kwargs['shape']
        self.e = 3
        self.s = 2

        self.encoders = nn.ModuleList([
            Encoder(shape=(rows, cols, bands), expert = self.e, slot = self.s, down_rate=2 ** _l)
            for _l in range(self.num_layers)
        ])

        self.decoders = nn.ModuleList([
            Decoder(shape=(rows, cols, bands))
            for _l in range(self.num_layers)
        ])
        
        self.text_encoders = Text_layer(512)
        self.conv = nn.Conv2d(bands,bands,kernel_size=3,stride=1, padding=1)

    def forward(self, x, text_feature):
        x = x.permute(2, 0, 1).unsqueeze(0)

        decoding_list = []
        decoding_text_list = []

        # Encoding
        encoding_sum = x
        for _l in range(self.num_layers):
            text_output = self.text_encoders(text_feature) + text_feature
            output, sm = self.encoders[_l](encoding_sum,text_output)
            decoding_list.append(output)
            encoding_sum = sm + encoding_sum

        # Decoding
        decoding_sum = torch.zeros_like(x)
        for _cd in range(self.num_layers - 1, -1, -1):
            encoder_output = decoding_list[_cd]
            decoder_output = self.decoders[_cd](encoder_output, decoding_sum)
            decoding_sum = decoder_output + decoding_sum
            decoding_list[_cd] = decoder_output
            

        to_orig_shape = map(
            lambda _x: _x.squeeze(0).permute(1, 2, 0),
            decoding_list
        )

        return tuple(to_orig_shape)

