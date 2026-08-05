import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn
import torch
import pdb


# Generate overlapping HSI blocks using a sliding window.
class BlockGeneration(nn.Module):
    def __init__(self, block_size=15, stride=5):
        super(BlockGeneration, self).__init__()
        self.block_size = block_size
        self.stride = stride

    def pixel_padding(self, hsis, kernel_sizes, strides, rates=(1, 1)):
        assert len(hsis.size()) == 4
        batch_size, band, height, width = hsis.size()
        out_height = (height + strides[0] - 1) // strides[0]
        out_width = (width + strides[1] - 1) // strides[1]
        effective_k_row = (kernel_sizes[0] - 1) * rates[0] + 1
        effective_k_col = (kernel_sizes[1] - 1) * rates[1] + 1
        padding_rows = max(0, (out_height - 1) * strides[0] + effective_k_row - height)
        padding_cols = max(0, (out_width - 1) * strides[1] + effective_k_col - width)

        # Calculate padding for top, bottom, left, and right.
        padding_top = int(padding_rows // 2)
        padding_bottom = padding_rows - padding_top
        padding_left = int(padding_cols // 2)
        padding_right = padding_cols - padding_left

        paddings = (padding_left, padding_right, padding_top, padding_bottom)
        hsis = F.pad(hsis, paddings, mode='replicate')

        return hsis, paddings

    def extract_image_blocks(self, hsis, kernel_sizes, strides):
        assert len(hsis.size()) == 4
        images, paddings = self.pixel_padding(hsis, kernel_sizes, strides)

        unfold = torch.nn.Unfold(kernel_size=kernel_sizes, padding=0, stride=strides)
        blocks = unfold(images)

        return blocks, paddings

    def forward(self, x):
        _, band, height, width = x.size()

        # Blocks order: from left to right, and from top to down.
        blocks, paddings = self.extract_image_blocks(
            x, kernel_sizes=[self.block_size, self.block_size],
            strides=[self.stride, self.stride]
        )

        blocks = blocks.squeeze(0).permute(1, 0)
        blocks = blocks.view(blocks.size(0), band, self.block_size, self.block_size)

        return blocks, blocks, paddings


# Restore the original HSI from overlapping blocks.
class BlockRestore(nn.Module):
    def __init__(self, block_size=15, stride=5):
        super(BlockRestore, self).__init__()
        self.block_size = block_size
        self.stride = stride

    def forward(self, x, paddings, height, width):
        num_blocks = x.size(0)
        blocks_reshaped = x.view(num_blocks, -1).permute(1, 0).unsqueeze(0)

        # Judge padding value
        block_size_height = (height + 2 * paddings[2] - self.block_size) / self.stride + 1
        block_size_width = (width + 2 * paddings[0] - self.block_size) / self.stride + 1

        if block_size_height * block_size_width != num_blocks:
            pad = [paddings[3], paddings[1]]
        else:
            pad = [paddings[2], paddings[0]]
        # assert block_size_height * block_size_width == num_blocks, \
        #     f'Error: block_size_height * block_size_width = {block_size_height * block_size_width}, but does not equal {num_blocks}.'

        try:
            original_image = F.fold(blocks_reshaped, (height, width), (self.block_size, self.block_size), padding=pad,
                                    stride=self.stride)

        except Exception as e:
            print('Failed during fold operation:', e)
            pdb.set_trace()

        # Use the average operation to deal with the overlapped areas among blocks.
        overlapping_mask = torch.ones_like(original_image)
        mask_unfold = F.unfold(overlapping_mask, (self.block_size, self.block_size), padding=pad, stride=self.stride)
        fold_mask = F.fold(mask_unfold, (height, width), (self.block_size, self.block_size), padding=pad, stride=self.stride)

        out = original_image / fold_mask

        return out


class Dataset(data.Dataset):
    def __init__(self, data, block_size=15, stride=5):
        super(Dataset, self).__init__()
        self.data_processer = BlockGeneration(block_size=block_size, stride=stride)
        self.gt_blocks, self.input_blocks, self.padding = self.data_processer(data)

    def __getitem__(self, index):
        block_gt = self.gt_blocks[index]
        block_input = self.input_blocks[index]

        return {'block_gt': block_gt, 'block_input': block_input}

    def __len__(self):
        return self.gt_blocks.size(0)

    def get_all_blocks(self):
        return self.gt_blocks
