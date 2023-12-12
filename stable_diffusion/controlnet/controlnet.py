
import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from .unet import TimestepEmbedding, TransformerBlock, Transformer2D, ResnetBlock2D, UNetBlock2D 
from typing import Optional, Tuple, Union
from .config import ControlNetConfig,UNetConfig

def zero_module(module):
    #  TODO: check why this is used.  
    # for p in module.parameters():
        # nn.init.zeros_(p)
        # p = torch.zeros_like(p)
    return module

class ControlNetConditioningEmbedding(nn.Module):
    """
    Direct replication of the controlnet code, adjusted for MLX"
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = []

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )


    def __call__(self, conditioning):

        embedding = self.conv_in(conditioning)
        embedding = nn.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = nn.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding

class ControlNetModel(nn.Module):

    def __init__(
        self,
        config : ControlNetConfig, 
        unet_config: UNetConfig# unet config 
    ): 
        super().__init__()
        """
        TODO: Migrate to a singular config. 
        """

        self.conv_in = nn.Conv2d(
            unet_config.in_channels,
            unet_config.block_out_channels[0],
            unet_config.conv_in_kernel,
            padding=(unet_config.conv_in_kernel - 1) // 2,
        )
        
        # time 
        self.timesteps = nn.SinusoidalPositionalEncoding(
            unet_config.block_out_channels[0],
            max_freq=1,
            min_freq=math.exp(
                -math.log(10000) + 2 * math.log(10000) / unet_config.block_out_channels[0]
            ),
            scale=1.0,
            cos_first=True,
            full_turns=False,
        ) # time_proj in diffusers notation

        # time embedding
        self.time_embedding = TimestepEmbedding(
            unet_config.block_out_channels[0],
            unet_config.block_out_channels[0] * 4,
        )

        # control net conditioning embedding
        # the unet_config should be correct as it's pulled from huggingface 
        # still TODO: Check values.

        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels= config.block_out_channels[0], 
            block_out_channels=config.conditioning_embedding_out_channels,
            conditioning_channels= config.conditioning_channels,
        )

        self.down_blocks = []
        self.controlnet_down_blocks = []

        output_channel = unet_config.block_out_channels[0]

        controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        block_channels = [unet_config.block_out_channels[0]] + list(
            unet_config.block_out_channels
        )
        for i, (in_channels, out_channels) in enumerate(
                zip(block_channels, block_channels[1:])
            ):

            down_block = UNetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=unet_config.block_out_channels[0] * 4,
                num_layers=unet_config.layers_per_block[i],
                transformer_layers_per_block=unet_config.transformer_layers_per_block[i],
                num_attention_heads=unet_config.num_attention_heads,
                cross_attention_dim=unet_config.cross_attention_dim[i],
                resnet_groups=unet_config.norm_num_groups,
                add_downsample=(i < len(unet_config.block_out_channels) - 1),
                add_upsample=False,
                add_cross_attention=(i < len(unet_config.block_out_channels) - 1),
            )

            self.down_blocks.append(down_block)

            for _ in range(config.layers_per_block[i]):
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

            if not (i < len(config.block_out_channels ) -1):
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)


        # mid
        mid_block_channel = unet_config.block_out_channels[-1]

        controlnet_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block

        self.mid_blocks = [
            ResnetBlock2D(
                in_channels=unet_config.block_out_channels[-1],
                out_channels=unet_config.block_out_channels[-1],
                temb_channels=unet_config.block_out_channels[0] * 4,
                groups=unet_config.norm_num_groups,
            ),
            Transformer2D(
                in_channels=unet_config.block_out_channels[-1],
                model_dims=unet_config.block_out_channels[-1],
                num_heads=unet_config.num_attention_heads,
                num_layers=unet_config.transformer_layers_per_block[-1],
                encoder_dims=unet_config.cross_attention_dim[-1],
            ),
            ResnetBlock2D(
                in_channels=unet_config.block_out_channels[-1],
                out_channels=unet_config.block_out_channels[-1],
                temb_channels=unet_config.block_out_channels[0] * 4,
                groups=unet_config.norm_num_groups,
            ),
        ]

    def __call__(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        controlnet_cond,
        conditioning_scale,
        attention_mask = None,
    ):
    #  -> Union[ControlNetOutput, Tuple[Tuple[torch.FloatTensor, ...], torch.FloatTensor]]:
        

        # prepare attention_mask\
        # TODO: re-add attention mask and encoder attn_mask.
        # 

        # 1. time
        temb = self.timesteps(timestep)
        # t_emb = t_emb.to(dtype=sample.dtype)

        temb = self.time_embedding(temb)
        aug_emb = None

        # 2. pre-process
        sample = self.conv_in(sample)

        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        sample = sample + controlnet_cond
        # 3. down
        down_block_res_samples = [sample]
        for downsample_block in self.down_blocks:
            # if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(
                sample,
                temb = temb,
                encoder_x = encoder_hidden_states,
                attn_mask = attention_mask,
            )

            down_block_res_samples.extend(res_samples)


        # 4. mid
        sample = self.mid_blocks[0](sample, temb)
        sample = self.mid_blocks[1](sample, encoder_hidden_states, attention_mask, None)
        sample = self.mid_blocks[2](sample, temb)
        
        # 5. Control net blocks

        controlnet_down_block_res_samples = []

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples.append(down_block_res_sample)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        # guess mode switched off
        down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
        mid_block_res_sample = mid_block_res_sample * conditioning_scale

        return down_block_res_samples, mid_block_res_sample

       