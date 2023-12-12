
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
    #     nn.init.zeros_(p)
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

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = nn.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = nn.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding

class ControlNetModel(nn.Module):
    """
    A ControlNet model.

    Args:
        in_channels (`int`, defaults to 4):
            The number of channels in the input sample.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, defaults to 0):
            The frequency shift to apply to the time embedding.
        down_block_types (`tuple[str]`, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        only_cross_attention (`Union[bool, Tuple[bool]]`, defaults to `False`):
        block_out_channels (`tuple[int]`, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, defaults to 2):
            The number of layers per block.
        downsample_padding (`int`, defaults to 1):
            The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, defaults to 1):
            The scale factor to use for the mid block.
        act_fn (`str`, defaults to "silu"):
            The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the normalization. If None, normalization and activation layers is skipped
            in post-processing.
        norm_eps (`float`, defaults to 1e-5):
            The epsilon to use for the normalization.
        cross_attention_dim (`int`, defaults to 1280):
            The dimension of the cross attention features.
        transformer_layers_per_block (`int` or `Tuple[int]`, *optional*, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unet_2d_blocks.CrossAttnDownBlock2D`], [`~models.unet_2d_blocks.CrossAttnUpBlock2D`],
            [`~models.unet_2d_blocks.UNetMidBlock2DCrossAttn`].
        encoder_hid_dim (`int`, *optional*, defaults to None):
            If `encoder_hid_dim_type` is defined, `encoder_hidden_states` will be projected from `encoder_hid_dim`
            dimension to `cross_attention_dim`.
        encoder_hid_dim_type (`str`, *optional*, defaults to `None`):
            If given, the `encoder_hidden_states` and potentially other embeddings are down-projected to text
            embeddings of dimension `cross_attention` according to `encoder_hid_dim_type`.
        attention_head_dim (`Union[int, Tuple[int]]`, defaults to 8):
            The dimension of the attention heads.
        use_linear_projection (`bool`, defaults to `False`):
        class_embed_type (`str`, *optional*, defaults to `None`):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from None,
            `"timestep"`, `"identity"`, `"projection"`, or `"simple_projection"`.
        addition_embed_type (`str`, *optional*, defaults to `None`):
            Configures an optional embedding which will be summed with the time embeddings. Choose from `None` or
            "text". "text" will use the `TextTimeEmbedding` layer.
        num_class_embeds (`int`, *optional*, defaults to 0):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
            class conditioning with `class_embed_type` equal to `None`.
        upcast_attention (`bool`, defaults to `False`):
        resnet_time_scale_shift (`str`, defaults to `"default"`):
            Time scale shift config for ResNet blocks (see `ResnetBlock2D`). Choose from `default` or `scale_shift`.
        projection_class_embeddings_input_dim (`int`, *optional*, defaults to `None`):
            The dimension of the `class_labels` input when `class_embed_type="projection"`. Required when
            `class_embed_type="projection"`.
        controlnet_conditioning_channel_order (`str`, defaults to `"rgb"`):
            The channel order of conditional image. Will convert to `rgb` if it's `bgr`.
        conditioning_embedding_out_channels (`tuple[int]`, *optional*, defaults to `(16, 32, 96, 256)`):
            The tuple of output channel for each block in the `conditioning_embedding` layer.
        global_pool_conditions (`bool`, defaults to `False`):
            TODO(Patrick) - unused parameter.
        addition_embed_type_num_heads (`int`, defaults to 64):
            The number of heads to use for the `TextTimeEmbedding` layer.
    """


    def __init__(
        self,
        config : ControlNetConfig, 
        unet_config: UNetConfig# unet config 
    ): # TODO adjust arguments 

        super().__init__()
        """
        TODO: Migrate to a singular config. 
        """
        # Status: Done. 
        # Small TODOs are left and __init__ args is left

        # Things I have skipped
        # class embedding 
        # encoder_hid_proj
        # addition_embed_type 

        # num_attention_heads = config.num_attention_heads 
        # hb: we have config.block_out_channels instead

        # input
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

        # TODO: take a look the attention part
        # if isinstance(only_cross_attention, bool):
        #     only_cross_attention = [only_cross_attention] * len(down_block_types)

        # if isinstance(attention_head_dim, int):
        #     attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # if isinstance(num_attention_heads, int):
        #     num_attention_heads = (num_attention_heads,) * len(down_block_types)

        # down

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
                num_attention_heads=unet_config.num_attention_heads[i],
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
                num_heads=unet_config.num_attention_heads[-1],
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
        
    @classmethod
    def load_unet_weights(
        cls,
        unet,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        conditioning_channels: int = 3,
    ):
        ## TODO: modify this to load from the new unet2D class wihtout conditioning  
        r"""
        Instantiate a [`ControlNetModel`] from [`UNet2DConditionModel`].

        Parameters:
            unet (`UNet2DConditionModel`):
                The UNet model weights to copy to the [`ControlNetModel`]. All configuration options are also copied
                where applicable.
        """
        
        controlnet = cls() # call cls like CN diffusers 

        controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
        controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
        controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

        controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
        controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        return controlnet

    def forward(
        self,
        # sample: torch.FloatTensor,
        # timestep: Union[torch.Tensor, float, int],
        # encoder_hidden_states: torch.Tensor,
        # controlnet_cond: torch.FloatTensor,
        # conditioning_scale: float = 1.0,
        # class_labels: Optional[torch.Tensor] = None,
        # timestep_cond: Optional[torch.Tensor] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        # cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # guess_mode: bool = False,
        # return_dict: bool = True,
    ):
    #  -> Union[ControlNetOutput, Tuple[Tuple[torch.FloatTensor, ...], torch.FloatTensor]]:
        """
        The [`ControlNetModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor.
            timestep (`Union[torch.Tensor, float, int]`):
                The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states.
            controlnet_cond (`torch.FloatTensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            conditioning_scale (`float`, defaults to `1.0`):
                The scale factor for ControlNet outputs.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond (`torch.Tensor`, *optional*, defaults to `None`):
                Additional conditional embeddings for timestep. If provided, the embeddings will be summed with the
                timestep_embedding passed through the `self.time_embedding` layer to obtain the final timestep
                embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            added_cond_kwargs (`dict`):
                Additional conditions for the Stable Diffusion XL UNet.
            cross_attention_kwargs (`dict[str]`, *optional*, defaults to `None`):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor`.
            guess_mode (`bool`, defaults to `False`):
                In this mode, the ControlNet encoder tries its best to recognize the input content of the input even if
                you remove all prompts. A `guidance_scale` between 3.0 and 5.0 is recommended.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~models.controlnet.ControlNetOutput`] instead of a plain tuple.

        Returns:
            [`~models.controlnet.ControlNetOutput`] **or** `tuple`:
                If `return_dict` is `True`, a [`~models.controlnet.ControlNetOutput`] is returned, otherwise a tuple is
                returned where the first element is the sample tensor.
        """


        # prepare attention_mask\
        # TODO: re-add attention mask and encoder attn_mask.
        # 

        # 1. time
        timesteps = timestep # see if this is correct input
        temb = self.timesteps(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        # 2. pre-process
        sample = self.conv_in(sample)

        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        sample = sample + controlnet_cond

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            # if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

            down_block_res_samples += res_samples

        # 4. mid

        sample = self.mid_blocks[0](sample, temb)
        sample = self.mid_blocks[1](sample, encoder_x, attn_mask, encoder_attn_mask)
        sample = self.mid_blocks[2](sample, temb)
        
        # 5. Control net blocks

        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        # guess mode switched off
        down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
        mid_block_res_sample = mid_block_res_sample * conditioning_scale

        return down_block_res_samples, mid_block_res_sample

       