# Copyright © 2023 Apple Inc.

import time
from typing import Tuple

import mlx.core as mx

from .model_io import (
    load_unet,
    load_text_encoder,
    load_autoencoder,
    load_diffusion_config,
    load_tokenizer,
    load_controlnet,
    _DEFAULT_MODEL,
)
from .sampler import SimpleEulerSampler


def _repeat(x, n, axis):
    # Make the expanded shape
    s = x.shape
    s.insert(axis + 1, n)

    # Expand
    x = mx.broadcast_to(mx.expand_dims(x, axis + 1), s)

    # Make the flattened shape
    s.pop(axis + 1)
    s[axis] *= n

    return x.reshape(s)


class ControlNetStableDiffusionPipeline:
    def __init__(self, model: str = _DEFAULT_MODEL, float16: bool = True):
        self.dtype = mx.float16 if float16 else mx.float32
        self.diffusion_config = load_diffusion_config(model)
        self.unet, unet_config = load_unet(model, float16) # returns the unet config
        self.controlnet = load_controlnet(unet_config = unet_config)
       
        self.text_encoder = load_text_encoder(model, float16)
        self.autoencoder = load_autoencoder(model, float16)
        self.sampler = SimpleEulerSampler(self.diffusion_config)
        self.tokenizer = load_tokenizer(model)

    def generate_latents(
        self,
        text: str,
        n_images: int = 1,
        num_steps: int = 50,
        cfg_weight: float = 7.5,
        negative_text: str = "",
        latent_size: Tuple[int] = (64, 64),
        seed=None,
        controlnet_conditioning_scale: int = 1,
    ):
        # Set the PRNG state
        seed = seed or int(time.time())
        mx.random.seed(seed)
        # image = mx
        image = mx.random.uniform(-1,1,(1,512,512,3)) # transform image to [-1,1]
        # Tokenize the text
        tokens = [self.tokenizer.tokenize(text)]
        if cfg_weight > 1:
            tokens += [self.tokenizer.tokenize(negative_text)]
        lengths = [len(t) for t in tokens]
        N = max(lengths)
        tokens = [t + [0] * (N - len(t)) for t in tokens]
        tokens = mx.array(tokens)

        # Compute the features
        conditioning = self.text_encoder(tokens)

        # Repeat the conditioning for each of the generated images
        if n_images > 1:
            conditioning = _repeat(conditioning, n_images, axis=0)

        # Create the latent variables
        x_T = self.sampler.sample_prior(
            (n_images, *latent_size, self.autoencoder.latent_channels), dtype=self.dtype
        )

        # Perform the denoising loop
        controlnet_keep = []
        for i in range(num_steps):
            keeps = [
                1.0 - float(i / num_steps < s or (i + 1) / num_steps > e)
                for s, e in zip([0], [1])
            ]
            controlnet_keep.append(keeps[0])

        print(controlnet_keep)

        x_t = x_T
        for t, t_prev in self.sampler.timesteps(num_steps, dtype=self.dtype):
            x_t_unet = mx.concatenate([x_t] * 2, axis=0) if cfg_weight > 1 else x_t
            t_unet = mx.broadcast_to(t, [len(x_t_unet)])

            controlnet_prompt_embeds = conditioning

            controlnet_cond_scale = controlnet_conditioning_scale
            cond_scale = controlnet_cond_scale * controlnet_keep[i]
            down_block_res_samples, mid_block_res_samples = self.controlnet(
                    x_t_unet,
                    t_unet,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale
                )

            eps_pred = self.unet(
                x_t_unet, 
                t_unet,
                encoder_x=conditioning,
                cn_down_residuals = down_block_res_samples,
                cn_mid_residuals = mid_block_res_samples)

            if cfg_weight > 1:
                eps_text, eps_neg = eps_pred.split(2)
                eps_pred = eps_neg + cfg_weight * (eps_text - eps_neg)

            x_t_prev = self.sampler.step(eps_pred, x_t, t, t_prev)
            x_t = x_t_prev
            yield x_t

    def decode(self, x_t):
        x = self.autoencoder.decode(x_t / self.autoencoder.scaling_factor)
        x = mx.minimum(1, mx.maximum(0, x / 2 + 0.5))
        return x
