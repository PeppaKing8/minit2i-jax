"""Simple DDPM / flow-matching trainer model: forward loss + sampler dispatch."""

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.scipy.special import erfinv

from models.mmjit import get_mmjit_model_cls
from utils.pjit_util import enforce_ddp, ddp_rand_func

ModuleDef = Any


# ---------------------------------------------------------------------------
# t-schedule (the distribution of the diffusion timestep sampled per training
# step). Used in `SimDDPM.forward`.
# ---------------------------------------------------------------------------

def _uniform_to_gaussian(u):
    return jnp.sqrt(2) * erfinv(2 * u - 1)


class DiffusionSchedule(ABC):
    @abstractmethod
    def to_t(self, uniformity):
        ...

    def sample(self, key, B):
        return self.to_t(jax.random.uniform(key, (B,))).astype(jnp.float32)


class LinearSchedule(DiffusionSchedule):
    def to_t(self, uniformity):
        return uniformity


class LognormSchedule(DiffusionSchedule):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def to_t(self, uniformity):
        return jax.nn.sigmoid(self.mu + self.sigma * _uniform_to_gaussian(uniformity))

    def sample(self, key, B):
        t = jax.random.normal(key, (B,)) * self.sigma + self.mu
        return jax.nn.sigmoid(t).astype(jnp.float32)


def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)


class SimDDPM(nn.Module):
    """Simple DDPM."""

    model_str: str
    llm_str: str
    model_config: dict
    dtype: int = jnp.float32
    image_size: int = 256
    image_channel: int = 3

    # sampling configs
    n_T: int = 100  # inference steps
    average_loss: bool = True
    label_drop_rate: float = 0.1
    cfg_channels: int = 3
    prediction: str = 'v'

    # t-schedule (important in high dim!)
    t_sample_schedule: str = "uniform" # uniform, lognorm
    t_lognorm_mu: float = 0.0
    t_lognorm_sigma: float = 1.0
    noise_scale: float = 2.0
    
    # inference schedule
    sampler: str = "euler" # euler-4e-2, euler, heun, sde
    sample_clip_x0: bool = False
    # x-pred -> v-pred denominator floor (clip (1 - t) from below).
    # Smaller => sharper / more chaotic near t=1; larger => smoother.
    a_min: float = 0.05
    
    # sampling
    seed: int = 0
    
    def setup(self):
        model_str = self.model_str
        llm_str = self.llm_str

        net_fn = get_mmjit_model_cls(model_str, llm_str, img_size=self.image_size)
        assert net_fn is not None, f"Cannot infer net function from ({model_str}, {llm_str}, {self.image_size})."
        self.net = net_fn(name="net", **self.model_config)

        assert self.cfg_channels in [3, 4]

    def get_visualization(self, list_imgs):
        vis = jnp.concatenate(list_imgs, axis=1)
        return vis


    @property
    def v_pred_net(self):
        def pred(x, t, y, m, clip=False):
            # NB: the underlying MMJiT does not consume `t` (no adaLN);
            # `t` is still used here for the x-pred -> v-pred conversion.
            if self.prediction == 'v':
                return self.net(x, y, m)
            elif self.prediction == 'x':
                x0 = self.net(x, y, m)
                if clip:
                    x0 = jnp.clip(x0, -1.0, 1.0)
                return batch_mul((x0 - x), 1 / jnp.clip(1 - t, a_min=self.a_min))
            else:
                raise NotImplementedError(f'Unknown prediction type {self.prediction}')
        return pred
    
    def cfg_wrapped_net(self, inference_cfg_scale):
        def net_cfg(x, t, y, m):
            # Array form (matches text-jit's default cfg_interval=(0,1)); the
            # equivalent scalar form compiles to a different HLO under XLA/TPU
            # bf16 mixed-precision and drifts by ~0.5pp over 100 NFE.
            cfg = jnp.full((x.shape[0], 1, 1, 1), inference_cfg_scale, dtype=jnp.float32)

            B = x.shape[0]
            combined = jnp.concatenate([x, x], axis=0)
            y = jnp.concatenate([y, y], axis=0)
            # Unconditional branch: zero out the attention mask so the cross-attn
            # over text tokens degenerates to no-condition.
            m_null = jnp.zeros_like(m)
            m = jnp.concatenate([m, m_null], axis=0)
            t = jnp.concatenate([t, t], axis=0)
            out = self(combined, t, y, m, clip=self.sample_clip_x0)

            if self.cfg_channels == 3:
                eps, rest = out[:, :, :, :3], out[:, :, :, 3:]
                cond_eps, uncond_eps = jnp.split(eps, 2, axis=0)
                half_eps = uncond_eps + (cond_eps - uncond_eps) * cfg
                half_rest = rest[:B]
                return jnp.concatenate([half_eps, half_rest], axis=-1)
            else:
                cond, uncond = jnp.split(out, 2, axis=0)
                return uncond + (cond - uncond) * cfg
        return net_cfg
        
    def sample_one_step(self, *args, **kwargs):
        if 'euler' in self.sampler:
            x_next = self.sample_one_step_euler(*args, **kwargs)
        elif 'heun' in self.sampler:
            x_next = self.sample_one_step_heun(*args, **kwargs)
        elif 'sde' in self.sampler:
            x_next = self.sample_one_step_sde(*args, **kwargs)
        else:
            raise NotImplementedError(f"Unknown sampler {self.sampler}")

        return x_next

    def sample_one_step_euler(self, x_i, rng, i, timesteps, y, m, inference_cfg_scale):
        """
        FM ODE sampling
        """
        x_cur = x_i
        t_cur = timesteps[i].repeat(x_cur.shape[0])
        t_next = timesteps[i + 1].repeat(x_cur.shape[0])
        # ViT net API with cfg
        net_fn = self.cfg_wrapped_net(inference_cfg_scale=inference_cfg_scale)
        v_pred = net_fn(x = x_cur, t = t_cur, y = y, m = m)
        x_next = x_cur + batch_mul(t_next - t_cur, v_pred)
        return x_next
    
    def sample_one_step_heun(self, x_i, rng, i, timesteps, y, m, inference_cfg_scale):
        x_cur = x_i

        t_cur = timesteps[i].repeat(x_cur.shape[0])
        t_next = timesteps[i + 1].repeat(x_cur.shape[0])

        t_hat = t_cur
        x_hat = x_cur  # x_hat is always x_cur when gamma=0

        # ViT net API with cfg
        net_fn = self.cfg_wrapped_net(inference_cfg_scale=inference_cfg_scale)
        
        # Euler step.
        u_pred = net_fn(x = x_i, t = t_hat, y = y, m = m)
        d_cur = u_pred
        x_next = x_hat + batch_mul(u_pred, t_next - t_hat)

        # Apply 2nd order correction
        u_pred = net_fn(x = x_next, t = t_next, y = y, m = m)
        d_prime = u_pred
        x_next_ = x_hat + batch_mul(0.5 * d_cur + 0.5 * d_prime, t_next - t_hat)

        x_next = jnp.where(i < self.n_T - 1, x_next_, x_next)

        return x_next
    
    def sample_one_step_sde(self, x_i, rng, i, timesteps, y, m, inference_cfg_scale):
        x_cur = x_i
        t_cur = timesteps[i].repeat(x_cur.shape[0])
        t_next = timesteps[i + 1].repeat(x_cur.shape[0])
        
        t_cur_official = 1 - t_cur
        t_next_official = 1 - t_next

        net_fn = self.cfg_wrapped_net(inference_cfg_scale=inference_cfg_scale)
        is_last_step = (i == self.n_T -1)

        dt = t_next_official - t_cur_official

        diffusion = 2 * t_cur_official
        eps_i = jax.random.normal(rng, x_cur.shape, dtype=self.dtype)
        deps = batch_mul(eps_i, jnp.sqrt(jnp.abs(dt)))
        
        # compute drift
        v_cur = - net_fn(x = x_cur, t = t_cur, y = y, m = m) # the negative sign is due to official repo is negated velocity
        
        # get_score_from_velocity
        alpha_t, d_alpha_t = 1 - t_cur_official, t_cur_official*0 - 1
        sigma_t, d_sigma_t = t_cur_official, t_cur_official*0 + 1
        mean = x_cur
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = batch_mul((batch_mul(reverse_alpha_ratio, v_cur) - mean), 1 / var)
        s_cur = score
        
        
        d_cur = v_cur - 0.5 * batch_mul(diffusion, s_cur)

        x_next = x_cur + batch_mul(d_cur, dt) + batch_mul(jnp.sqrt(diffusion), deps) * (1 - is_last_step.astype(jnp.float32)) # no noise on last step

        return x_next

    def __call__(self, x, t, y, m, clip=False):
        return self.v_pred_net(x, t, y, m, clip=clip)

    def forward(self, imgs, text_embeddings, attn_masks):
        imgs = imgs.astype(self.dtype)
        x = imgs
        B = imgs.shape[0]
        H = W = self.image_size
        C = self.image_channel
        
        # Whole-caption CFG drop
        if self.label_drop_rate > 0:
            full_drop_mask = jnp.zeros_like(attn_masks)
            full_drop_mask = enforce_ddp(full_drop_mask)
            random_vars = ddp_rand_func("uniform", "ddp")(self.make_rng("drop"), (B,))
            attn_masks = jnp.where(
                random_vars[:, None] < self.label_drop_rate,
                full_drop_mask,
                attn_masks,
            )

        # -----------------------------------------------------------------
        if self.t_sample_schedule == "uniform":
            d = LinearSchedule()
        elif self.t_sample_schedule == "lognorm":
            d = LognormSchedule(mu=self.t_lognorm_mu, sigma=self.t_lognorm_sigma)
        else:
            raise NotImplementedError(f"Unknown t_sample_schedule {self.t_sample_schedule}")

        t_batch = d.sample(self.make_rng("gen"), B)
        
        noise_batch = ddp_rand_func("normal", "ddp")(
            self.make_rng("gen"), x.shape, dtype=self.dtype
        ) * self.noise_scale
        assert x.shape == (B, H, W, C), f"Expected image shape {(B, H, W, C)}, got {x.shape}"

        x_t = batch_mul(x, t_batch) + batch_mul(noise_batch, 1 - t_batch)
        v_pred = self(x_t, t_batch, text_embeddings, attn_masks) # use old t, mimicking the original image size
        
        # compute L2 loss on velocity prediction
        if self.prediction == 'v':
            target = x - noise_batch
        elif self.prediction == 'x':
            target = (x - x_t) / jnp.clip(1 - t_batch.reshape(-1, 1, 1, 1), a_min=self.a_min)
        residual = v_pred - target
        loss = residual ** 2

        if self.average_loss:
            loss = jnp.mean(loss, axis=(1, 2, 3))
        else: raise AttributeError("we recommend average loss")

        loss_monitor = loss.mean()
        loss = loss.mean(axis=0) # mean over batch

        dict_losses = {"loss": loss, "loss_monitor": loss_monitor}

        # convert the velocity predictor to x-predictor
        pred_x = x_t + batch_mul(v_pred, 1 - t_batch)
        pred_x_target = x_t + batch_mul(target, 1 - t_batch)
        images = self.get_visualization([x_t, imgs, pred_x, pred_x_target])

        return loss, dict_losses, images
    
# move this out from model for JAX compilation
def generate(variable, inference_cfg_scale, text_embedding, attention_masks, model: SimDDPM, rng, n_sample, config):
    """
    Generate samples from the model

    variable: {"params": params, "batch_stats": batch_stats}
    ---
    return shape: (n_sample, 32, 32, 3)
    """
    # prepare schedule (0 for noise, 1 for data)
    num_steps = model.n_T
    if model.sampler == "euler-4e-2":
        t_steps = jnp.linspace(0.0, 1 - 4e-2, num_steps, dtype=model.dtype)
        t_steps = jnp.concatenate([t_steps, jnp.ones((1,), dtype=model.dtype)], axis=0)
    elif model.sampler in ("euler", "heun", "sde"):
        t_steps = jnp.linspace(0.0, 1.0, num_steps + 1, dtype=model.dtype)
    else:
        raise NotImplementedError(f"Unknown sampler {model.sampler!r}")

    # Per-sample rng keys ensure each sample gets unique noise under pjit.
    sample_shape = (model.image_size, model.image_size, model.image_channel)
    rng_keys = jax.random.split(rng, n_sample + 1)
    rng = rng_keys[0]  # for subsequent use
    sample_keys = rng_keys[1:]  # one key per sample
    
    # Generate noise for each sample with its own key
    latents = jax.vmap(
        lambda key: jax.random.normal(key, sample_shape, dtype=model.dtype) * model.noise_scale
    )(sample_keys)  # (n_sample, H, W, C)

    x_i = latents

    labels = text_embedding

    def step_fn(i, inputs):
        x_i, rng = inputs
        rng_z = jax.random.fold_in(rng, i)
        x_i = model.apply(
            variable,
            x_i, rng_z, i, t_steps, labels, attention_masks, inference_cfg_scale,
            method=model.sample_one_step,
        )
        outputs = (x_i, rng)
        return outputs

    outputs = jax.lax.fori_loop(0, num_steps, step_fn, (x_i, rng))
    images = outputs[0]
    return images
