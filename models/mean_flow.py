"""Teacher-guided Pixel Mean Flow distillation for MiniT2I."""

import jax
import jax.numpy as jnp
from flax import linen as nn

from models.mean_flow_mmjit import (
    get_mean_flow_mmjit_model_cls,
    get_mean_flow_teacher_mmjit_model_cls,
)


def batch_mul(a, b):
    return jax.vmap(lambda a_i, b_i: a_i * b_i)(a, b)


class PixelMeanFlow(nn.Module):
    """Text-conditioned Pixel Mean Flow student plus frozen MiniT2I teacher."""

    model_str: str
    llm_str: str
    model_config: dict
    image_size: int = 512
    image_channel: int = 3
    dtype: jnp.dtype = jnp.float32

    # Noise distribution.
    P_mean: float = 0.8
    P_std: float = 0.8
    noise_scale: float = 2.0

    # CFG sampling/training.
    cfg_beta: float = 1.0
    cfg_max: float = 12.0

    # Mean Flow dynamics.
    data_proportion: float = 0.5
    norm_p: float = 1.0
    norm_eps: float = 0.01
    adapt_weight: bool = True

    def setup(self):
        assert self.image_channel == 3
        model_cfg = dict(self.model_config)
        student_cls = get_mean_flow_mmjit_model_cls(
            self.model_str,
            self.llm_str,
            img_size=self.image_size,
        )
        teacher_cls = get_mean_flow_teacher_mmjit_model_cls(
            self.model_str,
            self.llm_str,
            img_size=self.image_size,
        )
        self.net = student_cls(name="net", **model_cfg)
        self.pt_net = teacher_cls(name="pt_net", **model_cfg)

    def get_visualization(self, list_imgs):
        return jnp.concatenate(list_imgs, axis=1)

    def logit_normal_dist(self, batch_size):
        rnd = jax.random.normal(self.make_rng("gen"), (batch_size,), dtype=self.dtype)
        return nn.sigmoid(rnd * self.P_std + self.P_mean)

    def sample_tr(self, batch_size):
        t = self.logit_normal_dist(batch_size)
        r = self.logit_normal_dist(batch_size)
        data_size = int(batch_size * self.data_proportion)
        fm_mask = jnp.arange(batch_size) < data_size
        r = jnp.where(fm_mask, t, r)
        t, r = jnp.maximum(t, r), jnp.minimum(t, r)
        return t, r, fm_mask

    def sample_cfg_scale(self, batch_size, s_max=None):
        if s_max is None:
            s_max = self.cfg_max
        u = jax.random.uniform(
            self.make_rng("gen"), (batch_size,), minval=0.0, maxval=1.0, dtype=jnp.float32
        )
        if self.cfg_beta == 1.0:
            scale = jnp.exp(u * jnp.log1p(jnp.asarray(s_max, jnp.float32)))
        else:
            beta = jnp.asarray(self.cfg_beta, jnp.float32)
            log_base = (1.0 - beta) * jnp.log1p(jnp.asarray(s_max, jnp.float32))
            log_inner = jnp.log1p(u * jnp.expm1(log_base))
            scale = jnp.exp(log_inner / (1.0 - beta))
        return jnp.asarray(scale, jnp.float32)

    def sample_cfg_interval(self, batch_size, fm_mask=None):
        rng_start, rng_end = jax.random.split(self.make_rng("gen"))
        t_min = jax.random.uniform(
            rng_start, (batch_size,), minval=0.0, maxval=0.5, dtype=self.dtype
        )
        t_max = jax.random.uniform(
            rng_end, (batch_size,), minval=0.5, maxval=1.0, dtype=self.dtype
        )
        if fm_mask is not None:
            t_min = jnp.where(fm_mask, 0.0, t_min)
            t_max = jnp.where(fm_mask, 1.0, t_max)
        return t_min, t_max

    def x_fn(self, x, t, h, omega, t_min, t_max, text_embeddings, attn_masks):
        return self.net(x, t, h, omega, t_min, t_max, text_embeddings, attn_masks)

    def u_fn(self, x, t, *args):
        x_pred = self.x_fn(x, t, *args)
        return batch_mul(x - x_pred, 1 / jnp.clip(t, 0.05, 1.0))

    def sample_one_step(
        self, z_t, i, t_steps, omega, t_min, t_max, text_embeddings, attn_masks
    ):
        t = jnp.take(t_steps, i)
        r = jnp.take(t_steps, i + 1)
        batch_size = z_t.shape[0]
        t = jnp.broadcast_to(t, (batch_size,))
        r = jnp.broadcast_to(r, (batch_size,))
        omega = jnp.broadcast_to(omega, (batch_size,))
        t_min = jnp.broadcast_to(t_min, (batch_size,))
        t_max = jnp.broadcast_to(t_max, (batch_size,))
        u = self.u_fn(z_t, t, t - r, omega, t_min, t_max, text_embeddings, attn_masks)
        return z_t - jnp.einsum("n,n...->n...", t - r, u)

    def __call__(self, x, t, context):
        attn_mask = jnp.ones((x.shape[0], context.shape[1]), dtype=jnp.float32)
        teacher = self.pt_net(x, t, context, attn_mask)
        student = self.net(x, t, t, t, t, t, context, attn_mask)
        return student + teacher

    def teacher_guide_v(self, imgs, t, omega, text_embeddings, attn_masks):
        t = t.reshape(-1)
        imgs_input = jnp.concatenate([imgs, imgs], axis=0)
        t_input = jnp.concatenate([t, t], axis=0)
        embed_input = jnp.concatenate([text_embeddings, text_embeddings], axis=0)
        attn_input = jnp.concatenate([attn_masks, jnp.zeros_like(attn_masks)], axis=0)
        x_pred = self.pt_net(imgs_input, 1 - t_input, embed_input, attn_input)
        v_pred = batch_mul(imgs_input - x_pred, 1 / jnp.clip(t_input, 0.05, 1.0))
        pred_c, pred_u = jnp.split(v_pred, 2)
        guided = batch_mul(pred_c, omega) + batch_mul(pred_u, 1 - omega)
        return guided, pred_c

    def forward(self, imgs, text_embeddings, attn_masks):
        x = imgs.astype(self.dtype)
        batch_size = x.shape[0]

        t, r, fm_mask = self.sample_tr(batch_size)
        t_min, t_max = self.sample_cfg_interval(batch_size, fm_mask)
        omega = self.sample_cfg_scale(batch_size, s_max=self.cfg_max)

        rng = self.make_rng("gen")
        _, rng_e = jax.random.split(rng)
        noise = jax.random.normal(rng_e, x.shape, dtype=self.dtype) * self.noise_scale
        z_t = batch_mul(x, 1 - t) + batch_mul(noise, t)

        def u_fn_wrap(z, t_in, r_in):
            return self.u_fn(
                z, t_in, t_in - r_in, omega, t_min, t_max, text_embeddings, attn_masks
            )

        omega_t = jnp.where((t_min <= t) & (t <= t_max), omega, jnp.ones_like(omega))
        v_g, v_c = self.teacher_guide_v(z_t, t, omega_t, text_embeddings, attn_masks)
        v_g = jax.lax.stop_gradient(v_g)
        v_c = jax.lax.stop_gradient(v_c)

        dtdt = jnp.ones_like(t)
        dtdr = jnp.zeros_like(t)
        u, du_dt = jax.jvp(u_fn_wrap, (z_t, t, r), (v_g, dtdt, dtdr))
        target = u + batch_mul(jax.lax.stop_gradient(du_dt), t - r)

        loss_u = jnp.sum((target - v_g) ** 2, axis=(1, 2, 3))
        if self.adapt_weight:
            adapt = (loss_u + self.norm_eps) ** self.norm_p
            loss_u = loss_u / jax.lax.stop_gradient(adapt)
        loss_u_raw = jnp.mean((target - v_g) ** 2)
        loss = loss_u.mean()

        pred_x_g = z_t - batch_mul(v_g, t)
        pred_x_c = z_t - batch_mul(v_c, t)
        images = self.get_visualization([z_t, imgs, pred_x_g, pred_x_c])
        metrics = {
            "loss": loss,
            "loss_u": jnp.mean(loss_u),
            "loss_u_raw": loss_u_raw,
        }
        return loss, metrics, images


def generate(
    variable,
    inference_cfg_scale,
    text_embeddings,
    attn_masks,
    model: PixelMeanFlow,
    rng,
    n_sample,
    config,
):
    """Few-step PMF sampler. Timesteps run from noise t=1 to data t=0."""
    num_steps = int(config.eval.num_steps)
    t_min = float(config.eval.t_min)
    t_max = float(config.eval.t_max)
    image_size = int(config.dataset.image_size)
    image_channels = int(config.dataset.image_channels)

    rng, rng_xt = jax.random.split(rng)
    z_t = jax.random.normal(
        rng_xt, (n_sample, image_size, image_size, image_channels), dtype=model.dtype
    ) * model.noise_scale
    t_steps = jnp.linspace(1.0, 0.0, num_steps + 1)

    def step_fn(i, x_i):
        return model.apply(
            variable,
            x_i,
            i=i,
            t_steps=t_steps,
            omega=jnp.asarray(inference_cfg_scale),
            t_min=jnp.asarray(t_min),
            t_max=jnp.asarray(t_max),
            text_embeddings=text_embeddings,
            attn_masks=attn_masks,
            method=model.sample_one_step,
        )

    return jax.lax.fori_loop(0, num_steps, step_fn, z_t)
