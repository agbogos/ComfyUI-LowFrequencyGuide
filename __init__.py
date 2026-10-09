"""
ComfyUI-LowFrequencyGuide

Iterate on composition at a cheap draft resolution, then render the *same* image
natively at full resolution.

Nodes:
  Multiscale Seeded Noise      - final-resolution noise plus a pooled draft version
                                 that shares its low frequencies.
  Low-Frequency Guide (ILVR)   - model patch that pins the low-frequency band of
                                 generation to a reference latent, then releases it.

See README.md for the full write-up.
"""

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Multiscale Seeded Noise
# ---------------------------------------------------------------------------
#
# The noise is always drawn at the FINAL latent resolution from the seed. The
# draft gets that same tensor, f x f average-pooled and multiplied by f (the mean
# of f^2 iid N(0,1) samples has std 1/f, so this restores unit variance). The
# draft therefore starts from exactly the low-frequency part of the final run's
# initial noise.

class _MultiscaleNoise:
    def __init__(self, seed, factor):
        self.seed = seed
        self.factor = factor

    def generate_noise(self, input_latent):
        latent = input_latent["samples"]
        if getattr(latent, "is_nested", False):
            raise ValueError("MultiscaleSeededNoise does not support nested latents.")

        f = self.factor
        shape = list(latent.shape)          # (B, C, H, W) or (B, C, T, H, W)
        lead, h, w = shape[:-2], shape[-2], shape[-1]

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        noise = torch.randn(lead + [h * f, w * f], generator=gen,
                            dtype=torch.float32, device="cpu")
        if f > 1:
            noise = noise.reshape(*lead, h, f, w, f).mean(dim=(-3, -1)) * f
        return noise.to(latent.dtype)


class MultiscaleSeededNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                       "control_after_generate": True}),
                "factor": ("INT", {"default": 3, "min": 1, "max": 8,
                                   "tooltip": "Final latent size / draft latent size. Must be exact: "
                                              "1280x720 -> 3840x2160 is 3."}),
            }
        }

    RETURN_TYPES = ("NOISE", "NOISE")
    RETURN_NAMES = ("noise_full", "noise_draft")
    FUNCTION = "get_noise"
    CATEGORY = "sampling/custom_sampling/noise"
    DESCRIPTION = ("Seeded noise at the final resolution, plus a pooled draft-resolution "
                   "version that shares its low frequencies.")

    def get_noise(self, noise_seed, factor):
        return (_MultiscaleNoise(noise_seed, 1), _MultiscaleNoise(noise_seed, factor))


# ---------------------------------------------------------------------------
# Low-Frequency Guide (ILVR-style, on the x0 prediction)
# ---------------------------------------------------------------------------
#
# At every step the model predicts the clean latent x0 ("denoised"). Within the
# active window we replace that prediction's low-frequency band with the
# reference's:
#
#     x0' = x0 + w(sigma) * strength * (LP(ref) - LP(x0))
#
# The sampler integrates towards x0', so the layout follows the reference while
# everything above the cutoff is generated natively. Working on x0 rather than
# the noisy latent x_t means no noise-level matching is needed. w(sigma) eases
# the guide out so the model can reconcile details with the pinned layout.

def _resize(x, h, w, mode):
    shape = x.shape
    flat = x.reshape(-1, 1, shape[-2], shape[-1]).float()
    kw = {} if mode == "area" else {"align_corners": False}
    out = F.interpolate(flat, size=(h, w), mode=mode, **kw)
    return out.reshape(*shape[:-2], h, w)


def _lowpass(x, factor):
    """
    Gaussian low-pass in the frequency domain, keeping structure coarser than
    ~`factor` latent pixels. Gaussian = no ringing and no block grid; reflect
    padding stops FFT wrap-around bleeding opposite edges into each other.
    """
    shape = x.shape
    h, w = shape[-2], shape[-1]
    flat = x.reshape(-1, 1, h, w).float()
    pad_h, pad_w = min(3 * factor, h - 1), min(3 * factor, w - 1)
    flat = F.pad(flat, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
    hp, wp = flat.shape[-2], flat.shape[-1]

    sigma_f = 1.0 / (2.0 * factor)                       # cycles per latent pixel
    fy = torch.fft.fftfreq(hp, device=flat.device)[:, None]
    fx = torch.fft.rfftfreq(wp, device=flat.device)[None, :]
    mask = torch.exp(-0.5 * (fy ** 2 + fx ** 2) / sigma_f ** 2)

    out = torch.fft.irfft2(torch.fft.rfft2(flat) * mask, s=(hp, wp))
    out = out[..., pad_h:pad_h + h, pad_w:pad_w + w]
    return out.reshape(shape)


def _falloff_weight(sigma, sigma_start, sigma_end, falloff):
    """1 at the start of the window, easing to 0 at its end."""
    if falloff == "constant" or sigma_start <= sigma_end:
        return 1.0
    t = (sigma_start - sigma) / (sigma_start - sigma_end)
    t = min(max(t, 0.0), 1.0)
    if falloff == "linear":
        return 1.0 - t
    return 0.5 * (1.0 + math.cos(math.pi * t))           # cosine


class LowFrequencyGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "reference": ("LATENT", {"tooltip": "The draft, decoded, upscaled to the final size in PIXEL "
                                                    "space and re-encoded. Upscaling the latent directly "
                                                    "causes a token-grid mesh (see README)."}),
                "lowpass_factor": ("INT", {"default": 6, "min": 1, "max": 64,
                                           "tooltip": "Cutoff in final-resolution latent pixels. Lower pins more "
                                                      "(6 keeps faces/expressions); higher pins only the layout "
                                                      "and lets mid-scale structure be reinvented."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Release the guide by this fraction of the schedule. "
                                                     "Guiding to the end causes ghosting / duplicated features."}),
                "falloff": (["cosine", "linear", "constant"], {"default": "cosine",
                            "tooltip": "How the guide eases out over the window."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches"
    DESCRIPTION = "Pins the low-frequency band of generation to a reference latent, then releases it."

    def patch(self, model, reference, lowpass_factor, strength, start_percent, end_percent, falloff="cosine"):
        ref_raw = reference["samples"]
        if getattr(ref_raw, "is_nested", False):
            raise ValueError("LowFrequencyGuide does not support nested latents.")

        ms = model.get_model_object("model_sampling")
        sigma_start = float(ms.percent_to_sigma(start_percent))
        sigma_end = float(ms.percent_to_sigma(end_percent))
        cache = {}

        def post_cfg(args):
            denoised = args["denoised"]
            sigma = float(args["sigma"].flatten()[0])
            if strength <= 0.0 or sigma > sigma_start + 1e-5 or sigma < sigma_end - 1e-5:
                return denoised
            weight = strength * _falloff_weight(sigma, sigma_start, sigma_end, falloff)
            if weight <= 0.0:
                return denoised

            key = (tuple(denoised.shape), denoised.device, denoised.dtype)
            ref_lp = cache.get(key)
            if ref_lp is None:
                ref = ref_raw.to(device=denoised.device, dtype=torch.float32)
                ref = args["model"].process_latent_in(ref)        # sampler's internal latent space
                if ref.ndim == 4 and denoised.ndim == 5:
                    ref = ref.unsqueeze(2)
                if ref.ndim != denoised.ndim or ref.shape[1] != denoised.shape[1]:
                    raise ValueError(f"Reference latent {tuple(ref.shape)} is incompatible with "
                                     f"the model's latents {tuple(denoised.shape)}.")
                if ref.shape[-2:] != denoised.shape[-2:]:
                    ref = _resize(ref, denoised.shape[-2], denoised.shape[-1], "bicubic")
                if ref.shape[0] != denoised.shape[0]:
                    ref = ref[:1].expand(denoised.shape[0], *ref.shape[1:])
                ref_lp = _lowpass(ref, lowpass_factor).to(denoised.dtype)
                cache.clear()
                cache[key] = ref_lp

            correction = ref_lp - _lowpass(denoised, lowpass_factor).to(denoised.dtype)
            return denoised + weight * correction

        m = model.clone()
        m.set_model_sampler_post_cfg_function(post_cfg)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "MultiscaleSeededNoise": MultiscaleSeededNoise,
    "LowFrequencyGuide": LowFrequencyGuide,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiscaleSeededNoise": "Multiscale Seeded Noise",
    "LowFrequencyGuide": "Low-Frequency Guide (ILVR)",
}
