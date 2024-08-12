

import torch
import comfy.supported_models
import comfy.model_base
import comfy.ldm.flux.model
import comfy.model_patcher


def MZ_EmptyModel_call(kwargs):
    import comfy.supported_models
    import comfy.model_base
    import comfy.ldm.flux.model
    import comfy.model_patcher

    model_type = kwargs["model_type"]

    if model_type == "Flux":
        model_config = comfy.supported_models.Flux(unet_config={
            "image_model": "flux",
            "in_channels": 16,
            "vec_in_dim": 768,
            "context_in_dim": 4096,
            "hidden_size": 3072,
            "mlp_ratio": 4.0,
            "num_heads": 24,
            "depth": 19,
            "depth_single_blocks": 38,
            "axes_dim": [16, 56, 56],
            "theta": 10000,
            "qkv_bias": True,
            "guidance_embed": True
        })
        model_config.manual_cast_dtype = torch.bfloat16

        model = comfy.model_base.Flux(
            model_config=model_config,
            device=None,
        )

        return (model,)


def MZ_ModelLoadStateDict_call(kwargs):
    model = kwargs["model"]
    safetensors_file = kwargs["safetensors_file"]
    import comfy.model_patcher

    import safetensors.torch
    from comfy import model_management

    print(f"Load state_dict from {safetensors_file}")

    state_dict = safetensors.torch.load_file(safetensors_file)

    from . import q_awq_marlin_loader
    model.diffusion_model = q_awq_marlin_loader.quantize_loader(
        model=model.diffusion_model,
        state_dict=state_dict,
        bits=4,
    )

    load_device = model_management.get_torch_device()
    offload_device = model_management.unet_offload_device()
    model_patcher = comfy.model_patcher.ModelPatcher(
        model, load_device=load_device, offload_device=offload_device)

    return (model_patcher,)


def MZ_FluxQuantizeUNETLoader_call(kwargs):
    unet_name = kwargs["unet_name"]
    import folder_paths
    unet_path = folder_paths.get_full_path("unet", unet_name)

    model = MZ_EmptyModel_call({"model_type": "Flux"})[0]

    model_patcher = MZ_ModelLoadStateDict_call({
        "model": model,
        "safetensors_file": unet_path,
    })[0]

    return (model_patcher,)
