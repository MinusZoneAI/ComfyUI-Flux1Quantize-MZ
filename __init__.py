import inspect
import json
import os
import traceback
import folder_paths
import importlib
from . import mz_flux_quantize
import comfy

AUTHOR_NAME = "MinusZone"
CATEGORY_NAME = f"{AUTHOR_NAME} - Flux1Quantize"


class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


NODE_CLASS_MAPPINGS = {
}


NODE_DISPLAY_NAME_MAPPINGS = {
}


class MZ_Flux1QuantizeUNETLoader():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "unet_name": (folder_paths.get_filename_list("unet"), ),
                }}

    RETURN_TYPES = ("MODEL",)

    RETURN_NAMES = ("model",)

    FUNCTION = "load_unet"

    CATEGORY = CATEGORY_NAME

    def load_unet(self, **kwargs):
        importlib.reload(mz_flux_quantize)
        return mz_flux_quantize.MZ_Flux1QuantizeUNETLoader_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_Flux1QuantizeUNETLoader"] = MZ_Flux1QuantizeUNETLoader
NODE_DISPLAY_NAME_MAPPINGS["MZ_Flux1QuantizeUNETLoader"] = f"{AUTHOR_NAME} - Flux1QuantizeUNETLoader"
