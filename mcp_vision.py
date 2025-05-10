import time
from typing import Any
import base64
from io import BytesIO
import requests
import logging
import sys

from mcp.server.fastmcp import FastMCP
from PIL import Image
from groundingdino.gd_utils import run_grounding_dino, crop_image

logger = logging.getLogger(__name__)


DEFAULT_OBJDET_MODEL = "google/owlvit-base-patch32"

global models
models = {}


mcp = FastMCP("mcp-vision")


def pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str


def base64_to_pil(data_base64: str) -> Image.Image:
    image = Image.open(BytesIO(base64.b64decode(data_base64)))
    return image


def load_hf_objdet_pipeline(model_name: str, device: str = "cpu"):
    start = time.time()
    objdet_model = pipeline(task="zero-shot-object-detection", model=model_name)
    print(f"Loaded zero-shot object detection pipline for {model_name} in {time.time() - start:.2f} seconds.")
    return {"model_name": model_name, "model": objdet_model}


def locate_object_bboxes(image_path: str, candidate_labels: list[str]) -> list[dict[str, Any]] | None:
    """
    Locate objects in image, output list of dictionaries using HF object detection format
    Example:
    [
        {'score': 0.997, 'label': 'bird', 'box': {'xmin': 69, 'ymin': 171, 'xmax': 396, 'ymax': 507}}, 
        {'score': 0.999, 'label': 'bird', 'box': {'xmin': 398, 'ymin': 105, 'xmax': 767, 'ymax': 507}}
    ]

    :param image_path: local path or URL to image
    :param candidate_labels: list of candidate object labels as strings
    """
    global models
    objdet_model = models["object_detection"]["model"]
    try:
        return objdet_model(image_path, candidate_labels=candidate_labels)
    except Exception as e:
        print(f"Error while calling model with image: {e}")
        return None


def init_objdet_pipeline(hf_model: str | None = None) -> None:
    global models

    if hf_model is None:
        hf_model = DEFAULT_OBJDET_MODEL

    objdet_config = models.get("object_detection", None)
    if not objdet_config:
        print("First time loading an object detection model")
        objdet_config = load_hf_objdet_pipeline(model_name=hf_model)
        models["object_detection"] = objdet_config

    objdet_model_name = objdet_config.get("model_name", None)    
    objdet_model = objdet_config.get("model", None)

    if not objdet_model_name or not objdet_model or objdet_model_name != hf_model:
        print(f"No object detection model exists for {hf_model=} (existing {objdet_model_name=}), reloading")
        objdet_config = load_hf_objdet_pipeline(model_name=hf_model)
        models["object_detection"] = objdet_config


@mcp.tool()
def locate_objects(image_path: str, candidate_labels: list[str], hf_model: str | None = None) -> str:
    """Detect, find and/or locate objects in the image found at image_path.

    Args:
        image_path: path to the image
        candidate_labels: list of candidate object labels as strings
        hf_model (optional): huggingface zero-shot object detection model (default = "google/owlvit-base-patch32")
    """
    init_objdet_pipeline(hf_model)

    bboxes = locate_object_bboxes(image_path, candidate_labels=candidate_labels)
    if not bboxes or len(bboxes) == 0:
        return f"No objects were located in the image."

    return f"{len(bboxes)} objects were found in the image at the following locations: {bboxes}."


# TODO: await requests result and make this async
# @mcp.tool()
# def zoom_to_object(image_path: str, label: str, hf_model: str | None = None) -> Any:
#     """Zoom into an object in the image, allowing you to analyze it more closely. Crop image to the object bounding box and return the cropped image. 
#     If many objects are present in the image, will return the 'best' one as represented by object score. 

#     Args:
#         image_path: path to the imamge
#         label: object label to find and crop to
#         hf_model (optional): huggingface zero-shot object detection model (default = "google/owlvit-base-pathch32")
#     """
#     init_objdet_pipeline(hf_model)

#     bboxes = locate_object_bboxes(image_path, candidate_labels=[label])
#     if not bboxes or len(bboxes) == 0:
#         return None
    
#     bboxes = sorted(bboxes, key=lambda x: x["score"], reverse=True) # this may be superfluous as hf models return in this order already
#     if not bboxes or len(bboxes) == 0:
#         return None
    
#     best_box = bboxes[0]
#     left, top, right, bottom = best_box["box"].values()

#     # image = Image.open(image_path)
#     image = Image.open(requests.get(image_path, stream=True).raw)

#     crop = image.crop((left, top, right, bottom))

#     crop_base64 = pil_to_base64(crop)
#     if len(crop_base64) > 500 * 1024:
#         print(f"Crop is too large, max allowed 500KB, got {len(crop_base64)}")
#         return None

#     return crop_base64


@mcp.tool()
def zoom_to_object(image_path: str, label: str) -> Any:
    """Zoom into an object in the image, allowing you to analyze it more closely. Crop image to the object bounding box and return the cropped image. 
    If many objects are present in the image, will return the 'best' one as represented by object score. 

    Args:
        image_path: path to the imamge
        label: object label to find and crop to
    """
    if image_path.startswith("http"):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)

    box_threshold = 0.25
    text_threshold = 0.25

    output_dict = run_grounding_dino(image, label, box_threshold, text_threshold)

    print(output_dict)

    crop = crop_image(image, output_dict["boxes"][0])

    crop_base64 = pil_to_base64(crop)
    if len(crop_base64) > 500 * 1024:
        print(f"Crop is too large, max allowed 500KB, got {len(crop_base64)}")
        return None

    return crop_base64


def main():
    try:
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()