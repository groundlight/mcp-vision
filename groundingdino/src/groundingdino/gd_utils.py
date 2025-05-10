from pathlib import Path
from io import BytesIO

from PIL import Image
import torch
from torchvision.ops import box_convert
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import predict
import groundingdino.datasets.transforms as T


# Checkpoint and config
parent_path = Path(__file__).parent
config_file = str(parent_path / "config/GroundingDINO_SwinT_OGC.py")
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filename = "groundingdino_swint_ogc.pth"


def load_model_hf(model_config_path, repo_id, filename, device='cpu'):
    args = SLConfig.fromfile(model_config_path) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model    

def image_transform_grounding(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return init_image, image

def image_transform_grounding_for_vis(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return image

model = load_model_hf(config_file, ckpt_repo_id, ckpt_filename)

def run_grounding_dino(
        input_image: Image.Image,
        caption: str,
        box_threshold: float = 0.25,
        text_threahold: float = 0.25) -> dict:
    
    image = input_image.convert("RGB")
    _, image_tensor = image_transform_grounding(image)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    boxes, logits, labels = predict(
        model, image_tensor, caption, box_threshold, text_threahold, device=device
    )

    return {
        "boxes": boxes,
        "logits": logits,
        "labels": labels
    }

def crop_image(image: Image.Image, box: torch.Tensor) -> Image.Image: 
    """ Crop image to box
    box is a torch.Tensor in cx,cy,w,h format -- output of groundingdino inference
    """
    w, h = image.size
    box = box * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=box, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    crop = image.crop(xyxy)
    return crop


if __name__ == "__main__":
    image_path = "/Users/paulina/data/vstar_bench/vstar_bench/GPT4V-hard/0.JPG"
    image = Image.open(image_path)
    caption = "advertising board"

    box_threshold = 0.25
    text_threshold = 0.25

    output_dict = run_grounding_dino(image, caption, box_threshold, text_threshold)

    print(output_dict)

    crop = crop_image(image, output_dict["boxes"][0])

    plt.imshow(crop)
    plt.show()

