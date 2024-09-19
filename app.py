
import gradio as gr
from PIL import Image
import torch
from segment_anything import SamPredictor, sam_model_registry
from diffusers import StableDiffusionInpaintPipeline
from groundingdino.util.inference import load_model, predict, annotate
from GroundingDINO.groundingdino.util import box_ops
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision import transforms as T
# Set up models
device = "cuda"
model_type = "vit_h"
predictor = SamPredictor(sam_model_registry[model_type](checkpoint="text_based_img_editing/sam_vit_h_4b8939.pth").to(device=device))
pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16).to(device)
groundingdino_model = load_model("text_based_img_editing/GroundingDINO_SwinT_OGC.py", "text_based_img_editing/groundingdino_swint_ogc.pth")

def show_mask(mask, image, random_color=True):

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)    
   # mask_img = mask_image.cpu().numpy()
  #  new_img = image.cpu().numpy()
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def process_boxes(boxes, src):
    H, W, _ = src.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    return predictor.transform.apply_boxes_torch(boxes_xyxy, src.shape[:2]).to(device)

def edit_image(image, item, prompt, box_threshold, text_threshold):
    src,img = load_image(image)
    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=img,
        caption=item,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    predictor.set_image(src)
    new_boxes = process_boxes(boxes, src)
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=new_boxes,
        multimask_output=False,
    )
    #img_annotated_mask = show_mask(masks[0][0].cpu(),annotate(image_source=src, boxes=boxes, logits=logits, phrases=phrases)[...,::-1])
    edited_image =pipe(prompt=prompt,
        image=image.resize((512, 512)),
        mask_image=Image.fromarray(masks[0][0].cpu().numpy()).resize((512, 512))
    ).images[0]
    return edited_image
def load_image(image):
    transform = T.Compose(
        [
            T.Resize(800),  # Resize to a fixed size (800, 800) or use RandomResizedCrop for random sizes
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # to_pil = ToPILImage()
    # image_source = to_pil(image)
    # image = np.asarray(image_source)
    image_transformed = transform(image)  # Transform the PIL image directly
    image = np.asarray(image)
    return image, image_transformed
def gradio_interface(image, item, prompt, box_threshold, text_threshold):
    # image_tensor = transforms.ToTensor()(image)
    # image_tensor = image_tensor.to(device)
    edited_image = edit_image(image, item, prompt, box_threshold, text_threshold)
    return edited_image

# Gradio app
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(label="Item"),
        gr.Textbox(label="Prompt"),
        gr.Slider(minimum=0, maximum=1, value=0.5, label="Box Threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.2, label="Text Threshold"),
    ],
    outputs=gr.Image(type="pil"),
    title="Image Inpainting",
    description="Upload an image and specify your editing criteria to see the edited result."
)


iface.launch()
