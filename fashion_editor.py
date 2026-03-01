import torch
import PIL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from diffusers import StableDiffusionXLInpaintPipeline, EulerDiscreteScheduler, FluxFillPipeline, KandinskyV22InpaintPipeline, KandinskyV22PriorPipeline
import gradio as gr
import time
import warnings
warnings.filterwarnings('ignore')

# ---- Config ----
TARGET_SIZE = (512, 512)

LABELS = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses",
    4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress",
    8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face",
    12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm",
    16: "Bag", 17: "Scarf"
}

# ---- Load Models (once at startup) ----
print("Loading segmentation model...")
seg_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
seg_model = SegformerForSemanticSegmentation.from_pretrained(
    "mattmdjaga/segformer_b2_clothes"
).to("cuda")

print("Loading inpainting model...")
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    scheduler=EulerDiscreteScheduler.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", subfolder="scheduler"
    ),
    torch_dtype=torch.float16
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()
print("Models ready!")

# ---- Core Functions ----
def preprocess_image(image):
    # width, height = image.size
    # image = image.crop((0, height - width, width, height))
    return image.resize(TARGET_SIZE, Image.LANCZOS)

def get_segmentation(image):
    inputs = seg_processor(images=np.asarray(image), return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = seg_model(**inputs)
    return torch.nn.functional.interpolate(
        outputs.logits.cpu(),
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False
    ).argmax(dim=1)[0].numpy()

def create_seg_preview(image, pred):
    try:
        unique_classes = np.unique(pred)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(plt.cm.tab20(pred / pred.max()))
        axes[1].set_title("Segmentation Map")
        axes[1].axis("off")
        axes[1].legend(
            handles=[mpatches.Patch(color=plt.cm.tab20(cls / pred.max()),
                     label=f"[{cls}] {LABELS[cls]}") for cls in unique_classes],
            bbox_to_anchor=(1.05, 1), loc="upper left"
        )
        plt.tight_layout()

        # fix: use buffer_rgba() instead of tostring_rgb()
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        seg_preview = Image.frombuffer(
            "RGBA", (width, height), fig.canvas.buffer_rgba()
        ).convert("RGB")  # convert RGBA -> RGB for Gradio

        plt.close(fig)
        print(f"Preview created successfully: {seg_preview.size}")
        return seg_preview, unique_classes

    except Exception as e:
        import traceback
        print("ERROR in create_seg_preview:")
        print(traceback.format_exc())
        plt.close("all")
        return None, None

def extract_mask(pred, mask_index):
    return PIL.Image.fromarray((pred == mask_index).astype(np.uint8) * 255)

# ---- Gradio Functions ----
def step1_segment(input_image):
    try:
        print("Step 1: Starting segmentation...")

        # preprocessing
        source_image = preprocess_image(input_image)
        print(f"Image preprocessed: {source_image.size}, mode: {source_image.mode}")

        # segmentation
        pred = get_segmentation(source_image)
        print(f"Segmentation done: pred shape={pred.shape}, unique classes={np.unique(pred)}")

        # preview
        seg_preview, unique_classes = create_seg_preview(source_image, pred)
        print(f"Preview created: {type(seg_preview)}, size={seg_preview.size}")

        detected = {LABELS[cls]: int(cls) for cls in unique_classes}
        label_choices = [f"[{v}] {k}" for k, v in detected.items()]
        print(f"Labels detected: {label_choices}")

        return (
            source_image,
            pred,
            seg_preview,
            gr.Dropdown(choices=label_choices,
                        label="Select item to edit",
                        interactive=True)
        )

    except Exception as e:
        import traceback
        print("ERROR in step1_segment:")
        print(traceback.format_exc())  # prints full stack trace
        return None, None, None, None

def step2_preview_mask(pred, selected_label):
    """Show mask preview when user selects a label"""
    if pred is None or selected_label is None:
        return None

    mask_index = int(selected_label.split("]")[0].replace("[", ""))
    mask = extract_mask(pred, mask_index)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(plt.cm.tab20(pred / pred.max()))
    axes[0].set_title("Segmentation Map")
    axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Selected Mask: {selected_label}")
    axes[1].axis("off")
    plt.tight_layout()

    fig.canvas.draw()
    preview = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
    )
    plt.close(fig)
    return preview

def step3_generate(source_image, pred, selected_label, prompt, negative_prompt):
    """Run inpainting"""
    if any(x is None for x in [source_image, pred, selected_label, prompt]):
        return None

    mask_index = int(selected_label.split("]")[0].replace("[", ""))
    mask = extract_mask(pred, mask_index)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=torch.Generator(device="cuda"),
        image=source_image,
        mask_image=mask,
        strength=0.99,
        guidance_scale=8.5
    ).images[0]

    return result

# ---- Gradio UI ----
with gr.Blocks(title="Fashion Inpainting") as demo:
    gr.Markdown("# 👗 Fashion Inpainting")
    gr.Markdown("Upload an image → Segment → Select item → Enter prompt → Generate")

    # shared state
    pred_state = gr.State()
    processed_image_state = gr.State()

    # Step 1 - Upload & Segment
    gr.Markdown("## Step 1 — Upload & Segment")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            segment_btn = gr.Button("Segment Image", variant="primary")
            time.sleep(30)
        seg_preview = gr.Image(label="Segmentation Map")

    # Step 2 - Select Mask
    gr.Markdown("## Step 2 — Select Item to Edit")
    with gr.Row():
        item_dropdown = gr.Dropdown(label="Select item to edit")
        mask_preview = gr.Image(label="Mask Preview")

    # Step 3 - Generate
    gr.Markdown("## Step 3 — Enter Prompt & Generate")
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Edit Prompt",
                placeholder="e.g. blonde hair, blue denim jacket..."
            )
            negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                value="blurry edges, inconsistent lighting, artifacts, distorted, watermark"
            )
            generate_btn = gr.Button("Generate", variant="primary")
        output_image = gr.Image(label="Generated Image")

    # ---- Wire Up ----
    segment_btn.click(
        fn=step1_segment,
        inputs=[input_image],
        outputs=[processed_image_state, pred_state, seg_preview, item_dropdown]
    )

    item_dropdown.change(
        fn=step2_preview_mask,
        inputs=[pred_state, item_dropdown],
        outputs=[mask_preview]
    )

    generate_btn.click(
        fn=step3_generate,
        inputs=[processed_image_state, pred_state, item_dropdown,
                prompt_input, negative_prompt_input],
        outputs=[output_image]
    )

demo.launch(share=True, debug=True)