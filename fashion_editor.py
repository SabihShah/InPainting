import torch
import PIL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from diffusers import StableDiffusionXLInpaintPipeline, EulerDiscreteScheduler
import gradio as gr
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

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        seg_preview = Image.frombuffer(
            "RGBA", (width, height), fig.canvas.buffer_rgba()
        ).convert("RGB")

        plt.close(fig)
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

        source_image = preprocess_image(input_image)
        print(f"Image preprocessed: {source_image.size}, mode: {source_image.mode}")

        pred = get_segmentation(source_image)
        print(f"Segmentation done: pred shape={pred.shape}, unique classes={np.unique(pred)}")

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
        print(traceback.format_exc())
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
    preview = Image.frombuffer(
        "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
    ).convert("RGB")
    plt.close(fig)
    return preview


def add_edit_to_queue(pred, selected_label, prompt, current_edits):
    """Add selected label + prompt pair to the edit queue"""
    if pred is None or selected_label is None:
        return current_edits, "⚠️ Please segment an image first"

    if not prompt.strip():
        return current_edits, "⚠️ Please enter a prompt for the selected item"

    mask_index = int(selected_label.split("]")[0].replace("[", ""))
    current_edits = current_edits or []

    # prevent duplicate labels in queue
    existing_labels = [e["label"] for e in current_edits]
    if selected_label in existing_labels:
        return current_edits, f"⚠️ '{selected_label}' is already in the queue. Clear the queue to start over."

    current_edits.append({
        "label": selected_label,
        "mask_index": mask_index,
        "prompt": prompt
    })

    summary = "\n".join([f"{i+1}. {e['label']}  →  {e['prompt']}" for i, e in enumerate(current_edits)])
    return current_edits, summary


def clear_edits():
    """Clear all edits from the queue"""
    return [], ""


def step3_generate(source_image, pred, edits, negative_prompt):
    """Apply each mask+prompt pair sequentially"""
    if source_image is None or pred is None:
        return None, "⚠️ Please upload and segment an image first"

    if not edits:
        return None, "⚠️ No edits in queue. Add at least one edit first"

    current_image = source_image
    log = []

    for i, edit in enumerate(edits):
        print(f"Applying edit {i+1}/{len(edits)}: {edit['label']} → {edit['prompt']}")
        mask = extract_mask(pred, edit["mask_index"])

        try:
            current_image = pipe(
                prompt=edit["prompt"],
                negative_prompt=negative_prompt,
                generator=torch.Generator(device="cuda"),
                image=current_image,
                mask_image=mask,
                strength=0.99,
                guidance_scale=8.5
            ).images[0]
            log.append(f"✅ {i+1}. {edit['label']} → {edit['prompt']}")

        except Exception as e:
            import traceback
            print(f"ERROR on edit {i+1}:")
            print(traceback.format_exc())
            log.append(f"❌ {i+1}. {edit['label']} failed: {str(e)}")
            continue

    return current_image, "\n".join(log)


# ---- Gradio UI ----
with gr.Blocks(title="Fashion Inpainting") as demo:
    gr.Markdown("# 👗 Fashion Inpainting")
    gr.Markdown("Upload an image → Segment → Add edits → Generate")

    # shared state
    pred_state = gr.State()
    processed_image_state = gr.State()
    edits_state = gr.State(value=[])

    # ── Step 1 ──────────────────────────────────────────
    gr.Markdown("## Step 1 — Upload & Segment")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            segment_btn = gr.Button("Segment Image", variant="primary")
        seg_preview = gr.Image(label="Segmentation Map")

    # ── Step 2 ──────────────────────────────────────────
    gr.Markdown("## Step 2 — Select Items & Add Edits")
    with gr.Row():
        item_dropdown = gr.Dropdown(label="Select item to edit")
        mask_preview = gr.Image(label="Mask Preview")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt for selected item",
                placeholder="e.g. blonde hair, blue denim jacket with gold buttons..."
            )
            with gr.Row():
                add_edit_btn = gr.Button("+ Add to Queue", variant="secondary")
                clear_edits_btn = gr.Button("🗑 Clear Queue", variant="stop")
        with gr.Column():
            edit_queue_display = gr.Textbox(
                label="Edit Queue",
                placeholder="Added edits will appear here...",
                interactive=False,
                lines=6
            )

    # ── Step 3 ──────────────────────────────────────────
    gr.Markdown("## Step 3 — Generate")
    with gr.Row():
        with gr.Column():
            negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                value="blurry edges, inconsistent lighting, artifacts, distorted, watermark"
            )
            generate_btn = gr.Button("🎨 Generate All Edits", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Final Result")
            generation_log = gr.Textbox(
                label="Generation Log",
                interactive=False,
                lines=4
            )

    # ── Wire Up ─────────────────────────────────────────
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

    add_edit_btn.click(
        fn=add_edit_to_queue,
        inputs=[pred_state, item_dropdown, prompt_input, edits_state],
        outputs=[edits_state, edit_queue_display]
    )

    clear_edits_btn.click(
        fn=clear_edits,
        outputs=[edits_state, edit_queue_display]
    )

    generate_btn.click(
        fn=step3_generate,
        inputs=[processed_image_state, pred_state, edits_state, negative_prompt_input],
        outputs=[output_image, generation_log]
    )

demo.launch(share=True, debug=True)