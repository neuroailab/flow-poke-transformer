import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from torchvision.utils import flow_to_image
import cv2

print("Loading FPT model...")
model = torch.hub.load(".", "fpt_base", source="local")
model.eval()
model.to('cuda')
print("Model loaded!")

def clear_gpu_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def center_crop_to_square(image, size=256):
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    image_cropped = image.crop((left, top, right, bottom))

    image_resized = image_cropped.resize((size, size), Image.BICUBIC)

    return image_resized


def process_image_for_model(image):
    # (1, 3, H, W) in [-1, 1] for processing
    image_np = np.array(image).astype(np.float32) / 255.0  # [0, 1]
    image_t = torch.from_numpy(image_np).permute(2, 0, 1)  # (3, H, W)
    image_t = image_t.unsqueeze(0)  # (1, 3, H, W)
    image_t = image_t * 2 - 1  # [-1, 1]
    # breakpoint

    return image_t.to('cuda')


def generate_unconditional_flow(image, grid_size):
    print(f"Processing image with grid_size={grid_size}...")

    image_t = process_image_for_model(image)
    device = image_t.device

    from flow_poke.model import make_axial_pos_2d
    query_pos = make_axial_pos_2d(grid_size, grid_size, device=device)[None]  # (1, N, 2)

    print("Embedding image...")
    d_img = model.embed_image(image_t)

    poke_pos = torch.empty((1, 0, 2), device=device)
    poke_flow = torch.empty((1, 0, 2), device=device)

    print("Predicting unconditional flow distribution...")
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with torch.no_grad():  # Add this to prevent gradient tracking
            dist = model.predict_parallel(
                poke_pos=poke_pos,
                poke_flow=poke_flow,
                query_pos=query_pos,
                camera_static=True,
                d_img=d_img
            )

    print("Computing flow statistics...")

    expected_flow = dist.mean[0].cpu().numpy()  # (N, 2)
    flow_field = expected_flow.reshape(grid_size, grid_size, 2)  # (H, W, 2)

    mixture_probs = dist.mixture_distribution.probs[0].cpu().numpy()  # (N, 4)
    component_means = dist.component_distribution.loc[0].cpu().numpy()  # (N, 4, 2)

    del dist
    torch.cuda.empty_cache()

    # Compute flow magnitude
    flow_magnitude = np.linalg.norm(flow_field, axis=-1)  # (H, W)


    print("Creating visualizations...")

    flow_torch = torch.from_numpy(flow_field).permute(2, 0, 1).float()  # (2, H, W)
    flow_rgb = flow_to_image(flow_torch).permute(1, 2, 0).numpy()  # (H, W, 3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    ax = axes[0]
    ax.imshow(image)
    ax.set_title('Input Image (256×256)', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Flow direction (flow_to_image)
    ax = axes[1]
    ax.imshow(flow_rgb)
    ax.set_title('Flow Direction\n(Hue=Direction, Brightness=Magnitude)', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Flow magnitude
    ax = axes[2]
    im = ax.imshow(flow_magnitude, cmap='viridis')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f'Flow Magnitude\nMean: {flow_magnitude.mean():.4f}', fontsize=11, fontweight='bold')
    ax.axis('off')

    plt.suptitle('Unconditional Flow Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    viz2 = Image.open(buf)

    print("Done!")

    del expected_flow, flow_field, mixture_probs, component_means
    del flow_torch, flow_rgb, image_t, d_img
    clear_gpu_memory()

    return viz2, None

def process_and_generate(image, grid_size):
    if image is None:
        return None, None, "Please upload an image first"

    # Center crop to 256x256
    image_cropped = center_crop_to_square(image, size=256)

    # Generate flow
    viz, _ = generate_unconditional_flow(image_cropped, grid_size)

    clear_gpu_memory()

    return image_cropped, viz, None

# Create Gradio interface
with gr.Blocks(title="FPT Unconditional Flow Visualization") as demo:
    gr.Markdown("""
    # Flow Poke Transformer: Unconditional Flow vis
    1. Takes image and center-crops it to 256×256
    2. Predicts natural motion patterns (no pokes applied)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Image",
                sources=["upload"],
            )

            grid_size_slider = gr.Slider(
                minimum=16,
                maximum=256,
                value=64,
                step=8,
                label="Grid Resolution (flow field resolution)"
            )

            generate_btn = gr.Button("Generate Unconditional Flow", variant="primary", size="lg")

            cropped_output = gr.Image(
                label="Preprocessed (256×256 center crop)",
                type="pil"
            )

            stats_output = gr.Markdown(label="Statistics")

        with gr.Column(scale=2):
            flow_viz_output = gr.Image(label="Unconditional Flow Analysis")
            gr.Markdown("""Visualization shows: Input image (256×256), Flow direction (hue=direction, brightness=magnitude), Flow magnitude heatmap""")

    # Connect button
    generate_btn.click(
        fn=process_and_generate,
        inputs=[image_input, grid_size_slider],
        outputs=[cropped_output, flow_viz_output, stats_output]
    )


# Launch
if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )