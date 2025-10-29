"""
SpelkeBench Gradio Demo
Created segmentation script (dot prod methodolgogy) for FPT visualization
Only works on single point prompt (user denotes centroid)
TODO: allow user drag
"""

import os
import math
import torch
import numpy as np
import cv2
import gradio as gr
from einops import rearrange
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from flow_poke.model import FlowPokeTransformer_Base, make_axial_pos_2d, MixtureSameFamily
from torchvision.utils import flow_to_image
from scripts.utils import set_seed, offset_multiple_centroids, pixel_to_normalized, normalized_to_pixel, reconstruct_mixture_distribution

os.environ["GRADIO_TEMP_DIR"] = "/ccn2/u/lilianch/tmp"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def predict_and_segment(image_input, centroid_x, centroid_y, num_offsets, flow_resolution: int,
        prediction_mode, ar_downsampling_factor, seed, model, device):
    set_seed(seed)

    if image_input is None:
        return None, None, None, None, None

    # og img shape
    H_orig, W_orig = image_input.shape[:2]
    target_size = 256

    # crop to square (centr crop)
    min_dim = min(H_orig, W_orig)
    start_h = (H_orig - min_dim) // 2
    start_w = (W_orig - min_dim) // 2
    image_cropped = image_input[start_h:start_h + min_dim, start_w:start_w + min_dim]
    image_resized = np.array(Image.fromarray(image_cropped).resize((target_size, target_size), Image.BICUBIC))

    # adjust centroid for center crop + resize
    centroid_cropped_x = centroid_x - start_w
    centroid_cropped_y = centroid_y - start_h

    # scale
    centroid_resized = np.array([
        centroid_cropped_x * (target_size / min_dim),
        centroid_cropped_y * (target_size / min_dim)
    ])

    # Normalize to [0, 1]
    centroid_norm = torch.tensor([
        centroid_resized[0] / target_size,
        centroid_resized[1] / target_size
    ], dtype=torch.float32, device=device).unsqueeze(0)  # [1, 2]

    # Embed image
    d_img = {k: v.clone() for k, v in model.embed_image(
            rearrange(torch.from_numpy(image_resized), "h w c -> 1 c h w")
            .float().div(127.5).sub(1.0).to(device)).items()}

    # offsets
    dx, dy = offset_multiple_centroids(centroid_norm, num_offsets, min_mag=10.0/256.0, max_mag=25.0/256.0) # j assume 256 size image bc coords in [0,1]

    query_pos = make_axial_pos_2d(flow_resolution, flow_resolution, device=device)[None]

    all_flows = []
    all_drags_pixel = []

    query_batch_size = 4096

    if prediction_mode == "parallel":
        for j in range(num_offsets):
            poke_start = centroid_norm
            poke_end = centroid_norm + torch.stack([dx[j], dy[j]], dim=0)[None]

            # poke_pos: tensor([[[0.7388, 0.1116]]], device='cuda:0')
            # poke_flow: tensor([[[0.1004, 0.0223]]], device='cuda:0') for top right of img
            poke_pos = poke_start.unsqueeze(0)  # [1, 1, 2]
            poke_flow = (poke_end - poke_start).unsqueeze(0)  # [1, 1, 2]

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):

                num_queries = query_pos.shape[1]
                flow_chunks = []

                for i in range(0, num_queries, query_batch_size):
                    query_batch = query_pos[:, i:i + query_batch_size]

                    pred = model.predict_parallel(
                        poke_pos=poke_pos,
                        poke_flow=poke_flow,
                        query_pos=query_batch,
                        camera_static=True,
                        d_img=d_img
                    )
                    flow_chunks.append(pred.mean)

                flow_pred = torch.cat(flow_chunks, dim=1)
                flow = rearrange(
                    flow_pred,
                    "b (h w) c -> b h w c",
                    h=flow_resolution,
                    w=flow_resolution
                )

            all_flows.append(flow)

            #  drag in pixel coords
            drag_start_pixel = normalized_to_pixel(poke_start, (target_size, target_size))
            drag_end_pixel = normalized_to_pixel(poke_end, (target_size, target_size))
            drag_pixel = (
                drag_start_pixel[0, 0],
                drag_start_pixel[0, 1],
                drag_end_pixel[0, 0],
                drag_end_pixel[0, 1]
            )
            all_drags_pixel.append(drag_pixel)

    elif prediction_mode == "autoregressive":
        # autoregressive
        ar_flow_resolution = flow_resolution // ar_downsampling_factor
        query_pos_ar = make_axial_pos_2d(ar_flow_resolution, ar_flow_resolution, device=device)[None]

        for j in range(num_offsets):
            poke_start = centroid_norm
            poke_end = centroid_norm + torch.stack([dx[j], dy[j]], dim=0)[None]

            poke_pos = poke_start.unsqueeze(0)  # [1, 1, 2]
            poke_flow = (poke_end - poke_start).unsqueeze(0)  # [1, 1, 2]

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                # Autoregressive prediction at lower resolution
                flow_ar = rearrange(
                    model.predict_autoregressive(
                        poke_pos=poke_pos,
                        poke_flow=poke_flow,
                        query_pos=query_pos_ar,
                        camera_static=True,
                        d_img=d_img,
                        randomize_order=True,
                    ),
                    "b (h w) c -> b h w c",
                    h=ar_flow_resolution,
                    w=ar_flow_resolution,
                )

                # Upsample using parallel prediction
                if ar_downsampling_factor != 1:
                    num_queries = query_pos.shape[1]
                    flow_chunks = []

                    for i in range(0, num_queries, query_batch_size):
                        query_batch = query_pos[:, i:i + query_batch_size]

                        pred = model.predict_parallel(
                            poke_pos=torch.cat([poke_pos, query_pos_ar], dim=1),
                            poke_flow=torch.cat(
                                [poke_flow, rearrange(flow_ar, "b h w c -> b (h w) c")],
                                dim=1
                            ),
                            query_pos=query_batch,
                            camera_static=True,
                            d_img=d_img,
                        )
                        flow_chunks.append(pred.mean)

                    flow = rearrange(
                        torch.cat(flow_chunks, dim=1),
                        "b (h w) c -> b h w c",
                        h=flow_resolution,
                        w=flow_resolution,
                    )
                else:
                    flow = flow_ar

            all_flows.append(flow)

            #  drag in pixel coords
            drag_start_pixel = normalized_to_pixel(poke_start, (target_size, target_size))
            drag_end_pixel = normalized_to_pixel(poke_end, (target_size, target_size))
            drag_pixel = (
                drag_start_pixel[0, 0],
                drag_start_pixel[0, 1],
                drag_end_pixel[0, 0],
                drag_end_pixel[0, 1]
            )
            all_drags_pixel.append(drag_pixel)

    flows = torch.cat(all_flows, dim=0)  # [num_offsets, H, W, 2]

    # get segmentation
    all_dot_prods = []
    for i in range(num_offsets):
        dot_prod = get_dot_product_map(flows[i], all_drags_pixel[i])
        all_dot_prods.append(dot_prod)

    all_dot_prods = torch.stack(all_dot_prods, dim=0)
    mean_dot_prod = all_dot_prods.mean(dim=0).cpu().numpy()
    segment = threshold_heatmap(mean_dot_prod)

    segment_resized = cv2.resize(
        segment.astype(np.uint8),
        (W_orig, H_orig),
        interpolation=cv2.INTER_NEAREST
    )

    vis_outputs = create_visualizations(
        image_input,
        centroid_x,
        centroid_y,
        all_drags_pixel,
        flows,
        mean_dot_prod,
        segment_resized,
        target_size
    )

    return vis_outputs


def create_visualizations( rgb, centroid_x, centroid_y, drags_pixel,
        flows, mean_dot_prod, segment, target_size):
    H, W = rgb.shape[:2]

    scale_x = W / target_size
    scale_y = H / target_size

    drags_rgb = []
    for drag in drags_pixel:
        x1, y1, x2, y2 = drag
        drags_rgb.append((x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y))

    # 1. RGB with drags
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.imshow(rgb)
    ax1.plot(centroid_x, centroid_y, 'r*', markersize=20, label='Centroid')
    for i, drag in enumerate(drags_rgb):
        x1, y1, x2, y2 = drag
        ax1.arrow(x1, y1, x2 - x1, y2 - y1, color='yellow', width=2,
                  head_width=10, head_length=10, alpha=0.7, label=f'Drag {i + 1}' if i < 2 else '')
    ax1.set_title('Input with Poke Directions', fontsize=14, fontweight='bold')
    ax1.axis('off')
    ax1.legend()
    plt.tight_layout()

    fig1.canvas.draw()
    vis_drags = np.array(fig1.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig1)

    # 2. Flow visualizations (first 2 flows)
    fig2, axes = plt.subplots(2, 2, figsize=(10, 10))

    for idx in range(min(2, flows.shape[0])):
        # Magnitude
        flow = flows[idx].cpu().float()
        flow_mag = torch.norm(flow, dim=-1).numpy()
        im = axes[idx, 0].imshow(flow_mag, cmap='jet')
        plt.colorbar(im, ax=axes[idx, 0])
        axes[idx, 0].set_title(f'Flow {idx + 1} Magnitude')
        axes[idx, 0].axis('off')

        # Direction
        flow_rgb = flow_to_image(flow.permute(2, 0, 1)).permute(1, 2, 0).numpy()
        axes[idx, 1].imshow(flow_rgb)
        axes[idx, 1].set_title(f'Flow {idx + 1} Direction')
        axes[idx, 1].axis('off')

    plt.tight_layout()
    fig2.canvas.draw()
    vis_flows = np.array(fig2.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig2)

    # 3. Dot product map
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    im = ax3.imshow(mean_dot_prod, cmap='RdBu_r')
    plt.colorbar(im, ax=ax3)

    # Overlay first drag arrow
    flow_h, flow_w = mean_dot_prod.shape
    scale_x_flow = flow_w / target_size
    scale_y_flow = flow_h / target_size
    x1, y1, x2, y2 = drags_pixel[0]
    ax3.arrow(
        x1 * scale_x_flow, y1 * scale_y_flow,
        (x2 - x1) * scale_x_flow, (y2 - y1) * scale_y_flow,
        color='yellow', width=2, head_width=8, alpha=0.8
    )
    ax3.set_title('Average Dot Product Map', fontsize=14, fontweight='bold')
    ax3.axis('off')
    plt.tight_layout()

    fig3.canvas.draw()
    vis_dotprod = np.array(fig3.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig3)

    # 4. Segmentation
    fig4, ax4 = plt.subplots(figsize=(6, 6))
    ax4.imshow(rgb)
    ax4.imshow(segment, alpha=0.5, cmap='Reds')
    ax4.plot(centroid_x, centroid_y, 'r*', markersize=20)
    ax4.set_title(f'Predicted Segment ({segment.sum()} pixels)', fontsize=14, fontweight='bold')
    ax4.axis('off')
    plt.tight_layout()

    fig4.canvas.draw()
    vis_segment = np.array(fig4.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig4)

    # 5. Binary mask only
    fig5, ax5 = plt.subplots(figsize=(6, 6))
    ax5.imshow(segment, cmap='gray')
    ax5.set_title('Binary Segmentation Mask', fontsize=14, fontweight='bold')
    ax5.axis('off')
    plt.tight_layout()

    fig5.canvas.draw()
    vis_mask = np.array(fig5.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig5)

    return vis_drags, vis_flows, vis_dotprod, vis_segment, vis_mask


def demo(
        checkpoint: str = None,
        device: str = "cuda",
        share: bool = False,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
):

    # Load model
    print("Loading model...")
    if checkpoint is not None:
        model = FlowPokeTransformer_Base()
        checkpoint_data = torch.load(checkpoint)
        model.load_state_dict(checkpoint_data['model'])
    else:
        model = torch.hub.load(".", "fpt_base", source="local")

    model.eval()
    model.to(device)
    model.requires_grad_(False)
    print(f"Model loaded on {device}")

    with gr.Blocks(title="SpelkeBench Interactive Segmentation") as demo_app:
        gr.Markdown("# SpelkeBench Interactive Segmentation")
        gr.Markdown("""
        How u can use!
        1. Upload an image
        2. Click on the object you want to segment to place the poke point
        3. Adjust parameters (optional in advanced)
        4. Click "Segment" to see results
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Input Image",
                    type="numpy",
                    image_mode="RGB"
                )

                with gr.Row():
                    centroid_x = gr.Number(label="Centroid X", value=128, precision=0)
                    centroid_y = gr.Number(label="Centroid Y", value=128, precision=0)

                def update_centroid(img, evt: gr.SelectData):
                    """Update centroid when user clicks on image."""
                    if img is not None:
                        return evt.index[0], evt.index[1]
                    return 128, 128

                image_input.select(
                    update_centroid,
                    inputs=[image_input],
                    outputs=[centroid_x, centroid_y]
                )

                with gr.Accordion("Advanced Settings", open=False):
                    num_offsets = gr.Slider(
                        minimum=2,
                        maximum=8,
                        value=4,
                        step=1,
                        label="Number of Poke Directions"
                    )
                    flow_resolution = gr.Dropdown(
                        choices=[128, 256],
                        value=256,
                        label="Flow Resolution"
                    )
                    prediction_mode = gr.Dropdown(  # NEW
                        choices=["parallel", "autoregressive"],
                        value="parallel",
                        label="Prediction Mode",
                        info="Parallel is faster, autoregressive may be more accurate"
                    )
                    ar_downsampling_factor = gr.Dropdown(  # NEW
                        choices=[4, 8],
                        value=8,
                        label="AR Downsampling Factor",
                        info="Only used in autoregressive mode. Higher = faster but lower initial resolution.",
                        visible=False  # Hidden by default
                    )
                    seed = gr.Number(
                        label="Random Seed",
                        value=42,
                        precision=0
                    )

                def update_ar_visibility(mode):
                    return gr.update(visible=(mode == "autoregressive"))

                prediction_mode.change(
                    update_ar_visibility,
                    inputs=[prediction_mode],
                    outputs=[ar_downsampling_factor]
                )

                segment_btn = gr.Button("Segment", variant="primary", size="lg")

        with gr.Row():
            output_drags = gr.Image(label="1. Input with Poke Directions", type="numpy")
            output_flows = gr.Image(label="2. Flow Predictions", type="numpy")

        with gr.Row():
            output_dotprod = gr.Image(label="3. Dot Product Map", type="numpy")
            output_segment = gr.Image(label="4. Segmentation Overlay", type="numpy")

        with gr.Row():
            output_mask = gr.Image(label="5. Binary Mask", type="numpy")

        def run_segmentation(img, cx, cy, n_off, f_res, pred_mode, ar_down, sd):
            """Wrapper for predict_and_segment."""
            if img is None:
                gr.Warning("Please upload an image first!")
                return None, None, None, None, None

            return predict_and_segment(
                image_input=img,
                centroid_x=cx,
                centroid_y=cy,
                num_offsets=int(n_off),
                flow_resolution=int(f_res),
                prediction_mode=pred_mode,
                ar_downsampling_factor=int(ar_down),
                seed=int(sd),
                model=model,
                device=torch.device(device)
            )

        segment_btn.click(
            run_segmentation,
            inputs=[image_input, centroid_x, centroid_y, num_offsets, flow_resolution,
                    prediction_mode, ar_downsampling_factor, seed],
            outputs=[output_drags, output_flows, output_dotprod, output_segment, output_mask]
        )

        gr.Examples(
            examples=[
                ["example.jpg", 150, 200, 4, 256, "parallel", 8, 42],
            ],
            inputs=[image_input, centroid_x, centroid_y, num_offsets, flow_resolution,
                    prediction_mode, ar_downsampling_factor, seed],
        )

    demo_app.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        debug=True
    )


if __name__ == "__main__":
    import fire

    fire.Fire(demo)