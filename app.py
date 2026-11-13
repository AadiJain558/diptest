

from __future__ import annotations

import io
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image, ImageFilter, ImageOps
from plotly.colors import sample_colorscale
from skimage.filters import sobel
from skimage.restoration import denoise_bilateral
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


st.set_page_config(
    page_title="Image 3D Reconstruction",
    page_icon="🧊",
    layout="wide",
)

MODEL_OPTIONS = {
    "Depth Anything V2 Small (recommended)": "depth-anything/Depth-Anything-V2-Small-hf",
    "DPT-Hybrid MiDaS": "Intel/dpt-hybrid-midas",
}

FILTER_DESCRIPTIONS: Dict[str, str] = {
    "Mean": "Softens noise by averaging pixels in a local neighborhood.",
    "Median": "Replaces pixels with the median of neighbors to remove salt-and-pepper noise.",
    "Gaussian": "Applies Gaussian smoothing for gentle noise reduction.",
    "Bilateral": "Smooths while attempting to preserve edges using spatial and colour distance.",
    "Sharpen": "Boosts local edge contrast using unsharp masking.",
    "Sobel": "Highlights intensity changes and object boundaries.",
}

DEPTH_COLOR_MAPS = [
    "Viridis",
    "Turbo",
    "Inferno",
    "Magma",
    "Plasma",
    "Cividis",
    "Greys",
]





def _hash_image(image: Image.Image) -> bytes:
    """Return stable image bytes for Streamlit's image cache key."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _pil_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _array_to_pil(array: np.ndarray) -> Image.Image:
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


@st.cache_resource(show_spinner=False)
def load_depth_model(model_id: str):
    """Load and cache a Hugging Face monocular depth-estimation model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
    except OSError as exc:
        raise RuntimeError(
            f"Could not download or load '{model_id}' from Hugging Face. "
            "Check your internet connection, model ID, and Hugging Face cache."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "The depth model could not be initialized. Try updating Transformers "
            "with: pip install -U transformers"
        ) from exc

    model.to(device)
    model.eval()
    return processor, model, device


@st.cache_data(show_spinner=False, hash_funcs={Image.Image: _hash_image})
def estimate_depth_map(image: Image.Image, model_id: str) -> np.ndarray:
    """Run depth inference and return a 2-D relative depth map in [0, 1]."""
    processor, model, device = load_depth_model(model_id)
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        prediction = outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(image.height, image.width),
            mode="bicubic",
            align_corners=False,
        )[0, 0]

    depth = prediction.detach().float().cpu().numpy()
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    depth_range = depth_max - depth_min

    if depth_range < 1e-8:
        return np.zeros_like(depth, dtype=np.float32)

    depth = (depth - depth_min) / depth_range
    return depth.astype(np.float32)


@st.cache_resource(show_spinner=False)
def load_background_removal_session():
    """Create one reusable rembg session. rembg is an optional dependency."""
    try:
        from rembg import new_session
    except ImportError as exc:
        raise RuntimeError(
            "Background removal requires rembg. Install it with "
            '`pip install "rembg[cpu]"` and restart the app.'
        ) from exc
    return new_session("isnet-general-use")


@st.cache_data(show_spinner=False, hash_funcs={Image.Image: _hash_image})
def remove_background(image: Image.Image) -> Image.Image:
    """Return an RGBA image with the background made transparent."""
    try:
        from rembg import remove
    except ImportError as exc:
        raise RuntimeError(
            "Background removal requires rembg. Install it with "
            '`pip install "rembg[cpu]"` and restart the app.'
        ) from exc

    session = load_background_removal_session()
    result = remove(image.convert("RGB"), session=session)
    if isinstance(result, Image.Image):
        return result.convert("RGBA")
    return Image.open(io.BytesIO(result)).convert("RGBA")





def get_filter_controls() -> Dict[str, float]:
    """Render shared filter controls in the sidebar and return their values."""
    with st.sidebar.expander("DIP filter parameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            mean_radius = st.slider("Mean radius", 1, 10, 3)
            median_kernel = st.slider("Median kernel", 3, 15, 5, step=2)
            gaussian_sigma = st.slider("Gaussian sigma", 0.5, 5.0, 1.5, step=0.1)
        with col2:
            bilateral_color = st.slider("Bilateral colour σ", 0.01, 0.5, 0.1, step=0.01)
            bilateral_space = st.slider("Bilateral spatial σ", 1.0, 10.0, 3.0, step=0.5)
            sharpen_amount = st.slider("Sharpen amount", 0.5, 2.5, 1.2, step=0.1)

    return {
        "mean_radius": float(mean_radius),
        "median_kernel": float(median_kernel),
        "gaussian_sigma": float(gaussian_sigma),
        "bilateral_color": float(bilateral_color),
        "bilateral_space": float(bilateral_space),
        "sharpen_amount": float(sharpen_amount),
    }


def apply_one_filter(
    image: Image.Image,
    filter_name: str,
    controls: Dict[str, float],
) -> Tuple[Image.Image, str]:
    """Apply one selected DIP filter and return the result plus parameter text."""
    source = image.convert("RGB")

    if filter_name == "Mean":
        radius = controls["mean_radius"]
        return source.filter(ImageFilter.BoxBlur(radius)), f"radius={radius:g}"

    if filter_name == "Median":
        size = int(controls["median_kernel"])
        size = size if size % 2 else size + 1
        return source.filter(ImageFilter.MedianFilter(size=size)), f"kernel={size}"

    if filter_name == "Gaussian":
        sigma = controls["gaussian_sigma"]
        return source.filter(ImageFilter.GaussianBlur(radius=sigma)), f"sigma={sigma:.1f}"

    if filter_name == "Bilateral":
        sigma_color = controls["bilateral_color"]
        sigma_space = controls["bilateral_space"]
        filtered = denoise_bilateral(
            _pil_to_array(source),
            sigma_color=sigma_color,
            sigma_spatial=sigma_space,
            channel_axis=-1,
        )
        return _array_to_pil(filtered), f"σ_colour={sigma_color:.2f}, σ_space={sigma_space:.2f}"

    if filter_name == "Sharpen":
        amount = controls["sharpen_amount"]
        result = source.filter(
            ImageFilter.UnsharpMask(
                radius=2,
                percent=int(amount * 100),
                threshold=5,
            )
        )
        return result, f"amount={amount:.2f}"

    if filter_name == "Sobel":
        gray = np.asarray(source.convert("L"), dtype=np.float32) / 255.0
        edges = sobel(gray)
        if float(edges.max()) > 0:
            edges = edges / float(edges.max())
        edge_rgb = np.repeat(edges[..., np.newaxis], 3, axis=2)
        return _array_to_pil(edge_rgb), "normalized edge magnitude"

    raise ValueError(f"Unknown filter: {filter_name}")


def apply_filter_sequence(
    original: Image.Image,
    selected_filters: List[str],
    controls: Dict[str, float],
) -> Image.Image:
    """Apply the chosen filters sequentially; this image feeds depth inference."""
    processed = original.convert("RGB")
    for filter_name in selected_filters:
        processed, _ = apply_one_filter(processed, filter_name, controls)
    return processed


def apply_dip_filters(
    original: Image.Image,
    selected_filters: List[str],
    controls: Dict[str, float],
) -> List[Dict[str, object]]:
    """Create independent filter previews for the DIP lab."""
    results: List[Dict[str, object]] = []
    for name in selected_filters:
        preview, parameters = apply_one_filter(original, name, controls)
        results.append(
            {
                "Filter": name,
                "Description": FILTER_DESCRIPTIONS[name],
                "Parameters": parameters,
                "Preview": preview,
            }
        )
    return results





def colorize_depth(depth: np.ndarray, color_map: str) -> Image.Image:
    """Convert a normalized depth array to an RGB preview using a Plotly scale."""
    if color_map == "Greys":
        gray = np.clip(depth * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(np.repeat(gray[..., np.newaxis], 3, axis=2))

    palette = sample_colorscale(color_map, np.linspace(0.0, 1.0, 256))
    lookup_table = []
    for color in palette:
        channels = [int(value) for value in re.findall(r"\d+", color)[:3]]
        lookup_table.append(channels)
    lookup = np.asarray(lookup_table, dtype=np.uint8)
    indices = np.clip(depth * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(lookup[indices])


def sample_point_cloud(
    depth: np.ndarray,
    image: Image.Image,
    *,
    depth_scale: float = 1.5,
    max_points: int = 30000,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a depth map to a spatially distributed, coloured point cloud.

    Pixels are selected on a 2-D grid to avoid the horizontal banding that can
    occur when sampling a flattened image array. If a foreground mask is
    supplied, only its valid pixels are included. A deterministic random fill
    is used when the mask removes too many grid samples.
    """
    height, width = depth.shape
    total_pixels = height * width
    target_points = max(1, min(int(max_points), total_pixels))

    if valid_mask is None:
        mask = np.ones((height, width), dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != depth.shape:
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
            mask = np.asarray(
                mask_image.resize((width, height), Image.Resampling.NEAREST)
            ) > 0

    valid_flat = np.flatnonzero(mask.reshape(-1))
    if valid_flat.size == 0:
        raise ValueError("The foreground mask contains no visible pixels.")

    # Estimate a grid spacing that yields approximately target_points samples.
    spacing = max(1, int(np.sqrt(total_pixels / target_points)))
    row_positions = np.arange(spacing // 2, height, spacing, dtype=np.int32)
    col_positions = np.arange(spacing // 2, width, spacing, dtype=np.int32)
    if row_positions.size == 0:
        row_positions = np.array([height // 2], dtype=np.int32)
    if col_positions.size == 0:
        col_positions = np.array([width // 2], dtype=np.int32)

    grid_rows, grid_cols = np.meshgrid(row_positions, col_positions, indexing="ij")
    sampled_flat = (grid_rows * width + grid_cols).reshape(-1)
    sampled_flat = sampled_flat[mask.reshape(-1)[sampled_flat]]

    rng = np.random.default_rng(42)
    sampled_flat = np.unique(sampled_flat)

    if sampled_flat.size > target_points:
        sampled_flat = np.sort(
            rng.choice(sampled_flat, size=target_points, replace=False)
        )
    elif sampled_flat.size < target_points:
        remaining_count = min(target_points - sampled_flat.size, valid_flat.size - sampled_flat.size)
        if remaining_count > 0:
            sampled_set = np.zeros(total_pixels, dtype=bool)
            sampled_set[sampled_flat] = True
            remaining = valid_flat[~sampled_set[valid_flat]]
            extra = rng.choice(remaining, size=remaining_count, replace=False)
            sampled_flat = np.concatenate((sampled_flat, extra))

    pixel_y, pixel_x = np.unravel_index(sampled_flat, (height, width))
    x = (pixel_x.astype(np.float32) / max(width - 1, 1)) * 2.0 - 1.0
    y = 1.0 - (pixel_y.astype(np.float32) / max(height - 1, 1)) * 2.0
    z = depth[pixel_y, pixel_x].astype(np.float32) * float(depth_scale)

    rgb_image = np.asarray(image.convert("RGB").resize((width, height)), dtype=np.uint8)
    colors = rgb_image[pixel_y, pixel_x].astype(np.float32) / 255.0
    return x, y, z, colors


def _plotly_rgb_colors(colors: np.ndarray) -> List[str]:
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in rgb]


def render_point_cloud(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    colors: np.ndarray,
) -> go.Figure:
    """Build an interactive Plotly 3-D point-cloud visualization."""
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker={
                    "size": 2,
                    "color": _plotly_rgb_colors(colors),
                    "opacity": 0.9,
                },
                hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Depth: %{z:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Relative depth",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        height=650,
        showlegend=False,
    )
    return fig


def point_cloud_to_ply(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    colors: np.ndarray,
) -> bytes:
    """Serialize XYZ and RGB data into an ASCII PLY point-cloud file."""
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(x)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    rows = [
        f"{x[i]:.6f} {y[i]:.6f} {z[i]:.6f} "
        f"{int(rgb[i, 0])} {int(rgb[i, 1])} {int(rgb[i, 2])}"
        for i in range(len(x))
    ]
    return (header + "\n".join(rows) + "\n").encode("utf-8")





def main() -> None:
    st.title("Single Image → 3D Reconstruction")
    st.write(
        "Upload an RGB photograph, estimate its depth, and turn the visible scene "
        "into an interactive coloured 3-D point cloud."
    )

    st.sidebar.header("Reconstruction settings")
    selected_model_label = st.sidebar.selectbox(
        "Depth estimation model",
        list(MODEL_OPTIONS.keys()),
        index=0,
        help="Depth Anything V2 Small is selected by default. DPT-Hybrid MiDaS remains available for comparison.",
    )
    model_id = MODEL_OPTIONS[selected_model_label]
    max_points = st.sidebar.slider(
        "Point-cloud density",
        min_value=5000,
        max_value=60000,
        value=30000,
        step=5000,
    )
    depth_scale = st.sidebar.slider(
        "Depth exaggeration",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1,
    )
    depth_color_map = st.sidebar.selectbox(
        "Depth-map colour map",
        DEPTH_COLOR_MAPS,
        index=0,
    )
    remove_bg_enabled = st.sidebar.checkbox(
        "Remove background before reconstruction",
        value=False,
        help='Requires the optional package: pip install "rembg[cpu]"',
    )

    st.sidebar.subheader("Preprocessing before depth estimation")
    preprocess_filters = st.sidebar.multiselect(
        "Apply DIP filters in this order",
        list(FILTER_DESCRIPTIONS.keys()),
        default=[],
        help="Selected filters are applied sequentially before the depth model runs. Leave empty to use the unfiltered image.",
    )
    filter_controls = get_filter_controls()
    if "Sobel" in preprocess_filters:
        st.sidebar.warning(
            "Sobel converts the image to edge intensity. It is useful for experiments, "
            "but can reduce depth-estimation quality."
        )

    st.sidebar.caption(
        "Depth values are relative estimates. They are not calibrated distances in metres."
    )

    uploaded_file = st.file_uploader(
        "Drop a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_file is None:
        st.info("Upload an image to begin.")
        st.markdown(
            "**Pipeline:** image upload → optional background removal and DIP preprocessing "
            "→ depth estimation → point-cloud generation → interactive 3-D view."
        )
        return

    try:
        image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    except Exception as exc:
        st.error(f"The uploaded file could not be opened as an image: {exc}")
        return

    image_col, metadata_col = st.columns([2, 1])
    with image_col:
        st.image(image, caption="Original input image", use_container_width=True)
    with metadata_col:
        file_size_kb = getattr(uploaded_file, "size", len(uploaded_file.getvalue())) / 1024
        metadata = pd.DataFrame(
            [
                {"Property": "Resolution", "Value": f"{image.width} × {image.height}"},
                {"Property": "Colour mode", "Value": image.mode},
                {"Property": "File size", "Value": f"{file_size_kb:.1f} KB"},
                {"Property": "Selected model", "Value": selected_model_label},
            ]
        )
        st.dataframe(metadata, use_container_width=True, hide_index=True)

    
    foreground_mask: Optional[np.ndarray] = None
    depth_input = image
    background_removed_image: Optional[Image.Image] = None

    if remove_bg_enabled:
        with st.spinner("Removing the image background..."):
            try:
                background_removed_image = remove_background(image)
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:
                st.error(f"Background removal failed: {exc}")
                st.stop()

        rgba = np.asarray(background_removed_image, dtype=np.uint8)
        foreground_mask = rgba[..., 3] > 16
        white_background = Image.new("RGBA", background_removed_image.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(white_background, background_removed_image)
        depth_input = composited.convert("RGB")

    if preprocess_filters:
        with st.spinner("Applying selected image-processing filters..."):
            depth_input = apply_filter_sequence(depth_input, preprocess_filters, filter_controls)

    with st.spinner(f"Estimating depth with {selected_model_label}..."):
        try:
            depth = estimate_depth_map(depth_input, model_id)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(
                "Depth estimation failed. Check that Transformers and PyTorch are "
                f"installed correctly. Details: {exc}"
            )
            st.stop()

    if foreground_mask is not None:
        depth_display = depth.copy()
        depth_display[~foreground_mask] = 0.0
    else:
        depth_display = depth

    try:
        x, y, z, colors = sample_point_cloud(
            depth,
            image,
            depth_scale=depth_scale,
            max_points=max_points,
            valid_mask=foreground_mask,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    figure = render_point_cloud(x, y, z, colors)
    ply_data = point_cloud_to_ply(x, y, z, colors)
    depth_preview = colorize_depth(depth_display, depth_color_map)
    depth_png_buffer = io.BytesIO()
    depth_preview.save(depth_png_buffer, format="PNG")

    tab_reconstruction, tab_dip = st.tabs(["3-D Reconstruction", "Image Filters Lab"])

    with tab_reconstruction:
        st.subheader("Reconstruction input")
        processing_col, depth_col = st.columns(2)
        with processing_col:
            st.image(
                depth_input,
                caption="Image actually sent to the depth model",
                use_container_width=True,
            )
        with depth_col:
            st.image(
                depth_preview,
                caption=f"Estimated depth map — {depth_color_map}",
                use_container_width=True,
            )

        if background_removed_image is not None:
            st.image(
                background_removed_image,
                caption="Foreground after background removal",
                use_container_width=True,
            )

        st.subheader("Interactive 3-D point cloud")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Rendered points", f"{len(x):,}")
        metric_col2.metric("Depth-map resolution", f"{depth.shape[1]} × {depth.shape[0]}")
        metric_col3.metric("Compute device", "GPU" if torch.cuda.is_available() else "CPU")

        st.plotly_chart(figure, use_container_width=True)

        download_col1, download_col2 = st.columns(2)
        with download_col1:
            st.download_button(
                label="Download point cloud (PLY)",
                data=ply_data,
                file_name="single_image_point_cloud.ply",
                mime="application/octet-stream",
                use_container_width=True,
            )
        with download_col2:
            st.download_button(
                label="Download depth map (PNG)",
                data=depth_png_buffer.getvalue(),
                file_name="estimated_depth_map.png",
                mime="image/png",
                use_container_width=True,
            )

        st.caption(
            "This is a coloured point cloud generated from relative monocular depth. "
            "It does not infer hidden surfaces or create a watertight 3-D mesh."
        )

    with tab_dip:
        st.subheader("Digital Image Processing Playground")
        st.caption(
            "Preview each classical filter independently. To use filters in the 3-D "
            "pipeline, select them in the sidebar under Preprocessing before depth estimation."
        )

        st.write("**Filters applied to the reconstruction input:**")
        if preprocess_filters:
            st.write(" → ".join(preprocess_filters))
            st.image(depth_input, caption="Final preprocessed reconstruction input", use_container_width=True)
        else:
            st.info("No DIP filters are currently applied to reconstruction.")

        preview_defaults = ["Mean", "Median", "Bilateral", "Sharpen"]
        preview_filters = st.multiselect(
            "Choose filters to preview",
            list(FILTER_DESCRIPTIONS.keys()),
            default=preview_defaults,
        )

        if not preview_filters:
            st.info("Select at least one filter to view its output.")
        else:
            try:
                filter_results = apply_dip_filters(image, preview_filters, filter_controls)
            except Exception as exc:
                st.error(f"Could not compute filter previews: {exc}")
                filter_results = []

            if filter_results:
                summary = pd.DataFrame(
                    [
                        {
                            "Filter": result["Filter"],
                            "Description": result["Description"],
                            "Parameters": result["Parameters"],
                        }
                        for result in filter_results
                    ]
                )
                st.dataframe(summary, use_container_width=True, hide_index=True)

                preview_columns = st.columns(3)
                for index, result in enumerate(filter_results):
                    with preview_columns[index % 3]:
                        st.image(
                            result["Preview"],
                            caption=str(result["Filter"]),
                            use_container_width=True,
                        )


if __name__ == "__main__":
    main()
