✨ Highlights
🏆 Model Zoo
We release three series of models, each tailored for specific use cases in visual geometry.

🌟 DA3 Main Series (DA3-Giant, DA3-Large, DA3-Base, DA3-Small) These are our flagship foundation models, trained with a unified depth-ray representation. By varying the input configuration, a single model can perform a wide range of tasks:

🌊 Monocular Depth Estimation: Predicts a depth map from a single RGB image.
🌊 Multi-View Depth Estimation: Generates consistent depth maps from multiple images for high-quality fusion.
🎯 Pose-Conditioned Depth Estimation: Achieves superior depth consistency when camera poses are provided as input.
📷 Camera Pose Estimation: Estimates camera extrinsics and intrinsics from one or more images.
🟡 3D Gaussian Estimation: Directly predicts 3D Gaussians, enabling high-fidelity novel view synthesis.
📐 DA3 Metric Series (DA3Metric-Large) A specialized model fine-tuned for metric depth estimation in monocular settings, ideal for applications requiring real-world scale.

🔍 DA3 Monocular Series (DA3Mono-Large). A dedicated model for high-quality relative monocular depth estimation. Unlike disparity-based models (e.g., Depth Anything 2), it directly predicts depth, resulting in superior geometric accuracy.

🔗 Leveraging these available models, we developed a nested series (DA3Nested-Giant-Large). This series combines a any-view giant model with a metric model to reconstruct visual geometry at a real-world metric scale.

🛠️ Codebase Features
Our repository is designed to be a powerful and user-friendly toolkit for both practical application and future research.

🎨 Interactive Web UI & Gallery: Visualize model outputs and compare results with an easy-to-use Gradio-based web interface.
⚡ Flexible Command-Line Interface (CLI): Powerful and scriptable CLI for batch processing and integration into custom workflows.
💾 Multiple Export Formats: Save your results in various formats, including glb, npz, depth images, ply, 3DGS videos, etc, to seamlessly connect with other tools.
🔧 Extensible and Modular Design: The codebase is structured to facilitate future research and the integration of new models or functionalities.
🚀 Quick Start
📦 Installation
pip install xformers torch\>=2 torchvision
pip install -e . # Basic
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70 # for gaussian head
pip install -e ".[app]" # Gradio, python>=3.10
pip install -e ".[all]" # ALL
For detailed model information, please refer to the Model Cards section below.

💻 Basic Usage
import glob, os, torch
from depth_anything_3.api import DepthAnything3
device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)
example_path = "assets/examples/SOH"
images = sorted(glob.glob(os.path.join(example_path, "*.png")))
prediction = model.inference(
    images,
)
# prediction.processed_images : [N, H, W, 3] uint8   array
print(prediction.processed_images.shape)
# prediction.depth            : [N, H, W]    float32 array
print(prediction.depth.shape)  
# prediction.conf             : [N, H, W]    float32 array
print(prediction.conf.shape)  
# prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
print(prediction.extrinsics.shape)
# prediction.intrinsics       : [N, 3, 3]    float32 array
print(prediction.intrinsics.shape)
export MODEL_DIR=depth-anything/DA3NESTED-GIANT-LARGE
# This can be a Hugging Face repository or a local directory
# If you encounter network issues, consider using the following mirror: export HF_ENDPOINT=https://hf-mirror.com
# Alternatively, you can download the model directly from Hugging Face
export GALLERY_DIR=workspace/gallery
mkdir -p $GALLERY_DIR

# CLI auto mode with backend reuse
da3 backend --model-dir ${MODEL_DIR} --gallery-dir ${GALLERY_DIR} # Cache model to gpu
da3 auto assets/examples/SOH \
    --export-format glb \
    --export-dir ${GALLERY_DIR}/TEST_BACKEND/SOH \
    --use-backend

# CLI video processing with feature visualization
da3 video assets/examples/robot_unitree.mp4 \
    --fps 15 \
    --use-backend \
    --export-dir ${GALLERY_DIR}/TEST_BACKEND/robo \
    --export-format glb-feat_vis \
    --feat-vis-fps 15 \
    --process-res-method lower_bound_resize \
    --export-feat "11,21,31"

# CLI auto mode without backend reuse
da3 auto assets/examples/SOH \
    --export-format glb \
    --export-dir ${GALLERY_DIR}/TEST_CLI/SOH \
    --model-dir ${MODEL_DIR}