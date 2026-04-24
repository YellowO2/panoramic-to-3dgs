Installation
pip install -r requirements.txt
wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt

Download model
wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt

Using the mode:
To use a manually downloaded checkpoint, specify it with the -c flag:
sharp predict -i /path/to/input/images -o /path/to/output/gaussians -c sharp_2572gikvuh.pt
For our case:
sharp predict -i ./output_views/view_0_0.jpg -o ./output_3dgs -c ./models/sharp_2572gikvuh.pt


huggingface-cli download depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  --local-dir ./da3_model
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
to use depth model:
da3 auto ./output_views     --export-format glb     --export-dir ./output_depth     --model-dir ./models/models--depth-anything--DA3-LARGE-1.1/snapshots/0e109ae307c5982f319a67cf6f9f99ccdc0ec97c
da3 auto ./output_views     --export-format glb     --export-dir ./output_depth     --model-dir ./models/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7



da3 auto ./all_views     --export-format glb     --export-dir ./output_depth     --model-dir ./models/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7


  Architectural Summary:
   - datatype.py: Defines View objects, which is a data type which contains necessary info to generate and process a splat
   - components/DepthMapGenerator/:
       - DA360DepthModel: Handles panorama-level depth.
       - DA3Model: Handles multi-view depth and pose inference.
   - components/ViewExtractor/: Slices the panorama (and optional depth map) into standard View objects.
   - components/SplatGenerator/: Generates 3DGS from View images.
   - components/SplatProcessor/: A "model-free" component that aligns and merges splats based on whatever depth or extrinsics are found in
     the View objects.
   - main.py: Clean, readable orchestrator with a simple depth_mode toggle.