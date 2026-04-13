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