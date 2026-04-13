import os
import cv2 
import Equirec2Perspec as E2P 

def extract_views(input_image, output_dir, fov=90, height=1080, width=1080):
    equ = E2P.Equirectangular(input_image)    # load panorama image

    
    pitch_values = [0] 
    yaw_values = range(-180, 180, fov)  
    # fov 
    for pitch in pitch_values:
        for yaw in yaw_values:
            # 90 degree fov means top 45 + bottom 45 here
            # pitch refers to vertical axis, where 0 is center of image. The max top is 90 and bottom -90
            # yaw refers to horizontal axis. The max right is 180 and left -180.
            img = equ.GetPerspective(fov, yaw, pitch, height, width)
            output_path = os.path.join(output_dir, f'view_{yaw}_{pitch}.jpg')
            cv2.imwrite(output_path, img)  



if __name__ == '__main__':
    output_dir = 'output_views_2'
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    extract_views('panorama_test.jpg', output_dir, fov=60)