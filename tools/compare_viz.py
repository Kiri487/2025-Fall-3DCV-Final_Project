import os
import cv2
import numpy as np
from PIL import Image

EXP_PATHS = {
    'ResNet': './exp/epropnp_basic/2025-12-09T08-32-41.196248',
    'Swin Transformer': './exp/epropnp_swin_basic/2025-12-08T17-05-55.338687',
    'ConvNeXt': './exp/epropnp_convnext_basic/2025-12-09T00-02-05.249352',
    'HRNet': './exp/epropnp_hrnet_basic/2025-12-10T00-18-8.678907'
}

EPOCHS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
TARGET_IMG_IDS = list(range(0, 401, 50))

OUTPUT_DIR = './exp/comparison_viz'

LABEL_W = 400
SPACING = 20

def get_image(root_path, epoch, img_id, file_suffix):
    folder = os.path.join(root_path, f'test_vis_{epoch}')
    filename = f'{img_id}_{file_suffix}'
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        return cv2.imread(path)
    return None

def add_spacing(img, right_spacing=SPACING, bottom_spacing=SPACING):
    if img is None:
        return None
    h, w = img.shape[:2]
    new_img = np.ones((h + bottom_spacing, w + right_spacing, 3), dtype=np.uint8) * 255
    new_img[:h, :w] = img
    return new_img

def generate_gif_for_id(img_id):
    print(f"Generating GIF for Image ID: {img_id}...")
    frames = []
    
    first_exp_path = list(EXP_PATHS.values())[0]
    first_epoch = EPOCHS[0]
    img_inp_raw = get_image(first_exp_path, first_epoch, img_id, 'inp.png')
    
    if img_inp_raw is not None:
        img_inp_array = img_inp_raw.copy()
        input_h, input_w = img_inp_array.shape[:2]
    else:
        img_inp_array = None
        input_h, input_w = 256, 256
    
    for epoch in EPOCHS:
        model_rows = []
        
        for model_name, exp_path in EXP_PATHS.items():
            img_conf = get_image(exp_path, epoch, img_id, 'conf_pred.png')
            img_coor_x = get_image(exp_path, epoch, img_id, 'coor_x_pred.png')
            img_coor_y = get_image(exp_path, epoch, img_id, 'coor_y_pred.png')
            img_coor_z = get_image(exp_path, epoch, img_id, 'coor_z_pred.png')
            img_sphere = get_image(exp_path, epoch, img_id, 'orient_distr.png')

            if img_conf is None:
                full_row = np.zeros((256, LABEL_W + 1000, 3), dtype=np.uint8)
                cv2.putText(full_row, f"{model_name} (Missing)", (10, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            else:
                def resize_to_size(img, target_size=200):
                    if img is None:
                        return None
                    h, w = img.shape[:2]
                    return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                
                img_conf = resize_to_size(img_conf, 200)
                img_coor_x = resize_to_size(img_coor_x, 200)
                img_coor_y = resize_to_size(img_coor_y, 200)
                img_coor_z = resize_to_size(img_coor_z, 200)
                img_sphere = resize_to_size(img_sphere, 256)
                
                all_imgs = [img_conf, img_coor_x, img_coor_y, img_coor_z, img_sphere]
                max_h = max(img.shape[0] for img in all_imgs if img is not None)
                
                label_img = np.ones((max_h, LABEL_W, 3), dtype=np.uint8) * 255
                text_size = cv2.getTextSize(model_name, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                text_x = (LABEL_W - text_size[0]) // 2
                text_y = (max_h + text_size[1]) // 2
                cv2.putText(label_img, model_name, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3, cv2.LINE_AA)
                
                def pad_to_height(img, target_h):
                    if img is None:
                        return np.ones((target_h, 100, 3), dtype=np.uint8) * 255
                    h, w = img.shape[:2]
                    if h == target_h:
                        return img
                    padded = np.ones((target_h, w, 3), dtype=np.uint8) * 255
                    y_offset = (target_h - h) // 2
                    padded[y_offset:y_offset+h, :] = img
                    return padded
                
                img_conf = pad_to_height(img_conf, max_h)
                img_coor_x = pad_to_height(img_coor_x, max_h)
                img_coor_y = pad_to_height(img_coor_y, max_h)
                img_coor_z = pad_to_height(img_coor_z, max_h)
                img_sphere = pad_to_height(img_sphere, max_h)
                
                label_img = add_spacing(label_img, right_spacing=SPACING, bottom_spacing=0)
                img_conf = add_spacing(img_conf, right_spacing=SPACING, bottom_spacing=0)
                img_coor_x = add_spacing(img_coor_x, right_spacing=SPACING, bottom_spacing=0)
                img_coor_y = add_spacing(img_coor_y, right_spacing=SPACING, bottom_spacing=0)
                img_coor_z = add_spacing(img_coor_z, right_spacing=SPACING, bottom_spacing=0)
                img_sphere = add_spacing(img_sphere, right_spacing=0, bottom_spacing=0)  # 最後一張不加右側間距

                imgs_list = [label_img, img_conf, img_coor_x, img_coor_y, img_coor_z, img_sphere]
                full_row = np.hstack(imgs_list)
            
            model_rows.append(full_row)
        
        rows_with_spacing = []
        for i, row in enumerate(model_rows):
            if i < len(model_rows) - 1:
                row_with_spacing = add_spacing(row, right_spacing=0, bottom_spacing=SPACING)
                rows_with_spacing.append(row_with_spacing)
            else:
                rows_with_spacing.append(row)
        
        epoch_frame = np.vstack(rows_with_spacing)

        header_h = 80
        header = np.ones((header_h, epoch_frame.shape[1], 3), dtype=np.uint8) * 255

        if img_conf is not None:
            img_w = img_conf.shape[1] - SPACING 
            x_start = LABEL_W + SPACING
            titles = ['Confidence', 'Coord X', 'Coord Y', 'Coord Z', 'Orientation']
            
            for i, title in enumerate(titles):
                text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                if i < 4:
                    text_x = x_start + i * (img_w + SPACING) + img_w // 2 - text_size[0] // 2
                else:
                    sphere_w = img_sphere.shape[1] if img_sphere is not None else img_w
                    text_x = x_start + i * (img_w + SPACING) + sphere_w // 2 - text_size[0] // 2
                cv2.putText(header, title, (text_x, 52), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3, cv2.LINE_AA)
        
        epoch_frame = np.vstack([header, epoch_frame])

        if img_inp_array is not None:
            left_col_w = input_w + SPACING * 2
            left_col_h = epoch_frame.shape[0]
            left_col = np.ones((left_col_h, left_col_w, 3), dtype=np.uint8) * 255

            y_start = header_h + SPACING
            left_col[y_start:y_start+input_h, SPACING:SPACING+input_w] = img_inp_array

            epoch_frame = np.hstack([left_col, epoch_frame])
        
        cv2.putText(epoch_frame, f"Epoch {epoch}", (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3, cv2.LINE_AA)
        
        frame_rgb = cv2.cvtColor(epoch_frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        frames.append(pil_frame)

    save_path = os.path.join(OUTPUT_DIR, f'comparison_viz_{img_id}.gif')
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,
        loop=0,
        optimize=False
    )
    print(f"  -> Saved: {save_path}")
    return save_path

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_gif_paths = []

    for img_id in TARGET_IMG_IDS:
        gif_path = generate_gif_for_id(img_id)
        all_gif_paths.append(gif_path)

    print("\nStitching all GIFs together...")
    all_frames = []
    
    for gif_path in all_gif_paths:
        gif = Image.open(gif_path)
        for frame_idx in range(gif.n_frames):
            gif.seek(frame_idx)
            all_frames.append(gif.copy())
    
    final_save_path = os.path.join(OUTPUT_DIR, 'comparison_viz_all.gif')
    all_frames[0].save(
        final_save_path,
        save_all=True,
        append_images=all_frames[1:],
        duration=500,
        loop=0,
        optimize=False
    )
    print(f"All done! Final GIF saved to: {final_save_path}")

if __name__ == '__main__':
    main()