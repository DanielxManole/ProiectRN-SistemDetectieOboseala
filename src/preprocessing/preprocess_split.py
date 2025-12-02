import os
import shutil
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(SCRIPT_DIR, '../../data/raw')
DEST_DIR = os.path.join(SCRIPT_DIR, '../../data/processed')
SPLIT_RATIO = (0.8, 0.1, 0.1)

def create_dirs():
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    for split in ['train', 'validation', 'test']:
        for class_name in ['Open', 'Closed']:
            os.makedirs(os.path.join(DEST_DIR, split, class_name), exist_ok=True)

def split_data():
    classes = ['Open', 'Closed']

    for class_name in classes:
        src_path = os.path.join(SOURCE_DIR, class_name)
        images = [f for f in os.listdir(src_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)
        
        train_count = int(len(images) * SPLIT_RATIO[0])
        val_count = int(len(images) * SPLIT_RATIO[1])
        
        train_imgs = images[:train_count]
        val_imgs = images[train_count:train_count + val_count]
        test_imgs = images[train_count + val_count:]
        
        def copy_images(img_list, split_type):
            for img in img_list:
                src = os.path.join(src_path, img)
                dst = os.path.join(DEST_DIR, split_type, class_name, img)
                shutil.copy(src, dst)
        
        print(f"Procesare clasa {class_name}...")
        copy_images(train_imgs, 'train')
        copy_images(val_imgs, 'validation')
        copy_images(test_imgs, 'test')
        
        print(f"--> {class_name}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

if __name__ == "__main__":
    print("Începere preprocesare și splituire date...")
    create_dirs()
    split_data()
    print("Gata! Datele sunt organizate în data/processed.")