import os

img_path = 'sroie-2019/raw/img'
box_path = 'sroie-2019/raw/box'

img_files = [i.split('.')[0] for i in os.listdir(img_path) if i.endswith('.jpg')]
box_files = [i.split('.')[0] for i in os.listdir(box_path) if i.endswith('.csv')]

for box_file in box_files:
    if box_file not in img_files:
        print(box_file)
        os.remove(os.path.join(box_path, box_file+'.csv'))

