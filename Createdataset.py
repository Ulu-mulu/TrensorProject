from ultralytics.data.annotator import auto_annotate
import torch
from pathlib import Path
from ultralytics import YOLO, SAM
import os
import cv2
import numpy as np
from cv2 import VideoCapture
import random
import shutil
import glob
import yaml


def frame_shatter(path_to_video, folder_name):
    # Создание папки с исходниками
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    # Открытие видео для дробления на кадры
    capture = cv2.VideoCapture(path_to_video)

    FrameNum = 0
    FrameName = 'Test_pic'

    while True:

        success, frame = capture.read()

        if success:
            # Выделение определенного фрагмента
            sliceOfImage = frame

            cv2.imwrite(f'{folder_name}/{FrameName}_{FrameNum}.jpg',
                        sliceOfImage)

        else:
            break

        FrameNum = FrameNum + 1

    capture.release()


def split_dataset(folder_name):
    # Путь к исходной папке с картинками
    source_folder = folder_name

    # Создание папок датасета
    folder_list = ['train', 'val', 'test']
    subfolder_list = ['images', 'labels']
    for folder in folder_list:
        os.makedirs(os.path.join(source_folder, folder), exist_ok=True)
        level_path = os.path.join(source_folder, folder)
        for subfolder in subfolder_list:
            os.makedirs(os.path.join(level_path, subfolder), exist_ok=True)

    # Пути к целевым папкам
    folder_80 = (folder_name + '/train/images')
    folder_15 = (folder_name + '/val/images')
    folder_5 = (folder_name + '/test/images')

    # Получаем список всех картинок в исходной папке
    image_list = glob.glob(source_folder + '/*.jpg', recursive=False)

    # Распределяем картинки по папкам
    for image in image_list:
        random_number = random.randint(1, 100)
        if random_number <= 80:
            shutil.move(image, folder_80)
        elif random_number <= 95:
            shutil.move(image, folder_15)
        else:
            shutil.move(image, folder_5)


def train_on_dataset(folder_name):
    model = YOLO('yolov8m.pt')
    shutil.copy('yolov8m.pt', f'{os.getcwd()}/{folder_name}')
    shutil.copy('yolov8n.pt', f'{os.getcwd()}/{folder_name}')
    old_directory = os.getcwd()
    os.chdir(f'{os.getcwd()}/{folder_name}')
    results = model.train(data='data.yaml', epochs=100, imgsz=640, device=0, auto_augment='autoaugment', copy_paste=0.5,
                          batch=8, cache=True, dropout=0.5)
    os.chdir(old_directory)


def create_yaml(folder_name):
    # Создание данных для YAML файла (словарь в данном случае)
    data = {
        'train': f'{os.getcwd()}/{folder_name}/train/images',
        'val': f'{os.getcwd()}/{folder_name}/val/images',
        'test': f'{os.getcwd()}/{folder_name}/test/images',
        'names': ['Object detected'],
        'nc': 1
    }

    # Путь к файлу YAML
    yaml_file_path = folder_name + '/data.yaml'

    # Сохранение данных в файл в формате YAML
    with open(yaml_file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    print(f'Файл {yaml_file_path} успешно создан и сохранен.')


def auto_annotate_dark(data, det_model="yolov8m-seg.pt", sam_model="sam_b.pt", device="0", output_dir=None, current_id=0):

    det_model = YOLO(det_model)
    sam_model = SAM(sam_model)

    data = Path(data)
    if not output_dir:
        output_dir = data.parent / f"{data.stem}_auto_annotate_labels"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    det_results = det_model(data, stream=True, device=device, classes=[current_id])

    for result in det_results:
        class_ids = result.boxes.cls.int().tolist()  # noqa
        if len(class_ids):
            boxes = result.boxes.xyxy  # Boxes object for bbox outputs
            sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device)
            segments = sam_results[0].masks.xyn  # noqa

            with open(f"{Path(output_dir) / Path(result.path).stem}.txt", "w") as f:
                for i in range(len(segments)):
                    s = segments[i]
                    if len(s) == 0:
                        continue
                    segment = map(str, segments[i].reshape(-1).tolist())
                    f.write(f"{0} " + " ".join(segment) + "\n")


def folder_search(folder_name):
    path = os.getcwd()

    # Получаем список всех папок в директории
    folders = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]

    # Получаем список папок с одинаковой частью имени
    same_name_folders = []

    for folder in folders:
        if folder_name in folder:
            same_name_folders.append(folder)

    return f'{folder_name}{len(same_name_folders)}'


# path_to_video = "RawVideos/Test.avi"
# main_folder_name: str = 'testdir1'

# frame_shatter(path_to_video, main_folder_name)

# split_dataset(main_folder_name)

# folder_list = ['/train', '/val', '/test']

# for folder in folder_list:
    # current_dir = main_folder_name + folder
    # auto_annotate_dark(data=current_dir + '/images', det_model='yolov8x.pt', sam_model='sam_l.pt', device='0', output_dir=current_dir + '/labels')

# create_yaml(main_folder_name)

# train_on_dataset()
