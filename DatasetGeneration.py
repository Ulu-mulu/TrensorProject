import os
import random

# Путь к папке с изображениями
images_path = "ShatteredFrames/"

# Путь к папке с аннотациями
annotations_path = "ShatteredLabels/"

# Создаем файлы списка изображений для обучения и тестирования
with open("train.txt", "w") as train_file, open("test.txt", "w") as test_file:
    images = os.listdir(images_path)
    random.shuffle(images)
    split_ratio = 0.8  # Доля изображений для обучения

    for i, image in enumerate(images):
        annotation = os.path.join(annotations_path, image.split('.')[0] + ".txt")

        if i < len(images) * split_ratio:
            train_file.write(os.path.join(images_path, image) + " " + annotation + "\n")
        else:
            test_file.write(os.path.join(images_path, image) + " " + annotation + "\n")
