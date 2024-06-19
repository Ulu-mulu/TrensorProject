from PIL import Image


def resize_image_with_coordinates(input_image_path, output_image_path, new_width, new_height, x, y):
    original_image = Image.open(input_image_path)
    original_width, original_height = original_image.size

    # Рассчитываем коэффициенты масштабирования для каждой оси
    width_ratio = new_width / original_width
    height_ratio = new_height / original_height

    # Находим минимальное значение коэффициента, чтобы сохранить пропорции изображения
    min_ratio = min(width_ratio, height_ratio)

    # Рассчитываем новые размеры и координаты
    new_width = int(original_width * min_ratio)
    new_height = int(original_height * min_ratio)
    new_x = int(x * min_ratio)
    new_y = int(y * min_ratio)

    # Масштабируем изображение
    resized_image = original_image.resize((new_width, new_height))

    # Сохраняем измененное изображение
    resized_image.save(output_image_path)

    return new_x, new_y, min_ratio


input_image_path = "ShatteredFrames/Me_2.jpg"
output_image_path = "output_image.jpg"
new_width = 368
new_height = 600
x = 1000
y = 1000

new_x, new_y, min_r = resize_image_with_coordinates(input_image_path, output_image_path, new_width, new_height, x, y)

print(f"Новые координаты изображения: x={new_x}, y={new_y} ratio={min_r}")
