import os
import numpy as np
import cv2
from shapely.geometry import Point, Polygon
import os
import cv2
import numpy as np
from ultralytics import YOLO, SAM
from ultralytics.utils.plotting import Annotator


def format_array_to_tuples(input_array):
    tuple_array = [(input_array[i], input_array[i+1]) for i in range(0, len(input_array), 2)]
    return tuple_array


def is_point_inside_polygon(point, polygons):
    for polygon in polygons:
        #print(point)
        #print(polygon['polygon'])
        if polygon['polygon'].contains(point):
            return polygon['class_id']
    return False


det_model = YOLO('yolov8x.pt')
sam_model = SAM('sam_l.pt')

datalist = []

# Загрузка изображения с полигонами и определение координат точки
img = cv2.imread('ShatteredFrames/Me_0.jpg')
point = Point(502, 414)  # Координаты точки

det_results = det_model(img, stream=True, device='0')

for result in det_results:
    class_ids = result.boxes.cls.int().tolist()  # noqa
    if len(class_ids):
        boxes = result.boxes.xyxy  # Boxes object for bbox outputs
        sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device='0')
        segments = sam_results[0].masks.xy  # noqa

        for i in range(len(segments)):
            s = segments[i]
            if len(s) == 0:
                continue
            segment = (segments[i].reshape(-1).tolist())
            segment = np.array(segment).astype(np.int64, copy=False)
            segment = format_array_to_tuples(segment)
            segment = Polygon(segment)
            dataframe = {'class_id': class_ids[i], 'polygon': segment}
            datalist.append(dataframe)
            # (f"{class_ids[i]} " + " ".join(segment) + "\n")


# Создание полигонов на изображении (для примера создается два полигона)
# polygon1 = Polygon([(50, 50), (50, 100), (100, 100), (100, 50)])
# polygon2 = Polygon([(150, 150), (150, 200), (200, 200), (200, 150)])

# Функция для определения попадания точки в полигон

# polygons = [polygon1, polygon2]

# Проверка попадания точки в полигон
contained = is_point_inside_polygon(point, datalist)
print(contained)

# Отображение изображения с нарисованными полигонами (опционально)
for polygon in datalist:
    points = list(polygon['polygon'].exterior.coords)
    for i in range(len(points)-1):
        cv2.line(img, (int(points[i][0]), int(points[i][1])), (int(points[i+1][0]), int(points[i+1][1])), (255, 0, 0), 2)

cv2.imshow('Polygons', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
