import socketio
import os
import cv2
import numpy as np
from cv2 import VideoCapture
from ultralytics import YOLO, SAM
from ultralytics.utils.plotting import Annotator
import Createdataset
from shapely.geometry import Point, Polygon

# Connect to server
sio_client = socketio.Client()

app = socketio.ASGIApp(sio_client)

sio_client.connect("http://localhost:8080")

current_id = -1

drop_stream = False

# sio_client.emit('get-message-ai', {'start': True})

typical_folder = 'testdir'


def format_array_to_tuples(input_array):
    tuple_array = [(input_array[i], input_array[i + 1]) for i in range(0, len(input_array), 2)]
    return tuple_array


def search_model():
    global typical_folder
    path = os.getcwd()

    # Получаем список всех папок в директории
    folders = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]

    # Получаем список папок с одинаковой частью имени
    same_name_folders = []

    for folder in folders:
        if typical_folder in folder:
            same_name_folders.append(folder)

    for folder in same_name_folders:
        path_to_model = f'{folder}/runs/detect/train/weights/best.pt'
        if os.path.exists(path_to_model):
            return path_to_model

    return False


def is_point_inside_polygon(point, polygons):
    for polygon in polygons:
        print(point)
        # print(polygon['polygon'])
        if polygon['polygon'].contains(point):
            print(f'Hit in {polygon["class_id"]} polygon')
            return polygon['class_id']
    return -1


def stream_connect():
    rtsp_username = "admin"
    rtsp_password = "Qwerty1234"

    rtsp: str = "rtsp://" + rtsp_username + ":" + rtsp_password + "@10.200.1.51:554/Streaming/channels/" + "1" + "01"

    cap: VideoCapture = cv2.VideoCapture()
    cap.open(rtsp)

    cap.set(3, 1920)  # ID number for width is 3
    cap.set(4, 1080)  # ID number for height is 480
    cap.set(10, 100)  # ID number for brightness is 10qq

    return cap


@sio_client.on('post-coord-ai')
def on_coords(data):

    global drop_stream

    cap = stream_connect()

    det_model = YOLO('yolov8x.pt')
    sam_model = SAM('sam_l.pt')

    datalist = []

    ret, frame = cap.read()

    if ret:

        img = frame

        cv2.imwrite('OnClickImg.jpg', img)

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

    point = Point(data[0], data[1])
    global current_id
    current_id = is_point_inside_polygon(point, datalist)

    iterate = 15

    if current_id != -1:

        while not drop_stream:

            ret, frame = cap.read()

            if ret:

                if iterate == 15:

                    print(current_id)
                    print('detection tick')

                    img = frame

                    det_results = det_model(img, stream=True, device='0', classes=[current_id])

                    for result in det_results:
                        class_ids = result.boxes.cls.int().tolist()  # noqa
                        if len(class_ids):
                            boxes = result.boxes.xyxy  # Boxes object for bbox outputs
                            sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False,
                                                    device='0')
                            segments = sam_results[0].masks.xy  # noqa

                            datalist_ids = []

                            for i in range(len(segments)):
                                s = segments[i]
                                if len(s) == 0:
                                    continue
                                segment = (segments[i].reshape(-1).tolist())
                                segment = np.array(segment).astype(np.int64, copy=False).tolist()
                                datalist_ids.append(segment)
                                iterate = 0

                            send_array(datalist_ids)

                else:
                    iterate = iterate + 1

            else:
                break

    else:
        sio_client.emit('get-metric-ai', 'Object undefined')

    cap.release()
    cv2.destroyAllWindows()
    pass


def send_array(array_of_coordinates):
    # Отсылаем координаты тут
    sio_client.emit('get-coords-ai', array_of_coordinates)


def initialize_new_model(folder_name, model_path):

    if model_path == '':
        existing_model = search_model()
        if not existing_model:
            sio_client.emit('get-metric-ai', 'no existing model')
            return

        else:
            new_model = YOLO(existing_model)  
            new_model.to('cuda')

    else:
        new_model = YOLO(model_path) 
        new_model.to('cuda')

    print('Finish model initialize')

    cap = stream_connect()

    iterate = 250

    saved_status = 'initialized'

    global drop_stream

    while not drop_stream:

        ret, frame = cap.read()

        if ret:

            if iterate == 300:

                status = 'no detections'

                sliced_image = frame

                # Write the frame into the model
                results = new_model.predict(sliced_image, verbose=False)
                for r in results:
                    # print(r.masks)
                    boxes = r.boxes
                    for box in boxes:
                        c = box.cls
                        if box.conf < 0.8:
                            continue
                        status = (new_model.names[int(c)])
                        print('Object class: ', (new_model.names[int(c)]))
                        print('Probability: ', box.conf)

                if status != saved_status:
                    saved_status = status
                    sio_client.emit('get-metric-ai', status)
                    iterate = 0
                else:
                    iterate = 0

            iterate = iterate + 1

        # Break the loop
        else:
            break
        cap.release()
        cv2.destroyAllWindows()

    else:
        cap.release()
        cv2.destroyAllWindows()


@sio_client.on('post-message-ai')
def start_record(data):

    global drop_stream

    drop_stream = True

    global current_id

    if current_id != -1:

        cap = stream_connect()
        print('Start recording')
        output = cv2.VideoWriter("RawVideos/Test3.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 50, (1920, 1080))
        sio_client.emit('get-message-ai', {'recording': True})
        WrittenFrame = 0

        while True:

            ret, frame = cap.read()

            if ret:

                if WrittenFrame != 100:

                    # Write the frame into the file 'output.avi'
                    output.write(frame)
                    WrittenFrame = WrittenFrame + 1

                else:
                    break

            # Break the loop
            else:
                break
        print('Finish recording')

        cap.release()
        cv2.destroyAllWindows()

        sio_client.emit('get-message-ai', {'recording': False})

        print('Start creating dataset')

        global typical_folder

        main_folder_name = Createdataset.folder_search(typical_folder)
        path_to_video = "RawVideos/Test.avi"
        Createdataset.frame_shatter(path_to_video, main_folder_name)

        sio_client.emit('get-message-ai', {'dataset': True})

        Createdataset.split_dataset(main_folder_name)
        Createdataset.create_yaml(main_folder_name)

        folder_list = ['/train', '/val', '/test']

        for folder in folder_list:
            current_dir = main_folder_name + folder
            Createdataset.auto_annotate_dark(data=current_dir + '/images', det_model='yolov8x.pt', sam_model='sam_l.pt',
                                             device='0',
                                             output_dir=current_dir + '/labels', current_id=current_id)

        print('Finish creating dataset')

        sio_client.emit('get-message-ai', {'dataset': False})
        sio_client.emit('get-message-ai', {'learning': True})

        Createdataset.train_on_dataset(main_folder_name)

        sio_client.emit('get-message-ai', {'learning': False})

        print('Model initialize')

        drop_stream = False

        current_id = -1

        model_path = f'{main_folder_name}/runs/detect/train/weights/best.pt'

        initialize_new_model(main_folder_name, model_path=model_path)

    else:

        print('current_id undefined')

        drop_stream = False

        current_id = -1

        sio_client.emit('get-message-ai', {'recording': False})


# initialize_new_model(typical_folder, model_path='')
