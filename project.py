import cv2
import mediapipe as mp
import numpy as np
from random import randint, choice
import time

import face_recognition
import tkinter as tk
from tkinter import simpledialog, messagebox
import os

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(max_num_hands=6)
draw = mp.solutions.drawing_utils
hands_count = 0

num_shapes = 5
shapes = []
shape_size = 50
colors = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (0, 0, 0)
]

start_time = time.time()
countdown_time = 60
counter = 0
game_over = False  

window_x1, window_y1 = 10, 10  
window_x2, window_y2 = 200, 200

def generate_random_shapes(num_shapes: int, image_shape: tuple) -> list:
    '''
    Генерация рандомных фигур

    :param num_shapes: число генерируемых фигур
    :type num_shapes: int
    :param image_shape: размер изображения
    :type image_shape: tuple
    :returns: возвращает список сгенерированных фигур
    :rtype: list
    :raises TypeError: Если `num_shapes` не является целым числом, или `image_shape` не является кортежем с тремя целыми числами
    '''
    shapes = []
    height, width, _ = image_shape
    if isinstance(num_shapes, int) and isinstance(image_shape, tuple):
        for _ in range(num_shapes):
            while True:
                x = randint(30, width - 30)  
                y = randint(30, height - 30)
                if not is_inside_small_window((x, y)):
                    shape_type = choice(['circle', 'square', 'triangle'])
                    color = choice(colors)  
                    shapes.append({'position': (x, y), 'type': shape_type, 'color': color})
                    break
        return shapes
    else:
        raise TypeError

def is_finger_over_shape(index_x: int, index_y: int, shape: dict) -> bool:
    '''
    Проверяет попадает ли палец на фигуру

    :param index_x: горизантальная координата кончика указательного пальца
    :type index_x: int
    :param index_y: вертикальная координата кончика указательного пальца
    :type index_y: int
    :param shape: фигура с которой взаимодействует палец
    :type shape: dict
    :returns: возвращает True или False, в зависимости от результата
    :rtype: bool
    :raises TypeError: Если значения `index_x` или `index_y` не являются целыми числами, или `shape` не имеет ключа `position` с координатами
    '''
    if isinstance(index_x, int) and isinstance(index_y, int):
        return (shape['position'][0] - 15 <= index_x <= shape['position'][0] + 15) and \
               (shape['position'][1] - 15 <= index_y <= shape['position'][1] + 15)
    else:
        raise TypeError  
    
def is_inside_small_window(position: tuple) -> bool:
    '''
    Проверяет попадает ли сгенерированная фигура на область маленького окна

    :param position: положение фигуры на экране
    :type position: tuple
    :returns: возвращает True или False, в зависимости от результата
    :rtype: bool
    :raises TypeError: Если значения x и y не являются целыми числами
    '''
    x, y = position
    if isinstance(x, int) and isinstance(y, int):
        return window_x1 < x < window_x2 and window_y1 < y < window_y2
    else:
        raise TypeError

shapes = generate_random_shapes(num_shapes, (480, 640, 3))
small_window_shape = choice(shapes)

def detect_hands_from_webcam():
    """
    Обнаруживает руки с веб-камеры в реальном времени.

    :raises ValueError: Если не удается открыть веб-камеру.
    """
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise ValueError
    
    global counter, game_over, shapes, small_window_shape, start_time
    while True:

        elapsed_time = time.time() - start_time  # Время с начала
        remaining_time = countdown_time - elapsed_time  # Оставшееся время

        if remaining_time < 0:
            remaining_time = 0
            game_over = True  # Устанавливаем флаг завершения игры

        if cv2.waitKey(1) & 0xFF == 27:
            break

        success, image = cap.read()
        if not success:
            print("Не удалось получить кадр из камеры.")
            continue

        image = cv2.flip(image, +1)
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                draw.draw_landmarks(image, handLms, mp.solutions.hands.HAND_CONNECTIONS)

                index_finger_tip = handLms.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                index_x = int(index_finger_tip.x * image.shape[1])
                index_y = int(index_finger_tip.y * image.shape[0])

                for shape in shapes:
                    if is_finger_over_shape(index_x, index_y, shape):
                        # Проверяем совпадение цвета
                        if shape['color'] == small_window_shape['color']:
                            counter += 1  # Увеличиваем счетчик
                        else:
                            counter -= 1  # Уменьшаем счетчик
                        
                        # Генерация новых фигур
                        shapes = generate_random_shapes(num_shapes, image.shape)
                        small_window_shape = choice(shapes)
                        break  # Выходим из цикла обработки фигур

        for shape in shapes:
            if shape['type'] == 'circle':
                cv2.circle(image, shape['position'], shape_size, shape['color'], -1)
            elif shape['type'] == 'square':
                cv2.rectangle(image, (shape['position'][0] - shape_size, shape['position'][1] - shape_size),
                              (shape['position'][0] + shape_size, shape['position'][1] + shape_size), shape['color'], -1)
            elif shape['type'] == 'triangle':
                pts = [shape['position'],
                       (shape['position'][0] - shape_size, shape['position'][1] + shape_size),
                       (shape['position'][0] + shape_size, shape['position'][1] + shape_size)]
                cv2.fillPoly(image, [np.array(pts)], shape['color'])

        cv2.rectangle(image, (window_x1, window_y1), (window_x2, window_y2), (200, 200, 200), 2)

        if small_window_shape['type'] == 'circle':
            cv2.circle(image, (window_x1 + 90, window_y1 + 90), 30, small_window_shape['color'], -1)
        elif small_window_shape['type'] == 'square':
            cv2.rectangle(image, (window_x1 + 60, window_y1 + 60), (window_x1 + 120, window_y1 + 120), small_window_shape['color'], -1)
        elif small_window_shape['type'] == 'triangle':
            pts = [(window_x1 + 90, window_y1 + 50),
                   (window_x1 + 60, window_y1 + 130),
                   (window_x1 + 120, window_y1 + 130)]
            cv2.fillPoly(image, [np.array(pts)], small_window_shape['color'])

        # Отображение таймера
        timer_text = f"Time left: {int(remaining_time)} seconds"
        cv2.putText(image, timer_text, (220, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        counter_text = f"Your Score: {counter}"
        cv2.putText(image, counter_text, (220, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Проверяем, завершена ли игра
        if game_over:
            cv2.putText(image, "Time is Over!", (700, 650), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("Our Camera with recognized hands", image)
    cv2.destroyAllWindows()


def process_frame(frame, face_locations):
    """
    Обрабатывает один кадр, распознает лица и обновляет список face_locations.

    :param frame: Кадр видео (или изображение), в котором нужно найти лица.
    :type frame: numpy.ndarray
    :param face_locations: Список для хранения координат найденных лиц.
    :type face_locations: list
    :raises TypeError: Если кадр передан как None
    """
    if frame is None:
        raise TypeError("Кадр не может быть None")

    rgb_frame = frame[:, :, ::-1]
    detected_faces = face_recognition.face_locations(rgb_frame)
    face_locations.clear()
    face_locations.extend(detected_faces)


def detect_faces_from_webcam():
    """
    Обнаруживает лица с веб-камеры в реальном времени.

    :raises ValueError: Если не удается открыть веб-камеру.
    """
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Не удалось открыть веб-камеру.")
        raise ValueError("Не удалось открыть веб-камеру")

    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_skip = 3
    frame_count = 0
    face_locations = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Не удалось получить кадр с веб-камеры.")
            break

        if frame_count % frame_skip == 0:
            process_frame(frame, face_locations)

        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        number_of_faces = len(face_locations)
        text = f"Faces: {number_of_faces}"
        font_scale = 1.5
        font_thickness = 3
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = 10
        text_y = 30 + text_size[1]

        cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10),
                      (text_x + text_size[0] + 10, text_y + 10),
                      (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thickness)

        cv2.imshow('Webcam Face Detection', frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()


def detect_faces_from_image(image_path):
    """
    Обнаруживает лица на изображении.

    :param image_path: Путь к изображению, на котором нужно обнаружить лица.
    :type image_path: str
    :raises Exception: Если изображение не удается загрузить.
    """
    try:
        image = face_recognition.load_image_file(image_path)
    except Exception as e:
        print(f"Ошибка загрузки изображения: {e}")
        return

    rgb_image = image[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_image)
    image_with_boxes = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_location in face_locations:
        top, right, bottom, left = face_location
        cv2.rectangle(image_with_boxes, (left, top), (right, bottom), (0, 255, 0), 2)

    number_of_faces = len(face_locations)
    text = f"faces: {number_of_faces}"
    font_scale = 1.5
    font_thickness = 3
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
    text_x = 10
    text_y = 30 + text_size[1]

    cv2.rectangle(image_with_boxes, (text_x - 10, text_y - text_size[1] - 10),
                  (text_x + text_size[0] + 10, text_y + 10),
                  (0, 0, 0), -1)

    cv2.putText(
        image_with_boxes,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        font_thickness
    )

    print(f"Количество найденных лиц: {number_of_faces}")
    cv2.imshow("Image Face Detection", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """
    Главная функция, которая запускает приложение и позволяет выбрать режим работы программы.

    :raises ValueError: Если сделан неверный выбор в меню.
    """
    root = tk.Tk()
    root.withdraw()

    default_image_path = "image_test.jpg"  # Имя файла изображения в папке с кодом
    if not os.path.exists(default_image_path):
        print(f"Файл {default_image_path} не найден в папке с кодом.")
        messagebox.showerror("Ошибка", f"Файл {default_image_path} не найден в папке с кодом.")
        return

    while True:
        choice = simpledialog.askstring(
            "Выбор режима",
            "Выберите режим нейросети:\n1 - Обнаружение лиц с веб-камеры\n2 - Обнаружение лиц на изображении\n3 - Распознавание рук\n4 - Выход",
        )

        if choice == '1':
            print("Запуск обнаружения лиц с веб-камеры...")
            detect_faces_from_webcam()

        elif choice == '2':
            print(f"Использование изображения: {default_image_path}")
            detect_faces_from_image(default_image_path)

        elif choice == '3':
            print("Запуск распознавания рук...")
            detect_hands_from_webcam()

        elif choice == '4':
            print("Выход из программы.")
            messagebox.showinfo("Выход", "Выход из программы.")
            break

        else:
            print("Неверный ввод. Попробуйте снова.")
            messagebox.showerror("Ошибка", "Неверный ввод, попробуйте снова.")
            raise ValueError

if __name__ == "__main__":
    main()