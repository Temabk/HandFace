import pytest
import numpy as np
from unittest.mock import patch
from main import process_frame, main, detect_faces_from_image, detect_faces_from_webcam
from unittest import mock
import builtins
from tkinter import simpledialog
import os
import cv2


def test_process_frame_positive():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    face_locations = []
    known_face_location = [(100, 200, 150, 250)]
    with patch('face_recognition.face_locations', return_value=known_face_location):
        process_frame(frame, face_locations)
    assert len(face_locations) == 1


def test_process_frame_negative():
    frame = None
    face_locations = []
    with pytest.raises(TypeError):
        process_frame(frame, face_locations)


def test_detect_faces_from_webcam_positive():
    try:
        video_capture = cv2.VideoCapture(0)
        assert video_capture.isOpened(), "Не удалось открыть веб-камеру"
    finally:
        if video_capture:
            video_capture.release()


def test_detect_faces_from_webcam_negative():
    with mock.patch('cv2.VideoCapture') as mock_video_capture:
        if mock_video_capture.return_value.isOpened.return_value is False:
            with pytest.raises(ValueError):
                detect_faces_from_webcam()


def test_detect_faces_from_image_positive():
    test_image_path = "image_test.jpg"
    assert os.path.exists(test_image_path), f"Файл {test_image_path} не найден."


def test_detect_faces_from_image_negative():
    with mock.patch('face_recognition.load_image_file') as mock_load_image:
        mock_load_image.side_effect = FileNotFoundError("Файл изображения не найден")
        with mock.patch('builtins.print') as mock_print:
            detect_faces_from_image('non_existent_image.jpg')
            mock_print.assert_called_once_with("Ошибка загрузки изображения: Файл изображения не найден")


def test_main_negative():
    choice = '5'
    with pytest.raises(ValueError):
        main()


def test_main_positive():
    with mock.patch.object(simpledialog, 'askstring', return_value='3'):
        with mock.patch('tkinter.messagebox.showinfo') as mock_showinfo:
            main()
            mock_showinfo.assert_called_once_with("Выход", "Выход из программы.")
