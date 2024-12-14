from main import is_inside_small_window, generate_random_shapes, is_finger_over_shape, window_x1, window_x2, window_y1, window_y2
import pytest

def test_is_inside_small_window_positive():

    position = (100,100)
    expected_result = True

    assert is_inside_small_window(position)==expected_result

def test_is_inside_small_window_negative():
    position = ('aaa', 'aaa')
    with pytest.raises(TypeError):
        is_inside_small_window(position)


def test_is_finger_over_shape_positive():
    shape = {
        'position': (50, 50),
        'type': 'circle',
        'color': 'red'
    }

    assert is_finger_over_shape(50, 50, shape) is True  


def test_is_finger_over_shape_negative():
    shape = {
        'position': (50, 50),
        'type': 'circle',
        'color': 'red'
    }
    index_x = '65'
    index_y = '66'
    with pytest.raises(TypeError):
        is_finger_over_shape(index_x, index_y, shape)


def test_generate_random_shapes_positive():
    num_shapes = 5
    image_shape = (480, 640, 3)
    shapes = generate_random_shapes(num_shapes, image_shape)
    
    assert isinstance(shapes, list)
    assert len(shapes) == num_shapes

def test_generate_random_shapes_negative():
    num_shapes = '20'
    image_shape = (100, 100, 5)
    with pytest.raises(TypeError):
        generate_random_shapes(num_shapes, image_shape)