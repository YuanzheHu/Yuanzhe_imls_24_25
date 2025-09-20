import os
from keras.utils import img_to_array, load_img
import cv2 # add opencv-python to the requirements.txt file
import numpy as np
import dlib

# PATH TO ALL IMAGES
global basedir, image_paths, target_size
basedir = './dataset'
images_dir = os.path.join(basedir,'celeba')
labels_filename = 'labels.csv'


def import_celeba_dataset():
    """
    Imports the CelebA dataset, loading images and their corresponding gender labels.
    This function reads image files from a specified directory and their corresponding
    gender labels from a labels file. It returns the images and labels as lists.

    Returns:
        tuple: A tuple containing two lists:
            - all_images (list): A list of loaded images.
            - all_labels (list): A list of gender labels corresponding to the images.

    Raises:
        FileNotFoundError: If the labels file or images directory does not exist.
        ValueError: If there is an issue with reading the labels or images.
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    gender_labels = {line.split(',')[0] : int(line.split(',')[6]) for line in lines[2:]}
    if os.path.isdir(images_dir):
        all_images = []
        all_labels = []
        for img_path in image_paths:
            file_name= img_path.split('.')[1].split('/')[-1]
            # file_name= img_path.split('.')[1].split('\\')[-1] # if using windows use this line
            img = img_to_array(
                load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            all_images.append(img)
            all_labels.append(gender_labels[file_name])
    return all_images, all_labels

def extract_features_from_images(images):
    """
    Extracts facial landmarks from images using dlib's facial landmark detector.
    This function extracts facial landmarks from images using dlib's facial landmark
    detector. It returns the extracted landmarks as a list of NumPy arrays.

    Args:
        images (list): A list of images

    Returns:
        list: A list of NumPy arrays containing the extracted
            landmarks for each image, or None if no landmarks were detected.
    """
    all_features = []
    for img in images:
        features, _ = run_dlib_shape(img)
        if features is not None:
            all_features.append(features)
        else:
            all_features.append(None)
    return all_features


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def shape_to_np(shape, dtype="int"):
    """ 
    Converts dlib shape object to NumPy array.
    This function converts the dlib shape object to a NumPy array. It returns the array.
    
    Args:
        shape (dlib.shape): A dlib shape object.
        dtype (str): The data type of the NumPy array.
        
    Returns:
        np.array: A NumPy array containing the (x, y) coordinates of the shape object.  
    """
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    """
    Converts dlib rectangle to bounding box (x, y, w, h).
    This function converts a dlib rectangle to a bounding box in the format (x, y, w, h).

    Args:
        rect (dlib.rectangle): A dlib rectangle object.

    Returns:
        tuple: A 4-tuple containing the (x, y, w, h) coordinates of the bounding box.
    """
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    """
    Detects facial landmarks in an image using dlib.
    This function loads an image, detects the landmarks of the face, and returns the image
    and the landmarks.

    Args:
        image (np.array): A NumPy array representing an image.

    Returns:
        tuple: A tuple containing two elements:
            - dlibout (np.array): A NumPy array containing the detected facial landmarks.
            - resized_image (np.array): A NumPy array representing the resized image.
    """
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale

    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])
    return dlibout, resized_image