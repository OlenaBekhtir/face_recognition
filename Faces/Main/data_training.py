import os
import cv2
import numpy as np
from PIL import Image
import random

# в переменную recognizer создаем новую нейросеть
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "F:\Python_files\Faces\Content" # Путь, где находится картинка для обучения
# создаем функцию для распределения семантических весов нейросети
def getImageID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)] # Обрабтываем картинки как файлы в список пути
    faces = [] # изображения с нарезанными лицами
    IDs = [] # Семантический вес список

    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L') # Конвертирует картинку, чтобы дважды показать ее нейросети
        faceNp = np.array(faceImg, 'uint8')
        ID = int(len(os.path.split(imagePath)[-1].split('.')[0]) - 4)
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("Training", faceNp) # Показываем пользователю, что идет тренировка по выбранным картинкам
        cv2.waitKey(1000) # Ожидание 10с между показом картинок
    return np.array(IDs), faces

ids, faces = getImageID(path)
recognizer.train(faces, ids)
recognizer.save('trainingData.yml') # Создаем фойл для сохранения результатов тренировки нашей нейросети
cv2.destroyAllWindows() # Закрываем все окна
