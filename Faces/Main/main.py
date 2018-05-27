import cv2 # - Open CV
import numpy # -NunPy

# Cascade to be classified by algorithm
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0) # Inable camera, 0 means base parameters

# основной алгоритм
while True:
    resolution, img = cam.read() # считываем данные с камеры: разрешение и изображение
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # переводим изображние в градации серого

    faces = face_detect.detectMultiScale(gray, 1.3, 5) # указываем в скобках минимальное и максимальное разрешение видеокамеры
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2) #  в скобках задано: картика, координаты левого верхнего угла, кооординтаы правого нижнего угла, цвет, количество точек опоры
    cv2.imshow("Face", img)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release() # Отключить камеру
cv2.destroyAllWindows() # закрыть все окошки
