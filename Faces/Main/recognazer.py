import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Переменная, которая хранит в себе инструмент распознаваня лиц
#подлючаем вебкамеру, вначале в положении выключено
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create() # Создание (инициализация) новой нейросети
rec.read('trainingData.yml') # Обучение нейросети
id = 0 # Создаем переменную идентификатор, стартовый семантический вес
font = cv2.FONT_HERSHEY_COMPLEX_SMALL # Создаем шрифт, которым будем подписывать

while True:
    ret, img = cam.read() # Считываем с камеры (расрешение и саму картинку),деструктивное присваивание
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # ПЕРЕВОДИМ В ГРАДАЦИИ СЕРОГО
    faces = faceDetect.detectMultiScale(gray, 1.3, 5) # Нарезка лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2) # Отрисовываем прямоугольник, в котором находится лицо
        id, conf = rec.predict(gray[y:y+h, x:x+w])
        if id == 1:
            id = "Olena"
        cv2.putText(img, str(id), (x, y+h), font, 1, (255, 0, 0), 2) # Надпись будет передвигаться за лицом
    cv2.imshow("face", img)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
