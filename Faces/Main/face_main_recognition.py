# подключаем библиотеку распознавания лиц
import face_recognition as fr

# загружаем в переменную фотографию
picture = fr.load_image_file("../Content/people.jpg")
print('картинка загружена в нейросеть!')

# Обнаруживаем на фотографии лица
faces = fr.face_locations(picture)
print('Лица обнаружены и сохранены в список!')

# Определяем сколько лиц нашлось
faces_num = len(faces)
print('Найдено столько лиц: ' + str(faces_num))

person_picture = fr.load_image_file("../Content/person.jpg")
chars_list = fr.face_landmarks(person_picture)
print('Характеристики Хаара рассчитаны!')

# Сравнение двух лиц
# Загружаем сравниваемые фотографии
known_image = fr.load_image_file("../Content/dgobs1.jpg")
unknown_image = fr.load_image_file("../Content/dgobs2.jpg")

# Получаем лицевые кодировки
known_encodings = fr.face_encodings(known_image)[0]
unknown_encodings = fr.face_encodins(unkhown_image)[0]

# Сравниваем характеристики лицевых кодировок
result = fr.compare_faces(known_encodings, unknown_encodings)
if result:
    print('Да, это один и тот же человек!')
