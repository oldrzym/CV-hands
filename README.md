# ML-project
ML junior contest

Это проект позволяет видеть в VR-мире свое собственное тело

pspnet big new - файл с тренировкой и сохранением модели "вырезания" рук и ног

binary H classifacitaor - модель для определения, есть ли руки и ноги на фото - чтобы первая модель не пыталась вырезать что-то там, где ничего не надо вырезать

CVThread - файлы с чтением потока видео с камеры

HandExtracrtor - файлы с тензорными операциями на c++ TensorFlow - обработка входящего с камеры мзображения моделями

HeThread - файлы с морфологическими операциями над маской и последующим применением маски к входящему изображению и трансляцией этого изображения в VR-мир в Unigine
