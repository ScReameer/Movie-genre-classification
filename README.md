# <center>Классификация фильмов по жанрам
В данном проекте стояла задача обучить *ML*-модель, которая **классифицирует описания фильмов на английском языке по жанрам**. Всего в данных было **27 уникальных жанров**

**Результат** (инференс модели на облачном сервисе https://streamlit.io/) можно протестировать [<b>здесь</b>](https://movie-genre-classification-screameer.streamlit.app/)

**Ноутбук** со всеми этапами проекта можно найти [<b>тут</b>](./notebook/Movie%20genre%20classification.ipynb)

**Датасет** был взят с [<b>соревнования на <i>Kaggle</i></b>](https://www.kaggle.com/competitions/sf-dl-movie-genre-classification)

Простые *ML*-модели не показали удовлетворительных результатов, поэтому конечной моделью стала дообученная (<b><i>Fine-tuning</i></b>) нейросеть *[<b>BERT (Bidirectional Encoder Representations from Transformers)</b>](https://www.kaggle.com/models/tensorflow/bert)*.

## Архитектура использованной нейросети:
1. **Первый слой** `text_input` - текст, который необходимо подать в модель

2. **Второй слой** `preprocessor` - часть *BERT*, **векторизация текста**

3. **Четвертый слой** `encoder` - предобученная нейросеть *BERT*. На выходе нас интересует только `pooled_output` - **768-мерное представление входных данных**.  
    **Количество обучаемых параметров** составило $\approx 10^8$

4. **Пятый слой** `dropout_layer` - **регуляризация** 30% нейронов

5. **Шестой слой** `output_layer` - **выходной слой из 27 нейронов** (количество классов) с активацией `softmax` для мультиклассовой классификации.

6. В качестве **оптимизатора** был выбран `AdamW` - модификация `Adam`. Именно этот оптимизатор использовался на обучении оригинального *BERT*. 

7. **Темп обучения** `learning_rate` был выбран очень низким ($10^{-6}$). 

8. <b>Дообучение (*Fine-tuning*)</b> было произведено на 15 эпохах, из которых **самой эффективной** (по метрике на валидационной выборке) **стала 14 эпоха**, именно она используется как финальная модель в этом проекте
9. **Функция потерь** - $\text{LogLoss}$
10. **Метрики**:
    * **Точность** (<b><i>Accuracy</i></b> для всего датасета, а не *Precision*). Именно эта метрика используется как **целевая** в соревновании на *Kaggle*

    * $F_1$<b>-мера</b> - среднее гармоническое для *Precision* и *Recall*

## Использование готовой модели в *Python*
### Зависимости:
1. <b><i>Python 11.x</i></b>

2. Установить [<b>необходимые версии библиотек</b>](https://github.com/ScReameer/Movie-genre-classification/blob/main/requirements.txt)

3. [<b>Скачать</b>](https://drive.google.com/drive/folders/1qpWe3tq9HEpmBQK4-ke-86xlICm9wd30?usp=drive_link) модель и энкодер:

    * `bert_tuned.h5` - предобученная модель *BERT*.  
        Для того, чтобы корректно загрузить, необходимо выполнить следующий код:
        ```py
        import keras
        import tensorflow as tf
        import tensorflow_text
        import tensorflow_hub as hub
        import pickle
        # Путь к файлу к моделью
        model_path = r'%YOUR_PATH%/bert_tuned.h5'
        # Загрузка модели из файла
        model = keras.models.load_model(
            model_path, 
            custom_objects=dict(KerasLayer=hub.KerasLayer)
        )
        ```

    * `label_encoder.pkl` - энкодер `LabelEncoder` из библиотеки *scikit-learn*, хранит в себе правильный порядок классов после предсказания и взятия $\text{arg max}$. Пример использования:  
        ```py
        # Загрузка энкодера из файла
        with open(r'%YOUR_PATH%/label_encoder.pkl', 'rb') as encoder_file:
        label_encoder = pickle.load(encoder_file)
        # lambda-функция для получения предсказания
        get_prediction = lambda desc: label_encoder.inverse_transform(model.predict([desc], verbose=0).argmax(axis=1))[0]
        # Название фильма на английском языке
        desc = '%ANY%'
        # Получение предсказания в виде названия жанра
        prediction = get_prediction(desc)
        ```