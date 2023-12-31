# HSE - Linear Regression homework

### Результаты работы
В ходе работы была обучена модель регрессии предсказания стоимости автомобилей, а также реализован веб-сервис, использующий обученную модель для предсказания стоимости на новых входных данных в форматах csv- или json-файла.

В качестве основной метрики качества был выбран коэффициент детерминации (R<sup>2</sup>). Также по требованию заказчика модель оценивалась с помощью кастомной метрики (M) - доля предсказаний, отклонившихся от истинного значения не более, чем на 10% в обе стороны.

Полученные наилучшие значения метрик:
- R<sup>2</sup> - 0.836
- M - 0.272

### Выводы
Наибольший вклад в улучшение предсказания целевой переменной внесло логарифмирование целевой переменной, которое заштрафовало тяжелые хвосты распределения и привело распределение к похожему на нормальное. Работа с признаками: добавление нового на основе производителя, другая агрегация категориальных признаков, замена числового признака на его квадрат и т д - внесла меньший вклад, но в совокупности все эти преобразования дали еще более точный результат предсказания.

В ходе работы не вышло довести до ума разделение признака torque на 2 отдельных, возможно, там скрывался какой-то инсайт. В целом не удалось достаточно поработать над признаками, кажется, можно было вытащить еще что-то полезное. Точно стоило бы проверить полиномиальные признаки, заполнение пропущенных значений не медианой, проанализировать распределения каждого признака.

### Сервис
Ниже представлены результаты работы сервиса
- эндпоинт /predict_items_via_csv
  
  Ответ сервиса на загруженный csv-файл. В результате возвращается csv-файл с добавленным столбцом с предсказанием модели
![image](https://github.com/yuri-pavar/LR_hw_ml/assets/33356873/296ee9c2-1746-486b-a420-5a1c48d6b7e3)

- эндпоинт /predict_item

  Ответ сервиса на загруженный json-файл с признаками одного объекта. В результате возвращается предсказание модели по данному объекту
![image](https://github.com/yuri-pavar/LR_hw_ml/assets/33356873/07d7c25c-d1c7-4b09-8802-e4af04be0e1b)

- эндпоинт /predict_item

  Ответ сервиса на загруженный json-файл с признаками нескольких объектов. В результате возвращается предсказанная модели по каждому объекту
![image](https://github.com/yuri-pavar/LR_hw_ml/assets/33356873/0fee12d4-25ca-4140-a1d7-369013aa0710)

