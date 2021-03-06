# Как научить нейронную сеть распознавать жестовые команды - первые шаги разработчика

## Открытый урок в рамках интенсива [«Архипелаг 20.35»](https://2035.university/arkhipelag-20-35/) ([10.11.2020, 13:00-15:00 MSC](https://xle.2035.university/Archipelago20.35/event/41097))

### Набор данных **gesture_set**

Заполните [специальную форму](https://forms.gle/S8oixqohuK2HzdtG9). К ней прикреплена ссылка на набор данных.

[Демо](https://youtu.be/zmCqylqOvXY) проекта.

### Тизер

**Целевая аудитория:** начинающие разработчики, имеющие минимальный опыт работы с Python и элементарное понимание принципов Машинного Обучения.

**Чему можно научиться:**

- Настроить рабочую среду, узнать о необходимых инструментах для начала работы;

- Изучить и визуализировать набор данных [gesture_set](https://forms.gle/S8oixqohuK2HzdtG9);

- Выполнить обработку данных при помощи инструментов [PyTorch](https://pytorch.org/): предварительная обработка, аугментация;

- Построить классификатор на базе state-of-the-art архитектуры сверточной нейронной сети [LeNet-5](http://yann.lecun.com/exdb/lenet/), произвести обучение и тестирование;

- Увеличить производительность классификатора с помощью простых модификаций базовой архитектуры и настройки гиперпараметров;

- Получить представление о прикладном направлении проекта: работа с видеоданными (детектирование, классификация, трекинг, визуализация);

- Обсудить идеи для усиления проекта и его дальнейшего развития.

**Предполагаемое время:** 30-40 минут

### Требования к программному обеспечению

- [Anaconda + Python 3.7](https://www.anaconda.com/products/individual)

- [PyTorch](https://pytorch.org/get-started/locally/)

- Для установки библиотеки moviepy : `pip install moviepy`

### Полезные ссылки

- [PyTorch at Tesla (Andrej Karpathy)](https://youtu.be/oBklltKXtDE) + [Tesla Autopilot is better than you think!](https://youtu.be/zRnSmw1i_DQ)

- [Using deep learning and PyTorch to power next gen aircraft at Caltech](https://youtu.be/se206WBk2dM)

- [Трансформации изображений в torchvision](https://pytorch.org/docs/stable/torchvision/transforms.html)

- [Как посчитать размерности слоев нейронной сети](https://deeplizard.com/learn/video/cin4YcGBh3Q)

- [Как посчитать размерность слоя после операции AvgPool2d](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html)

- [Введение в архитектуры нейронных сетей](https://habr.com/ru/company/oleg-bunin/blog/340184/)
