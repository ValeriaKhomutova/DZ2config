# Конфигурационное управление дз №2
## Общее описание
Разработать инструмент командной строки для визуализации графа 
зависимостей, включая транзитивные зависимости. Сторонние средства для 
получения зависимостей использовать нельзя. 

Зависимости определяются для git-репозитория. Для описания графа 
зависимостей используется представление PlantUML. Визуализатор должен 
выводить результат на экран в виде кода. 
Построить граф зависимостей для коммитов, в узлах которого находятся 
связи с файлами и папками, представленными уникальными узлами. Граф 
необходимо строить для ветки с заданным именем. 
Конфигурационный файл имеет формат xml и содержит: 
- Путь к программе для визуализации графов. 
- Путь к анализируемому репозиторию. 
- Путь к файлу-результату в виде кода. 
- Имя ветки в репозитории. 
Все функции визуализатора зависимостей должны быть покрыты тестами. 
##  Описание всех функций и настроек
1. git.puml - файл с описанием графа зависимостей в виде кода
2. visualize_commits.py - реализация, функции получения коммитов с репозитория на git и построения графа зависимостей

##  Описание команд для сборки проекта.

1. Запуск скрипта для демонстрации задания

```python visualize_commits.py```


## Примеры использования
![Screen](https://github.com/ValeriaKhomutova/DZ2config/blob/main/image.png)

