import os
import json
import zlib
import xml.etree.ElementTree as ET
from graphviz import Digraph
from plantuml import PlantUML


def parse_object(object_hash, description=None):
    """
    Извлечь информацию из git-объекта по его хэшу.
    Каждый объект после разжатия выглядит так:
    ┌────────────────────────────────────────────────────────┐
    │ {тип объекта} {размер объекта}\x00{содержимое объекта} │
    └────────────────────────────────────────────────────────┘
    Содержимое объекта имеет разную структуру в зависимости от типа
    """

    # Полный путь к объекту по его хэшу
    object_path = os.path.join(config['repository_path'], '.git', 'objects', object_hash[:2], object_hash[2:])

    # Открываем git-объект
    with open(object_path, 'rb') as file:
        # Разжали объект, получили его сырое содержимое
        raw_object_content = zlib.decompress(file.read())
        # Разделили содержимое объекта на заголовок и основную часть
        header, raw_object_body = raw_object_content.split(b'\x00', maxsplit=1)
        # Извлекли из заголовка информацию о типе объекта и его размере
        object_type, content_size = header.decode().split(' ')

        # Словарь с данными git-объекта:
        # {
        #   'label': текстовая метка, которая будет отображаться на графе
        #   'children': список из детей этого узла (зависимых объектов)
        # }
        object_dict = {}

        # В зависимости от типа объекта используем разные функции для его разбора
        if object_type == 'commit':
            object_dict['label'] = r'[commit]\n' + object_hash[:6]
            object_dict['children'] = parse_commit(raw_object_body)

        elif object_type == 'tree':
            object_dict['label'] = r'[tree]\n' + object_hash[:6]
            object_dict['children'] = parse_tree(raw_object_body)

        elif object_type == 'blob':
            object_dict['label'] = r'[blob]\n' + object_hash[:6]
            object_dict['children'] = []

        # Добавляем дополнительную информацию, если она была
        if description is not None:
            object_dict['label'] += r'\n' + description

        return object_dict


def parse_tree(raw_content):
    """
    Парсим git-объект дерева, который состоит из следующих строк:
    ┌─────────────────────────────────────────────────────────────────┐
    │ {режим} {имя объекта}\x00{хэш объекта в байтовом представлении} │
    │ {режим} {имя объекта}\x00{хэш объекта в байтовом представлении} │
    │ ...                                                             │
    │ {режим} {имя объекта}\x00{хэш объекта в байтовом представлении} │
    └─────────────────────────────────────────────────────────────────┘
    """

    # Дети дерева (соответствующие строкам объекта)
    children = []

    # Парсим данные, последовательно извлекая информацию из каждой строки
    rest = raw_content
    while rest:
        # Извлечение режима
        mode, rest = rest.split(b' ', maxsplit=1)
        # Извлечение имени объекта
        name, rest = rest.split(b'\x00', maxsplit=1)
        # Извлечение хэша объекта и его преобразование в 16ричный формат
        sha1, rest = rest[:20].hex(), rest[20:]
        # Добавляем потомка к списку детей
        children.append(parse_object(sha1, description=name.decode()))

    return children


def parse_commit(raw_content):
    """
    Парсим git-объект коммита, который состоит из следующих строк:
    ┌────────────────────────────────────────────────────────────────┐
    │ tree {хэш объекта дерева в 16ричном представлении}\n           │
    │ parent {хэш объекта коммита в 16ричном представлении}\n        │─╮
    │ parent {хэш объекта коммита в 16ричном представлении}\n        │ │
    │ ...                                                            │ ├─ родителей может быть 0 или несколько
    │ parent {хэш объекта коммита в 16ричном представлении}\n        │─╯ 
    │ author {имя} <{почта}> {дата в секундах} {временная зона}\n    │
    │ committer {имя} <{почта}> {дата в секундах} {временная зона}\n │
    │ \n                                                             │
    │ {сообщение коммита}                                            │
    └────────────────────────────────────────────────────────────────┘
    """

    # Переводим raw_content в кодировку UTF-8 (до этого он был последовательностью байтов)
    content = raw_content.decode()
    # Делим контент на строки
    content_lines = content.split('\n')

    # Словарь с содержимым коммита
    commit_data = {}

    # Извлекаем хэш объекта дерева, привязанного к коммиту
    commit_data['tree'] = content_lines[0].split()[1]
    content_lines = content_lines[1:]

    # Список родительских коммитов
    commit_data['parents'] = []
    # Парсим всех родителей, сколько бы их ни было
    while content_lines[0].startswith('parent'):
        commit_data['parents'].append(content_lines[0].split()[1])
        content_lines = content_lines[1:]

    # Извлекаем информацию об авторе и коммитере
    while content_lines[0].strip():
        key, *values = content_lines[0].split()
        commit_data[key] = ' '.join(values)
        content_lines = content_lines[1:]

    # Извлекаем сообщение к комиту
    commit_data['message'] = '\n'.join(content_lines[1:]).strip()
    
    # Возвращаем все зависимости объекта коммита (то есть его дерево и всех родителей)
    return [parse_object(commit_data['tree'])] + \
           [parse_object(parent) for parent in commit_data['parents']]


def get_last_commit():
    """Получить хэш для последнего коммита в ветке"""
    head_path = os.path.join(config['repository_path'], '.git', 'refs', 'heads', config['branch'])
    with open(head_path, 'r') as file:
        return file.read().strip()

# Достаем информацию из конфигурационного файла
def load_config_from_xml(config_file: str) -> dict:
    tree = ET.parse(config_file)
    root = tree.getroot()
    # Преобразуем XML-данные в словарь
    config = {
        "repository_path": root.find("repository_path").text,
        "graph_output_path": root.find("graph_output_path").text,
        "branch": root.find("branch").text
    }
    return config


def build_plantuml_graph(tree: dict) -> str:
    """
    Построить граф зависимостей в формате PlantUML на основе дерева.
    :param tree: Словарь, описывающий структуру дерева зависимостей Git-объектов.
    :return: Строка в формате PlantUML.
    """
    def traverse_node(node: dict, connections: list, nodes: set):
        """
        Рекурсивный обход дерева узлов.
        :param node: Узел дерева.
        :param connections: Список соединений между узлами.
        :param nodes: Множество всех узлов.
        """
        # Добавляем текущий узел
        nodes.add(f'"{node["label"]}"')

        # Обрабатываем детей узла
        for child in node['children']:
            connections.append(f'"{node["label"]}" --> "{child["label"]}"')
            traverse_node(child, connections, nodes)

    # Списки узлов и связей
    connections = []
    nodes = set()

    # Обход дерева начиная с корня
    traverse_node(tree, connections, nodes)

    # Формирование PlantUML
    plantuml_lines = ["@startuml"]
    plantuml_lines += list(nodes)  # Узлы
    plantuml_lines += connections  # Связи
    plantuml_lines.append("@enduml")

    return "\n".join(plantuml_lines)


def save_plantuml_graph(graph: str, output_file: str):
    """
    Сохранить граф в файл PlantUML.
    :param graph: Строка в формате PlantUML.
    :param output_file: Имя выходного файла.
    """
    with open(output_file, 'w') as file:
        file.write(graph)
    print(f"Граф сохранён в {output_file}")



config_file = "config.xml"
config = load_config_from_xml(config_file)
repo_path = config["repository_path"]
graph_output_path = config["graph_output_path"]
branch = config["branch"]
# Получение последнего коммита
last_commit_hash = get_last_commit()

# Построение дерева зависимостей
dependency_tree = parse_object(last_commit_hash)

# Построение графа в формате PlantUML
plantuml_graph = build_plantuml_graph(dependency_tree)

# Сохранение графа в файл
#output_file = "dependency_graph.puml"
#save_plantuml_graph(plantuml_graph, output_file)
save_plantuml_graph(plantuml_graph, graph_output_path)



'''def generate_dot(filename):
    """Создать DOT-файл для графа зависимостей"""

    def recursive_write(file, tree):
        """Рекурсивно перебрать все узлы дерева для построения связей графа"""
        label = tree['label']
        for child in tree['children']:
            # TODO: учитывать только уникальные связи, чтобы они не повторялись
            file.write(f'    "{label}" -> "{child["label"]}"\n')
            recursive_write(file, child)

    # Стартовая точка репозитория - последний коммит главной ветки
    last_commit = get_last_commit()
    # Строим дерево
    tree = parse_object(last_commit)
    # Описываем граф в DOT-нотации 
    with open(filename, 'w') as file:
        file.write('digraph G {\n')
        recursive_write(file, tree)
        file.write('}')
# Генерируем файл с DOT-нотацией графа зависимостей
generate_dot('graph.dot')'''

'''import os
import subprocess
from datetime import datetime
from typing import List, Tuple
import xml.etree.ElementTree as ET
from graphviz import Digraph
from plantuml import PlantUML

def load_config_from_xml(config_file: str) -> dict:
    """Load configuration from an XML file.
    Args:
        config_file (str): Path to the XML configuration file.
    Returns:
        dict: Parsed configuration data as a dictionary."""
    tree = ET.parse(config_file)
    root = tree.getroot()
    
    # Преобразуем XML-данные в словарь
    config = {
        "repository_path": root.find("repository_path").text,
        "graph_output_path": root.find("graph_output_path").text,
        "since_date": root.find("since_date").text
    }
    return config

def get_commits(repo_path: str, since_date: str) -> List[Tuple[str, str]]:
    """Get a list of commits from the repository since the given date.
    Args:
    repo_path (str): Path to the git repository.
    since_date (str): Date string in a format accepted by 'git log --since', e.g., '2023-01-01'.
    Returns:
    List[Tuple[str, str]]: List of tuples where each tuple contains the commit hash and the commit date.
    Raises:
    Exception: If the git command fails."""
    git_command = [
        "git",
        "-C",
        repo_path,
        "log",
        "--pretty=format:%H %ct",
        "--since",
        since_date,
    ]
    result = subprocess.run(git_command, stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise Exception(f"Error running git command: {result.stderr}")

    commits = result.stdout.splitlines()
    commit_data = [
        (
            c.split()[0],
            datetime.utcfromtimestamp(int(c.split()[1])).strftime("%Y-%m-%d %H:%M:%S"),
        )
        for c in commits]
    return commit_data[::-1]  # Reverse to have chronological order


def build_dependency_graph(commits: List[Tuple[str, str]]):
    dot = Digraph(comment="Git Commit Dependencies")
    # разбиваю данные на дату и сам коммит
    for i, (commit, date) in enumerate(commits):
        dot.node(str(i), f"Commit: {commit}\nDate: {date}")
        if i > 0:
            dot.edge(str(i - 1), str(i))  # Connect commits in chronological order
    return dot


def save_graph(graph: Digraph, output_file: str) -> None:
    """Save the dependency graph in PlantUML format and generate the diagram.
    Args:
    graph (Digraph): The dependency graph to be saved.
    output_file (str): Path to the output file without extension."""
    plantuml_code = "@startuml\n"

    # Сохраняем узлы
    node_names = {}  # Для хранения отображения узлов
    # Сохраняем связи
    print(graph.body)
    for line in graph.body:
        
        if 'Commit' in line:
            # Извлекаем имя узла из строки вида 'node "Commit: abc123"'
            node_name = line.split('"')[1]  # Имя узла
            node_id = line.split(' ')[-1][:-2]  # ID узла (например, i)
            node_names[node_id] = node_name
            #pl_code += f'node "{node_name}" as {node_id}\n'
            #plantuml_code += f'node "{node_name}" as {node_id}\n'
        if '->' in line:
            parts = line.split('->')
            tail = parts[0].strip()
            head = parts[1].strip()
            plantuml_code += f"{tail} --> {head} : {node_name}\n"
    plantuml_code += "@enduml"

    # Запись в файл
    with open(f"{output_file}.puml", "w") as f:
        f.write(plantuml_code)
    print(f"Graph saved in PlantUML format to {output_file}.puml")

    # Генерация диаграммы с использованием PlantUML

def main(config_file: str) -> None:
    """
    Main function to generate the commit dependency graph.

    Args:
        config_file (str): Path to the YAML configuration file.
    """
    config_file = "config.xml"
    config = load_config_from_xml(config_file)
    repo_path = config["repository_path"]
    graph_output_path = config["graph_output_path"]
    since_date = config["since_date"]
    if not os.path.exists(repo_path):
        print(f"Error: Repository path '{repo_path}' does not exist.")
        return

    commits = get_commits(repo_path, since_date)

    if not commits:
        print(f"No commits found since {since_date}")
        return

    graph = build_dependency_graph(commits)
    save_graph(graph, graph_output_path)


if __name__ == "__main__":
    config_file = "config.xml"
    main(config_file)
'''