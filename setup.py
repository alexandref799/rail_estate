from setuptools import find_packages
from setuptools import setup
# Import du module 'path' pour une gestion des chemins plus robuste
from pathlib import Path

# 1. Utilisation de Pathlib pour une meilleure gestion du chemin (plus moderne)
# Note: Le contenu de requirements.txt est défini dans votre fichier de configuration
# C'est une bonne pratique de toujours utiliser la même méthode de lecture.

# La méthode de lecture que vous avez est correcte, mais on peut la moderniser légèrement
# en utilisant Path.

BASE_DIR = Path(__file__).parent
with open(BASE_DIR / "requirements.txt") as f:
    content = f.readlines()
# Filtrage pour ignorer les lignes vides, les commentaires (#) et les liens git+
requirements = [
    x.strip()
    for x in content
    if x.strip() and not x.startswith("#") and "git+" not in x
]

# Optionnel : Lire le README.md pour une description longue
try:
    long_description = (BASE_DIR / "README.md").read_text()
except FileNotFoundError:
    long_description = "Modèle prédictif immobilier axé sur l'impact ferroviaire."


setup(
    name='projet_rail_estate',
    version="0.1.0",
    description="Modèle prédictif immobilier axé sur l'impact ferroviaire.",
    # Ajout d'une description longue pour PyPI
    long_description=long_description,
    long_description_content_type='text/markdown', # Spécifie le format Markdown

    license="MIT",
    author="Rail-Estate",
    author_email="alexandref799@gmail.com",
    # Décommenter si l'URL devient publique
    # url="https://github.com/alexandref799/rail_estate",

    # 2. Ajout des classifieurs (aide PyPI à catégoriser votre projet)
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],

    install_requires=requirements,

    # find_packages() cherche tous les dossiers contenant un fichier __init__.py
    packages=find_packages(),

    test_suite="tests",

    # Conserver l'inclusion des données si vous avez des fichiers non-Python (ex: .csv, .json)
    include_package_data=True,

    # zip_safe doit être True si vous n'avez pas de données à charger au moment de l'exécution
    # Si vous n'êtes pas certain, le laisser à False est plus sûr pour les ML projects
    zip_safe=False
)
