"""
Script pour la génération d'un tirage aléatoire des projets pour les étudiants de la cours de Mathématiques pour informaticien.
Ce script utilise la bibliothèque pandas pour créer un DataFrame et exporter les résultats dans un fichier Excel.
"""
import sys
import subprocess

packages = ['pandas', 'openpyxl', ]

def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

for package in packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Le package '{package}' n'est pas installé. Installation en cours...")
        install(package)
        __import__(package)

import random
import pandas as pd
print("\nTous les packages nécessaires sont installés et chargés avec succès.")

# Pour rendre le tirage reproductible
random.seed(2026)

students = [
    {"nom": "Nathanaël Cadet", "email": "nathanaelcadet34@gmail.com"},
    {"nom": "jovens constant", "email": "constantjovene@gmail.com"},
    {"nom": "Phéni Grande Puissance", "email": "geveusp@yahoo.com"},
    {"nom": "Ruud Jean Gilles", "email": "ruudjg004@gmail.com"},
    {"nom": "Jackendy Christopher Jean-Baptiste", "email": "jackendychristopher@icloud.com"},
    {"nom": "Galette Leydch", "email": "galetteleydch@gmail.com"},
    {"nom": "Ruth-Zaëlle Louis", "email": "louisruth81@gmail.com"},
    {"nom": "Mimide Michel", "email": "mimide.michel@uniq.edu"},
    {"nom": "Shouppyna Nah Charles", "email": "shouppyn@gmail.com"},
    {"nom": "Sebastien Olivier", "email": "SebastienShyneider.olivier@uniq.edu"},
    {"nom": "Cesar Ricaddy Laurens", "email": "cesarricadi@gmail.com"},
    {"nom": "Stanley TOUSSAINT", "email": "stanley.toussaint@uniq.edu"}
]

projects = [
    "Projet 1 : Réseau social universitaire",
    "Projet 2 : Recommandation de contenus sur une plateforme vidéo",
    "Projet 3 : Détection de spam",
    "Projet 4 : Application GPS",
    "Projet 5 : Analyse d'un réseau informatique",
    "Projet 6 : Gestion d'une bibliothèque numérique",
    "Projet 7 : Analyse du trafic réseau",
    "Projet 8 : Détection de fraude bancaire",
    "Projet 9 : Répartition des tâches dans un centre de données",
    "Projet 10 : Traitement d'images",
    "Projet 11 : Analyse des performances d'un algorithme",
    "Projet 12 : Plateforme de commerce électronique",
    "Projet 13 : Système de triage médical assisté",
    "Projet 14 : Analyse de la fréquentation d'une plateforme éducative",
    "Projet 15 : Cybersécurité et authentification"
]

# Mélange aléatoire
random.shuffle(projects)

assignments = []

for i, student in enumerate(students):
    assignments.append({
        "Nom": student["nom"],
        "Email": student["email"],
        "Projet attribué": projects[i % len(projects)]
    })

df = pd.DataFrame(assignments)

print(df)

# Export Excel
df.to_excel("tirage_projets_modelisation.xlsx", index=False)

print("\nFichier généré : tirage_projets_modelisation.xlsx")