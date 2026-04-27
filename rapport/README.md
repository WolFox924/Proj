readme généré par gemini

# Projet de Rapport LaTeX

Ce dépôt contient le code source de notre rapport. Nous utilisons LaTeX pour la mise en page et Git pour collaborer efficacement.

## Structure du projet

* `main.tex` : **Fichier principal.** C'est lui qu'il faut compiler. Il contient la structure globale. Évitez de le modifier sans prévenir les autres.
* `chapters/` : Contient les différentes sections du rapport.
    * Chaque membre travaille sur son propre fichier `.tex` pour éviter les conflits Git.
* `images/` : Stockez ici toutes les illustrations, schémas et captures d'écran.
* `references.bib` : Fichier central pour toutes nos sources bibliographiques.

## Comment contribuer ?

### 1. Organisation des fichiers
Pour éviter que deux personnes modifient la même ligne en même temps :
- Travaillez dans des fichiers séparés dans le dossier `/chapters`.
- Si vous créez un nouveau fichier (ex: `concl.tex`), n'oubliez pas de l'ajouter dans le `main.tex` avec la commande `\include{chapters/concl}`.

### 2. Bonne pratique d'écriture (pour Git)
**Important :** Écrivez **une phrase par ligne**. 
En LaTeX, un simple retour à la ligne ne crée pas de nouveau paragraphe. Cela permet à Git de comparer les versions ligne par ligne et d'éviter 99% des conflits de fusion.

### 3. Workflow Git
Avant de commencer à rédiger, suivez toujours ces étapes :
1. `git pull` : Récupérer les dernières modifications des autres.
2. Rédigez votre contenu.
3. `git add .`
4. `git commit -m "Ajout de la partie sur [votre sujet]"`
5. `git push`

## Installation & Compilation

1. Installez une distribution LaTeX (TeX Live ou MiKTeX).
2. Utilisez **VS Code** avec l'extension **LaTeX Workshop** pour une compilation automatique au "Save".
3. Le fichier à compiler est toujours `main.tex`.

---
*Note : Si vous ajoutez des images, essayez de ne pas mettre de fichiers trop lourds (> 5Mo) pour garder le dépôt fluide.*