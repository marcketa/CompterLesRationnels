---
jupyter:
  jupytext:
    comment_magics: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 0.8.5
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.6.5
---

# ** Compter les fractions positives irréductibles **


Nous savons que l'ensemble des <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;\bg_white&space;\fn_cm&space;\large&space;\mathbb{Q}" target="_blank">
<img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;\bg_white&space;\fn_cm&space;\large&space;\mathbb{Q}" title="\large \mathbb{Q}" /></a> est dénombrable. Dans ce notebook nousprésentons deux mises en oeuvre de ce résultat sous la forme de la construction d'arbres binaires dont les noeuds portent toutes les fractions irréductibles positives.  
J'ai été trés impressionné par le chapitre du livre de Donald Knuth `Concrete Mathematics` dédié à ce thème sous la forme de la construction de l'arbre de Stern-Brocot. Il m'a semblé que transformer ce chapitre en un "dynabook", grâce à *jupyter notebook*, avec Python permettant d'expérimenter, était une bonne idée.  
Par ailleurs, entretemps, un ami m'a fait lire un article de Calkin et Wilf sur la construction d'un autre arbre binaire dont les noeuds étaient aussi toutes les fractions irréductibles positives. L'idée de trouver quelle était la relation entre ces deux arbres m'a immédiatement excité. Et j'ai été trés heureux de trouver une élégante relation entre ces deux versions, mais quelques recherches sur le web m'ont vite appris que ma découverte, sous d'autres formes avait déjà été faite...  
J'espère cependant que ma version pythonique pourra plaire à certains...  
  
Ce notebook utilise certaines extensions à Jupyter Notebook qui m'ont semblé utiles. Si vous êtes interéssés, vous pouvez regardez ce site:  https://ndres.me/post/best-jupyter-notebook-extensions/  
  
J'utilise ici Python.3.6.5 et j'ai aussi essayé les *annotations de type* importées du module `typing`. Vous pouvez regarder à: https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html

