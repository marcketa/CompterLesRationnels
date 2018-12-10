# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     comment_magics: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.1'
#       jupytext_version: 0.8.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.5
# ---

# %%
import numpy as np
from typing import Any, List, Union, Optional, Tuple, Callable
from fractions import Fraction
from math import *
import matplotlib.pyplot as plt
from itertools import product

# %% [markdown]
# # Arbres binaires

# %% [markdown]
# ## Niveaux
# Dans ce qui suit nous représenterons les `m` premiers niveaux d'un arbre binaire par une liste de `m` listes, chacune de $1,2,\cdots,2^{m-1}$ noeuds representant au niveau $0$ la racine, au niveau $1$ Les fils gauches et droits de la racine, et au niveau $k+1$ les fils gauches et droits du $i^{ème}$ noeud du niveau $k$  par respectivement les noeuds d'indices $2i$ et $2i+1$ de ce niveau $k+1$ où $i \in \{0,1,\ldots,k-1\}$.  
# Ci-dessous une fonction `bin_levels(lst)` qui transforme une liste `lst` en  les niveaux de l'arbre binaire correspondant.

# %% {"code_folding": [0]}
def bin_levels(lst: List[Any]) -> List[List[Any]]:
    """ divide a list in blocks (lists) corresponding to the first levels of a binary tree
    
    Args: 
        lst:  a list
    Returns:
        a list of levels: [lvls[0],...,lvls[k],..., lvls[-1]]] 
        length(lvls[0]) == 1, lvls[0][0] is the root of the binary tree
        for each level k > 0, len(lvls[k]) == 2*len(lvls[k-1]), 
        but for the last level, len(lvls[-1]) <= 2*len(lvls[-2]) depending on len(lst)
    Example:
        if lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        bin_levels(lst) -> [[1], [2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
    """
    n = len(lst)
    lvls = []
    m = 1
    while 2*m <= n:
        lvls.append(lst[m-1:2*m-1])
        m = 2*m
    lvls.append(lst[m-1:n+1])
    return lvls

# %%
l13 = list(range(1,14))
print('l13 = {}'.format(l13))
print('levels:             0    1       2             3')
print('bin_levels(l13):  {}'.format(bin_levels(l13)))


# %% [markdown]
# Et `print_bintree(levels)` est un "pretty print" des arbres binaires représentés de cette façon.

# %% {"code_folding": [0]}
def print_bintree(lvls : List[List[Any]],r: int = 1, fmt: Callable[[Any],str] = str) -> None:
    """ pretty print of a binary tree defined by lvls as in the result 
        of the bin_levels function
        
    Args:
        lvls: binary tree as a list of levels: [lvls[0],...,lvls[k],..., lvls[-1]]] 
              length(lvls[0]) == 1, lvls[0][0] is the root of the binary tree
              for each level k > 0, len(lvls[k]) == 2*len(lvls[k-1]), 
              but for the last level, len(lvls[-1]) <= 2*len(lvls[-2]) 
        r:    (int) w = 2*r+1 is the width of printed label for each node.
              default: r=1 => 3 characters for each node's label
        fmt:  a function defining the print format of each node. 
              default: the function str
    Returns:
        None: this function is just for printing the first levels of a binary tree 
    """
    def bn(klvl: int, r: int, n: int) -> int:
        """returns the half-distance between two nodes at level klvl
        
        Args:
            klvl: (int) the level number
            r:    (int) w = 2*r+1 is the width of label for each node
            n:    number of levels
        Returns:
            (int) the half-distance between two nodes at level klvl
        """
        return (2*r+1)*(2**(n-1-klvl)-1)
    n = len(lvls)
    # lbn: list of the half-distance lbn[k] between two nodes at each level k
    lbn = [bn(k,r,n) for k in range(n)]
    # lu[k]: length of horizontal branches in level k
    lu = [el//2 for el in lbn]
    # w: number of chars for each node
    w = 2*r+1
    for k in range(n):
        print(''.join([(lbn[k]+r)*' ' + '|' + (lbn[k]+3*r+1)*' ' for el in lvls[k]]))
        print(''.join([(lu[k]+(k!=(n-1)))*' ' + lu[k]*'_' + '{:^{}}'.format(fmt(el),w) + 
                       lu[k]*'_'+ (lu[k]+(k!=(n-1))+2*r+1)*' ' for el in lvls[k]]))

# %%
print_bintree(bin_levels(l13))

# %% [markdown]
# ## Chemin d'accès à un noeud (path string)
# Nous utiliserons les lettres L (Left) et R (Right) quand nous nous déplaçons depuis la racine d'un arbre binaire, en suivant la branche gauche ou droite à chaque noeud, afin d'atteindre un noeud particulier de l'arbre; ainsi une chaîne composée de caractères L et 
# R identifie de manière unique la place d'un noeud dans l'arbre.  
# Par exemple dans l'arbre ci-dessus, pour atteindre le noeud portant l'entier 12 à partir de la racine 1 on va à droite vers 3 puis à gauche vers 6 et enfin à gauche vers 12. Ainsi 
# le noeud portant l'étiquette 12 est identifié par le "chemin" `'RLL'`.  
# La fonction `paths_level(k)` retourne la liste ordonnée des "chemins" identifiant tous les noeuds du $k^{ième}$ niveau d'un arbre binaire:  
# *  `['']` pour la racine au niveau k = 0 
# *  `['L', 'R']` pour les fils gauche et droit de la racine au premier niveau  k = 1
# *  `['LL', 'LR', 'RL', 'RR']` au niveau k = 2, pour les fils gauche et droit respecti vement des noeuds du niveau 1   
# Et ainsi de suite ... 

# %%
def paths_level(k: int) -> List[str]:
    """ return the path strings describing the kth level of a binary tree:
    
    Args: 
        k: an integer
    Returns:
        a list of strings, defining the ordered list of the level's nodes
    Examples:
        [''] for the root at level k = 0
        ['L', 'R'] for the left and right sons of the root at the first level k = 1,
        ['LL', 'LR', 'RL', 'RR'] at level k = 2 for the respective left and right sons of level 1 nodes
        """
    return [''.join(t) for t in product(('L','R'),repeat=k)]

# %%
print(paths_level(3))

# %% [markdown]
# ##  Bits representation  

# %%
def ints2bin(ints: List[int], nbits: Optional[int] = None) -> List[str]:
    """ return the list of the binary representations on nbits of all integers in the list ints.
    
    Args:
        ints: a list of integers 
        nbits: an integer, the fixed length for all the bits string representing the integers in ints.
        If nbits == None then for each integer in ints the binary representation is the string of minimal length
    Returns:
        The list of the binary representation of the integers in ints
    Example:
        ints2bin([1,2],nbits=3) ->  ['001', '010']
        ints2bin([1,2]) ->  ['1', '10']
    """
    return [np.binary_repr(k,nbits) for k in ints]

def rev_ints(ints: List[int], nbits: Optional[int] = None) -> List[int]:
    """return the list of integers resulting of the reversing of the binary repr of the integers in  ints
    
    Args:
        ints: a list of integers 
        nbits: an integer, the fixed length for all the bits string representing the integers in ints.
        If nbits == None then for each integer in ints the binary representation is the string of minimal length   
    Returns:
        the list of integers resulting of the reversing of the binary repr of the integers in  ints
    Example:
        rev_ints([1,2],nbits=3) -> [4, 2] (['001', '010'] -> ['100', '010'])
        rev_ints([1,2]) -> [1, 1] (['1', '10'] -> ['1', '1'])
    """
    return [int(bits[::-1],2) for bits in ints2bin(ints,nbits)]

def str_translate(s: str, fromchrs: str = '01', tochrs: str = '10') -> str:
    """substitute in the string s the characters in fromchrs to those in tochrs and return the resulting string
    
    Args:
        s: a string
        fromchrs: a string listing all the characters to be changed
        tochrs: a string listing the corresponding substitutes
    Returns:
        the string resulting from the substitutions in s.
    Examples:
       str_transl('1101') -> '0010'
       str_transl('LLRL','LR','01') -> '0010'
    """
    if s == '':
        return tochrs[0]
    return s.translate(dict(list(zip((map(ord,fromchrs)),tochrs))))

# %% [markdown]
# Par les substitutions: `L -> 0` and `R -> 1`, nous remplaçons l'indexation d'un noeud avec un "chemin" par une paire d'entiers `(level, idx)` indiquant la position `idx` du noeud dans la liste décrivant le niveau `level`.  
# Par exemple le noeud de "chemin" `'RRL'` est le noeud d'indice `6` du niveau `3` (`len('RRL')`)  
# parceque `str_translate('RRL','LR','01')` est `'110'` et `int('110',2)` is `6`.
# Se souvenir que le noeud d'indice `6` est le $7^{ième}$ noeud, l'indexation en Python commençant toujours par l'indice `0`.  
# Nous définissons ci-dessous deux fonctions réciproques l'une de l'autre pour passer d'une indexation à l'autre: `level_idx(path_string)` et `path_str(level,idx)`

# %%
def level_idx(path_string: str) -> Tuple[int, int]:
    """ returns a pair of integers(level_number,idx) identifying the positition  
        idx of a node in the list representing the level in a binary tree.
    
    Args:
        path_string: a string of 'L' and 'R',describing the left and right from the root moves to reach a node
    Returns:
        a tuple of 2 integers: the node's level number and his index position in the corresponding list
    Example:
        level_idx('RRL') -> (3, 6)
        """
    return len(path_string), int(str_translate(path_string,'LR','01'), 2)

# %%
level_idx('RRL')

# %%
def path_str(level: int, idx: int) -> str:
    """ returns the path string of the node defined by level and idx
    
    Args:
        level: (int) node's level number of the node the pair of integers(level,idx) identifies the positition idx of a node 
        idx: (int) node's index in the list representing the level
    Returns:
        the node's corresponding path string
    Example:
        path_str(3,6) == 'RRL' 
    """
    return str_translate(np.binary_repr(idx,level),'01','LR')


# %%
path_str(3,6)

# %% [markdown]
# ces fonctions sont réciproques l'une de l'autre:

# %%
level_idx(path_str(*(3,6))) == (3,6) and path_str(*level_idx('RRL')) == 'RRL'

# %% [markdown]
# Si nous effectuons la même substitution sur tous les "chemins" d'un niveau, comme par exemple ceux du niveau `3` donnés par `paths_level(3)` nous obtenons évidemment la liste des $2^3 = 8$ premiers entiers consécutifs:  

# %%
paths_lvl3 = paths_level(3)
print(paths_lvl3)
print([str_translate(s,'LR','01') for s in paths_lvl3])
print([level_idx(s)[1] for s in paths_lvl3])

# %% [markdown]
# #  L'arbre binaire de Stern-Brocot   
# Les cellules de textes de ce chapitre sont pour la plupart une transcription du chapitre 
# consacré à l'arbre de Stern-Brocot dans le livre de Donald Knuth _Concrete Mathematics_ (p.129) et les cellules de code matérialisent en Python les concepts exposés dans ce chapitre.

# %% [markdown]
# Une belle construction de l'ensemble de toutes les fractions positives ou nulle $m/n$ avec $m \perp n$ est celle de la construction de l'arbre binaire de Stern-Brocot.  
# Ici nous utilisons la notation $m \perp n$ de Donald Knuth pour désigner la relation entre deux entiers $m,n$: "$m$ et $n$ sont premiers entre eux (copremiers)". Ce qui revient aussi à dire que le plus grand commun dénominateur de $m$ et $n$ est 1 ($pgcd(m,n) = m \wedge n = 1$)   
# Une fraction $m/n$ est dite irréductible si et seulement si $m \perp n$ .  
# L'idée est la suivante: On commence avec la paire $(0/1, 1/0)$ la plus petite fraction positive ou nulle et la plus grande (en fait on a ajouté la paire d'entiers (numérateur=1,dénominateur=0) pour noter l'$\infty$ qui ne sera jamais atteint!)  
# Ensuite on répète l'opération suivante aussi longtemps qu'on le désire:  
# Insérer $(m + m')/(n + n')$ entre deux fractions adjacentes $m/n$ and $m'/n'$.
# La nouvelle fraction **(m+m')/(n+n')** est appelée la  **médiante** de **m/n** and **m'/n'**.  
# Le premier pas nous donne la médiante de $0/1$ et $1/0$, soit $1/1$.  
# le second pas nous donne deux nouvelles fractions: la médiante $1/2$ entre $0/1$ et $1/1$ d'un côté et la médiante $2/1$ entre $1/1$ et $1/0$.  
# Et ainsi de suite en construisant un arbre binaire de racine $1/1$.  
#

# %% [markdown]
# ##  Arbre des numérateurs  
# Pour commencer nous allons d'abord construire l'arbre binaire correspondant aux numérateurs de toutes les fractions de l'arbre de Stern-Brocot. 
# Nous partons donc de la paire d'entiers (0,1) et nous répétons autant de fois que nous le désirons, c'est à dire jusqu'au niveau désiré, l'opération suivante:  
# Insérer m + m' entre deux entiers adjacents m et m'.  
# Construire ainsi successivement chaque niveau et en même temps construire une liste "cumulée" de toutes les opérations effectuées:  
# niveau 0: `[0+1] = [1]`  
# cumul: `[1]`
# niveau 1: `[0+1,1+1] = [1,2]`  
# cumul:`[1,1,2]`  
# niveau 2: `[0+1,1+1,1+2,2+1] = [1,2,3,3]` 
# etc...    
# On trouve des sources pour une telle construction:

# %% [markdown]
# ##  Suite de Stern (diatomique)  
# C'est la suite qui peut être définie par:  
# $$s(0) = 0,\  s(1) = 1 \  \textrm{and} \ \ 
# s(2n) = s(n),\  s(2n + 1) = s(n) + s(n + 1)\ 
# \textrm{when} \ (n ≥ 1)$$
# Cette suite apparaît dans la littérature sous différentes formes et notations.  
# C'est la suite A002487 dans [_The On-Line Encyclopedia of Integer Sequences_](http://oeis.org), où l'on trouve de nombreuses propriétés de cette suite ainsi que
# d'autres références.  
# Les premiers termes non nuls de cette suite sont, d'aprés la définition ci-dessus:
#  **1**, **1**, 2, **1**, 3, 2, 3, **1**, 4, 3, 5, 2, 5, 3, 4, **1**, 5,. . . , où les termes dont l'indice est une puissance de 2 sont les valeurs **1** en caractères gras.  
# Voir aussi la référence [Rational Trees and Binary Partitions](http://www.luschny.de/math/seq/RationalTreesAndBinaryPartitions.html)  de Peter Luschny, Mars 2010.  
# La fonction Python `stern_levels` qui suit est une adaptation libre d'un programme Maple écrit par E.Dijkstra, Selected Writings on Computing, Springer, 1982, p. 232.

# %%
def stern_levels(m: int, a: int = 0, b: int = 1) -> Tuple[List[List[int]], List[int]]:
    """ This function build the first m levels of the numerators of the Stern-Brocot binary tree
        and the list of the first 2**m-1 terms of the Stern sequence
    
    Args:
        m: (int) the desired number of levels for the Stern-Brocot binary tree
        a: (int) first initial value
        b: (int) second initial value
    Returns: a tuple t = (levels, l)
        t[0]: levels, a m terms list where levels[k] is a list of 2**k integers, 
              a representation of the tree of the numerators of kth level in the Stern-Brocot tree.
        t[1]: list of the first 2**m-1 terms of the Stern sequence
    """
    l = [a,b]
    levels = []
    for k in range(1,m+1):
        l_k = [l[0]]
        level_k = []
        for i in range(len(l)-1):
            si = l[i]+l[i+1]
            level_k.append(si)
            l_k.append(si)
            l_k.append(l[i+1])
        levels.append(level_k)
        l = l_k[:]
    return levels,l[1:-1]

# %%
SBnums_5,stern_list_5 = stern_levels(5)

# %%
for l in SBnums_5:
    print(l)

# %% [markdown]
# Le premier résultat `SBnums_5` nous donne les 5 premiers niveaux de l'arbre des numérateurs: 

# %%
print_bintree(SBnums_5)

# %%
print(stern_list_5)

# %% [markdown]
# Quant au second resultat `stern_levels(5)[1]`, nous noterons qu'il représente la "projection ordonnée" sur une ligne (dans une liste) de tous les noeuds de cet arbre à  niveaux.

# %% [markdown]
# Nous noterons aussi que l'arbre des dénominateurs peut être obtenu de la même façon à partir des entiers (1,0), en utilisant la même fonction `stern_levels(m,1,0)` ou bien en renversant chaque liste dans le premier résultat de `SBnums(m,0,1)`:

# %%
SBdens_5, _ = stern_levels(5,1,0)

# %%
for l in SBdens_5:
    print(l)

# %% [markdown]
# ou en utilisant la notation `Python` `l[::-1]` pour renverser l'ordre d'une liste `l`

# %%
SBdens_5 = [l[::-1] for l in SBnums_5]

# %%
for l in SBdens_5:
    print(l)

# %% [markdown]
# Et pour chaque niveau nous pouvons construire la liste des noeuds, chaque noeud cinsidéré comme une paire `(numerateur,denominateur)`.  
# Par exemple les noeuds du niveau 3:

# %%
i=3
print(list(zip(SBnums_5[i],SBdens_5[i])))


# %% [markdown]
# Ou comme liste des fractions correspondantes:

# %%
i=3
print([str(Fraction(*pair)) for pair in zip(SBnums_5[i],SBdens_5[i])])

# %% [markdown]
# De cette façon nous pouvons construitre l'arbre de Stern-Brocot jusqu'au niveau `m` desiré.

# %%
def SBpairs(m: int) -> List[List[Tuple[int, int]]]:
    """ return the first m levels of Stern-Brocot tree, the nodes being a pair (numerator,denominator)
    
    Args:
        m: (int) the desired number of levels for the Stern-Brocot binary tree
    Returns:
        the first m levels of Stern-Brocot tree, with the pairs (numerator, denominator) as nodes
    """
    sbnums_m, _ = stern_levels(m,0,1)
    sbdens_m, _ = stern_levels(m,1,0)
    return [list(zip(sbnums_m[k],sbdens_m[k])) for k in range(m)]

# %%
for l in SBpairs(5):
    print(l)

# %%
print_bintree(SBpairs(5),fmt=lambda pair:'{},{}'.format(*pair))

# %%
print_bintree(SBpairs(5),fmt=lambda pair:str(Fraction(*pair)))

# %% [markdown]
# And if we combine the second results `stern_levels(m,a=0,b=1)[1]` and `stern_levels(m,a=1,b=0)[1]`
# we'll get the "projection", as an ordered sequence of all the fractions of the `m` levels tree just above.

# %%
m=5
for frac in [Fraction(*pair) for pair in zip(stern_levels(m,a=0,b=1)[1],stern_levels(m,a=1,b=0)[1])]:
    print(frac,end=' ')

# %% [markdown]
# # L'arbre de toutes les fractions positives irréductibles

# %% [markdown]
# Rappelons ce que nous avons écrit plus haut:  
# Si $m/n < m'/n'$ sont deux fractions consécutives, à n'importe quelle étape de la construction, nous avons $m'n − mn' = 1$    
# De par l'identité de Bezout cela signifie que $m,n$ sont premiers entre eux (ou copremiers): $m \perp n$ et qu'il en est de même pour les entiers $m',n'$: $m \perp n$.    
# Il s'ensuit que les fractions $m/n$ et $m'/n'$ sont irréductibles.  
#
# 1. C'est vrai pour $0/1$ et $1/0$ car $( 1 · 1 − 0 · 0 = 1 )$.
# Notosn de nouveau que nous acceptons $1/0$ pour $\infty$ car en fait nous raisonnons sur la paire `(1,0)`...
# 2. Quand nous insérons la médiante $(m + m')/(n + n')$ , il nous faut vérifier la nouveaux cas:   
# $$(m + m')n − m(n + n') = 1 $$ 
# $$m'(n + n') − (m + m')n' = 1$$ 
# Or aprés simplification, ces deux équations sont équivalente à la condition originelle $m'n − mn' = 1$.   
# Ainsi $m \perp n$ et  $m' \perp n'$ implique la même propriété pour la médiante: $(m + m') \perp (n + n')$.
# La médiante est elle aussi une fraction irréductible
# 3. De plus comme tous les entiers considérés sont positifs et comme:  
# $$m'n − mn' = 1 \Leftrightarrow \frac{m'}{n'} − \frac{m}{n} = \frac{1}{nn'}$$      
# nous avons: 
# $$\frac{m}{n} < \frac{m + m'}{n + n'} < \frac{m'}{n'}$$ 
# Donc la construction préserve l'ordre, expliquant pourquoi la "projection" décrite ci-dessus est la suite ordonnée des fractions considérées; il est donc impossible de rencontrer la même fraction en deux emplacements différents.
# 4. Il reste une question: Aurait on pu oublier une fraction $a/b$ avec $a ⊥ b$?  
# La réponse est non. Car nous pouvons examiner la construction dans un voisinage de $a/b$. Dans ce voisinage il est facile d'analyser la situation. Initialement nous avons:  
# $$\frac{m}{n} = \frac{0}{1} \  < \  (\frac{a}{b}) \  < \ \frac{1}{0} = \frac{m'}{n'}$$
# Nous avons mis $a/b$ entre parenthèses pour indiquer que cette fraction n'a pas encore été trouvée par la construction.  
# Donc si à une certaine étape nous avons:
# $$\frac{m}{n} \  < \  (\frac{a}{b}) \  < \  \frac{m'}{n'}$$ 
# trois cas sont possibles:  
# * $(m + m' )/(n + n') = a/b$ auquel cas nous avons gagné  
# * $(m + m')/(n + n') < a/b$ et nous pouvons continuer avec les substitutions suivante $m ← m + m' , n ← n + n'$   
# * $(m + m')/(n + n') > a/b$ et nous continuons avec les substitutions $m' ← m + m' , n' ← n + n'$ .  
# Ce processus ne peut continuer indéfiniment car les conditions:  
# $$ \frac{a}{b} - \frac{m}{n} > 0 \quad  \textrm{   and   } \quad  \frac{m'}{n'} - \frac{a}{b} > 0 $$
# impliquent que:
# $$ an − bm \ \geq \ 1  \quad  \textrm{   and   } \quad   bm' − an' \ \geq \ 1 $$
# Donc:
# $$ an − bm \ \geq \ 1  \quad  \textrm{   and   } \quad   bm' − an' \ \geq \ 1 $$
# Ce qui est le même que:  
# $$ a + b \  \geq \ m' + n'+ m + n$$ 
# Or soit $m$ ou $n$ ou $m'$ ou $n'$ augmente à chaque étape, donc en au plus $a + b$ étapes nous aurons atteint $a/b$.  
# Conclusion: **Les noeuds de l'arbre de Stern-Brocot sont toutes les fractions positives m/n with m ⊥ n.**

# %% [markdown]
# # L'arbre de Calkin-Wilf  
# Dans un article dont le titre est [_Recounting the rationals_](https://fermatslibrary.com/s/recounting-the-rationals#email-newsletter) Neil Calkin et Herbert S. Wilf présentent une autre représentation par un arbre binaire des rationnels positifs que nous appelerons arbre de Calkin-Wilf utilisant la **suite de Stern (diatomique) sequence**  dont nous avons déja parlé plus haut et dont nous avons montré que les termes la suite `stern_levels(m)[1]` sont justement les $2^m-1$ premiers termes (en excluant le premier terme égal à 0). Ce sont aussi les 
# $2^m-1$ numérateurs de la représentation de Calkin-Wilf selon leur article.  
# Par exemple pour $m = 5$:

# %%
stern_list_5 = stern_levels(5)[1]
print(stern_list_5)

# %% [markdown]
# Et si nous construisons l'arbre binaire correspondant à cette suite tel qu'engendré par `bin_levels(stern_list_5)` nous obtenons l'arbre bianier des numérateurs des fractions de l'arbre de Calkin-Wilf:

# %%
print_bintree(bin_levels(stern_list_5))

# %% [markdown]
# et selon le même article la liste des dénominateurs associés est:

# %%
print(stern_list_5[1:]+[1])

# %% [markdown]
# Et suivant le même article [_Recounting the rationals_](https://fermatslibrary.com/s/recounting-the-rationals#email-newsletter) nous pouvons construire l'arbre de Calkin-Wilf représentant toutes les fracions positives irréductibles. Plus exactement nous construirons une fonction `CWpairs(m)` qui retourne les $m$ 
# premiers niveaux portant les $2^m - 1$ noeuds de cet arbre, où les noeuds sont les paires d'entiers `(num,den)`
# plutôt que les fractions `num/den`.

# %%
def CWpairs(m):
    """ return the first m levels of Calkin-Wilf tree, the nodes being a pair (numerator,denominator)
    
    Args:
        m: (int) the desired number of levels for the Calkin-Wilf binary tree
    Returns:
        the first m levels of Calkin-Wilf tree, with the pairs (numerator, denominator) as nodes
    """
    nums = stern_levels(m)[1]
    dens = nums[1:]+[1]
    return bin_levels(list(zip(nums,dens)))

# %%
print_bintree(CWpairs(5),fmt=lambda pair:'{},{}'.format(*pair))

# %%
print_bintree(CWpairs(5),fmt=lambda pair:str(Fraction(*pair)))

# %% [markdown]
# **Nous remarquons que pour chaque  `m` l'arbre de Stern-Brocot tree et celui de Calkin-Wilf, contient les mêmes fractions, car pour chaque `0<k<m`, les niveaux `SBpairs(m)[k]` et  `CWpairs(m)[k]` contiennent deux différentes permutations des mêmes paires, provenant de la même suite `stern_levels(m)[1]`.**   
# Comme nous l'avons vu auparavant avec `m = 5`:

# %%
print(stern_levels(5)[1])

# %% [markdown]
# #  Arbre de Stern-Brocot ou de Calkin-Wilf comme représentations des fractions positives irréductibles   
# Comme nous l'avons vu dans le chapitre **Arbres Binaires** nous pouvons utiliser les lettres L et R pour se déplacer sur la branche gauche ou droite d'un noeud et à partir de la racine atteindre n'importe quel noeud, une chaîne de L et de R identifiant de manière unique une place dans l'arbre. Nous pouvons utiliser cette démarche pour atteindre une fraction précise dans l'un des deux arbres que nous avons construit.  
# Dans l'arbre de Stern-Brocot, par exemple, le chemin LRLL signifie que nous allons à gauche de 1/1 à 1/2, puis à droite à 2/3, à gauche à 3/5 et enfin à gauche jusqu'à 4/7.  Dans **cet arbre**, nous pouvons considérer LRLL comme une représentation de 4/7. Chaque fraction positive irréductible est ainsi représentée ainsi par un chemin unique.   
# Mais comme un chemin identifie la même position unique dans un arbre binaire quelconque, le même chemin LRLL identifiera une autre fraction dans l'arbre de Calkin-Wilf: De la racine 1/1 à gauche on passe par 1/2, puis à droite par 3/2, et à gauche par 3/5 et à gauche enfin pour atteindre 3/8.  
# La racine 1/1 dans les deux cas correspond au chemin vide '' que nous appelerons I comme 1, et comme le symbole "identité".  
# Dans cette représentation se posent deux questions:
# 1. Etant donnés deux entiers positifs $m$ et $n$ vérifiant $m \perp n$, quel est le chemin composé des lettres L et R qui correspond à la fraction $m/n$?  
# 2. Etant donné un chemin `S` composé des lettres L et R, quelle est la fraction représentée par ce chemin?  
#   
# La seconde question semble plus facile. C'est donc celle là que nous traiterons en premier.  
# Pour Stern-Brocot nous définirons une fonction:  
# `SBfrac(S) =` la fraction de l'arbre de Stern-Brocot correspondant au chemin `S`.  
# Nous avons vu, par exemple, que `SBfrac('LRLL') == Fraction(4,7)`.  
# De même, pour l'arbre de Calkin-Wilf, nous définirons la fonction:  
# `CWfrac(S) =` la fraction de l'arbre de Calkin-Wilf correspondant au chemin `S`.  
# Plus haut nous avons vu l'exemple `CWfrac('LRLL') == Fraction(3,8)`.

# %% [markdown]
# ##   Représentation matricielle pour l'arbre de Stern-Brocot
# Aux chemins élémentaires '' ou 'I', 'L', 'R' de l'arbre de Stern-Brocot nous associons des matrices 2x2:  
# Au chemin '' ou 'I' nous faisons correspondre la racine 1/1, notée aussi bien par le vecteur $\begin{bmatrix} 1 \\ 1 \end{bmatrix}$. Mais cette racine est aussi la médiante de 0/1 et 1/0, ou en raisonnant sur des vecteurs, la médiante des des deux vecteurs $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$ et $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$, ce qu'on peut aussi considérer comme les vecteurs colonnes de la matrice $\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$.  
# Il serait plus agréable, dans notre cas que cette matrice soit la matrice unité $\begin{equation*}I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}\end{equation*}$, ce qui revient, dans le cas de Stern-Brocot, à représenter la fraction 0/1 par le vecteur colonne $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ et plus généralement, toujours dans le cas de Stern-Brocot, une fraction num/den par le vecteur colonne $\begin{bmatrix} den \\ num \end{bmatrix}$  
#
# Aller à gauche avec la matrice $\begin{equation*}L = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}\end{equation*}$:  En effet le fils gauche $\begin{bmatrix} 2 \\ 1 \end{bmatrix}$ (pour  1/2) de la racine $\begin{bmatrix} 1 \\ 1 \end{bmatrix}$ est bien la médiante des deux vecteurs colonne $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ et $\begin{bmatrix} 1 \\ 1 \end{bmatrix}$ et d'ailleurs 
# $\begin{equation*}L \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}\end{equation*}$   
# Et de même à droite avec la matrice $\begin{equation*}R = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}\end{equation*}$:  En effet le fils droit $\begin{bmatrix} 1 \\ 2 \end{bmatrix}$ (pour  2/1) de la racine $\begin{bmatrix} 1 \\ 1 \end{bmatrix}$ est bien la médiante des deux vecteurs colonne $\begin{bmatrix} 1 \\ 1 \end{bmatrix}$ et $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ et d'ailleurs 
# $\begin{equation*}R \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}\end{equation*}$     
# Maintenant nous pouvons représenter un chemin composé des caractères pris dans 'IRL', c'est à dire une suite de 'L' et de 'R' par le produit M des matrices correspondantes L et R.   
# En Python nous définirons les matrices par des arrays numpy `np.array` de shape `(2, 2)` et les fonctions suivantes:    
# `powmat(M,n)`   pour $M^n$ où $M$ est une matrice  
# `matprod(mats)` pour la matrice produit $M_0 M_1 \cdots M_n$ if  $\textrm{mats} = [M_0, M_1, \cdots, M_n]$ où, pour $0 \leq i \leq n$,  $M_i$  est L ou R.    
# `path2mat(S)`   pour le produit matriciel des matrices élémentaires L, R dans le chemin S représentant un noeud de l'arbre. 

# %%
""" Specific 2x2 matrices in the Stern-Brocot context"""
L = np.array([[1,1],[0,1]])  # left move, left root's son
R = np.array([[1,0],[1,1]])  # right move, right root's son
I = np.eye(2,dtype=int)      #identity, root

def powmat(M: np.array, n: int) -> np.array:
    """ return the n-th power of a matrix M 
    
    Args: 
        M: (np.array) a square matrix pxp
        n: (int) exponent
    Returns:
        the n-th power of M
    """    
    assert n >= 0, "{} is not a positive or null integer".format(n)
    if n == 0:
        return np.eye(M.shape[0],dtype=int)
    return M @ powmat(M,n-1)

def matprod(mats: List[np.array], n: int = 2)-> np.array:
    """ mats is a list of matrices pxp. By defaut n == 2
    
    Args:
        mats: a list of square matrices pxp
    Returns:
        the matrix (np.array) product of all the matrices in the list in the same order
    """
    if len(mats) == 0:
        return np.eye(n,dtype=int)
    if len(mats) == 1:
        return mats[0]
    return mats[0] @ matprod(mats[1:])

def path2mat(S: str) -> np.array:
    """return the matrix product corresponding to a path string in a Stern-Brocot binary tree 
    
    Args:
        S: (str) a node's path string
    Returns:
        the corresponding product matrix (np.array)
    """
    return matprod([eval(chr) for chr in S])

print('I = ')
print(I)
print('L = ')
print(L)
print('R = ')
print(R)

# %%
powmat(L,0)

# %%
L.shape

# %%
L@L@L

# %%
path2mat('')

# %%
path2mat('LRLL')

# %%
M = L@R@L@L
print(M)

# %% [markdown]
# Comme dit plus haut, dans l'arbre de Stern-Brocot la fraction $\dfrac{n}{d}$ est représentée par le vecteur $[d,n]$. Avec cette notation le noeud à gauche de $\dfrac{1}{2}$ est le noeud $\dfrac{1}{3}$ décrit par le vecteur `L@[2,1] = [3 1]` et le noeud à droite de $\dfrac{1}{2}$ est le noeud décrit par le vecteur `R@[2,1] = [2 3]` pour la fraction $\dfrac{3}{2}$

# %% [markdown]
# et le produit
# $\begin{equation*}
# LRLL\begin{bmatrix} 1 \\ 1 \end{bmatrix}=\begin{bmatrix} 2 & 5 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 
# \begin{bmatrix} 7 \\ 4 \end{bmatrix}
# \end{equation*}$ represente la fraction $\dfrac{4}{7}$.

# %%
print(M@[1,1])

# %% [markdown]
# ##  Représentation matricielle pour l'arbre de Calkin-Wilf   
# Si nous considérons un noeud de l'arbre de Calkin-Wilf $\dfrac{n}{d}$ représenté par le vecteur ligne $\begin{bmatrix}n & d\end{bmatrix}$ et si nous utilisons les mêmes matrices L et R définies pour l'arbre de Stern-Brocot:  
# Le produit $\begin{equation*}\begin{bmatrix}n & d\end{bmatrix}L = \begin{bmatrix}n & d\end{bmatrix}\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix}n & n + d\end{bmatrix}\end{equation*}$ représente le noeud $\dfrac{n}{n+d}$  
# Le produit $\begin{equation*}\begin{bmatrix}n & d\end{bmatrix}R = \begin{bmatrix}n & d\end{bmatrix}\begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix}n + d & d \end{bmatrix}\end{equation*}$ représente le noeud $\dfrac{n+d}{d}$   
# Or ce sont exactement les règles de construction de l'arbre de Calkin-Wilf.
# Il s'agit de la même matrice M mais dans le cas de Stern Brocot on obtient un noeud sous la forme d'un vecteur colonne $\begin{equation*}M \begin{bmatrix}1 \\ 1\end{bmatrix} = \begin{bmatrix}d \\ n\end{bmatrix}\end{equation*}$   
# alors que dans le cas de Calkin-Wilf la multiplication de M se fait à gauche par un vecteur ligne, ce qui reviendrait à la multiplication à droite par la matrice transposée. 
# Ainsi avec la même produit M = LRLL nous obtenons la fraction $\dfrac{3}{8}$ dans l'arbre de Calkin-Wilf:

# %%
print([1,1]@L@R@L@L)
print([1,1]@M)


# %% [markdown]
# #  Fraction représentée par un chemin  
# Remarquons aussi que la matrice transposée L.T de L est justement R et donc que la transposée R.T de R est L.

# %%
print(L.T == R)
print(R)

# %% [markdown]
# Et sachant que la transposée d'un produit de matrices est le produit dans l'ordre inverse des transposées des matrices on commence à comprendre mieux les relations entre l'arbre de Stern-Brocot et celui de Calkin-Wilf

# %%
print((L@R@L@L).T == (L.T@L.T@R.T@L.T))
print((L@R@L@L).T == R@R@L@R) 
print(R@R@L@R)
print(M.T@[1,1])


# %% [markdown]
# Nous pouvons ainsi définir une fonction `SBfrac(S)` qui retourne la valeur de la fraction qui se trouve à la place définie par le chemin `S` dans l'arbre de Stern-Brocot:

# %%
def SBfrac(S: str) -> Tuple[int, int]:  # -> Fraction[int, int] type hint doesn't recognize Fraction
    """ return the Stern-Brocot node value as the fraction corresponding to the string path S
    
    Args:
        S: (str) a Stern-Brocot node path string
    Returns:
        the Fraction value of the corresponding node
    Example:
        SBfrac('LRLL') -> Fraction(4, 7)
    """
    M = matprod([eval(chr) for chr in S])
    den, num = M@[1,1]
    return Fraction(num, den)

# %%
SBfrac('LRLL')

# %%
print(SBfrac('LRLL'))

# %% [markdown]
# et de même définir la fonction `CWfrac(S)` qui retourne la valeur de la fraction qui se trouve à la place définie par le chemin S dans l'arbre de Calkin-Wilf:

# %%
def CWfrac(S: str) -> Tuple[int, int]:  # -> Fraction[int, int] type hint doesn't recognize Fraction
    """ return the Calkin-Wilf node value as the fraction corresponding to the string path S
    
    Args:
        S: (str) a Calkin-Wilf node path string
    Returns:
        the Fraction value of the corresponding node
    Example:
        CWfrac('LRLL') -> Fraction(3, 7)
    """
    M = matprod([eval(chr) for chr in S])
    num,dem = [1,1]@M
    return Fraction(num,dem)

# %%
print(CWfrac('LRLL'))

# %% [markdown]
# Une autre manière d'interpréter cette situation est de remarquer qu'une fraction $\dfrac{n}{d}$ correspondant à un chemin `S` dans un des arbres correspond au chemin inversé dans l'autre arbre `S[::-1]`.  
# **C'est le résultat le plus important qui établit la relation entre l'arbre de Stern-Brocot et celui de Calkin-Wilf**   
# Pour chaque chemin `S`: **`SBfrac(S) == CWfrac(S[::-1])`**  
# En reprenant les exemples précédents:  

# %%
SBfrac('LRLL') == CWfrac('LRLL'[::-1]) == CWfrac('LLRL')

# %%
print(CWfrac('LLRL'))

# %%
print(CWfrac('LRLL'))
print(SBfrac('LLRL'))

# %% [markdown]
# On peut ainsi reconstruire, par exemple les cinq premiers niveaux de l'arbre de Stern-Brocot, en appliquant la fonction `SBfrac` à tous les chemins du niveau `k` tels qu'ils sont engendrés par la fonction  `paths_level(k)` et en itérant sur les valeurs de `k` pour `k in [0,1,2,3,4]`:

# %%
SBlevels5 = [[SBfrac(S) for S in paths_level(k)] for k in range(5)]
print_bintree(SBlevels5)

# %% [markdown]
# Et nous pouvons construire les cinq premiers niveaux de l'arbre de Calkin-Wilf en appliquant de la même manière la fonction `CWfrac` aux mêmes chemins:

# %%
CWlevels5 = [[CWfrac(S) for S in paths_level(k)] for k in range(5)]
print_bintree(CWlevels5)

# %% [markdown]
# Nous avons ainsi résolu la question:  
# Etant donné un chemin composé de L et de R, à quelle fraction correspond ce chemin?
# Nous l'avons résolue dans la représentation de l'arbre de Stern-Brocot comme dans celle de l'arbre de Calkin-Wilf.

# %% [markdown]
# # Chemin correspondant à une fraction donnée

# %% [markdown]
# ## Arbre de Stern-Brocot (première version)   
# Nous avons vu que la fraction portée par un noeud quelconque de l'arbre de Stern-Brocot est plus grande que la fraction portée par le fils gauche de ce noeud et aussi plus petite que la fraction portée par le fils droit de ce noeud:  
# Pour tout chemin `S`: `SBfrac(S + 'L') < SBfrac(S) < SBfrac(S + 'R')`
# De sorte que pour atteindre le noeud portant la fraction `n/d` nous nous déplaçons depuis la racine `1` vers la gauche ou la droite selon que la fraction recherchée `n/d` est plus petite ou plus grande que la fraction portée par le noeud courant, tel que décrit dans la fonction `SBpathDemo(frac)`:

# %%
def SBpathDemo(frac: Union[Tuple[int, int], str]) -> str: 
    # -> Union[Tuple[int, int], str, Fraction[int, int]] type hint doesn't recognize Fraction
    """ find the Stern-Brocot path string corresponding to a fraction by a binary search 
        moving from the root down to the frac value
    Args:
        frac: a fraction as a tuple (numerator: int, denominator: int) as (3,8) 
        or a string as '3/8' or a fraction as Fraction(3,8)
    Returns:
        the path string S:
    Example:
        SBpathDemo(3/8) -> 'LLRL'        
    Demo: this version print the intermediate results of the binary search   
    """
    if type(frac) is tuple:
        frac = Fraction(*frac)
    else:
        frac = Fraction(frac)
    S = ''
    print('S = {:<7}: frac(S) = {:<3} '.format(S,str(SBfrac(S))),end=' -> ')
    while frac != SBfrac(S):
        if frac < SBfrac(S):
            print('{} < {:<3} go to left'.format(frac,str(SBfrac(S))))
            S += 'L'
        else:
            print('{} > {} go to right'.format(frac,SBfrac(S)))
            S += 'R'
        print('S = {0:<7}: frac(S) = {1:} ' .format(S,SBfrac(S)),end=' -> ')
    return S

# %%
SBpathDemo(3/8)

# %% [markdown]
# ##  Arbre de Calkin-Wilf  
# Selon la représentation matricielle de l'arbre de Calkin-Wilf nous savons qu'un noeud portant la fraction `n/d`:  
# * si `d > n` ce noeud est le fils gauche du noeud portant la fraction `n/(d-n)`  
# * if `d < n` ce noeud est le fils droit du noeud portant la fraction  `(n-d)/d`  
# Nous pouvons partir du neoud avec `n/d`et nous déplacer vers la racine en utilisant cette règle pour construire à chaque pas le chemin correspondant, comme on peut le voir en exécutant la fonction `CWpathDemo(frac)`:

# %% {"code_folding": []}
def CWpathDemo(frac: Union[Tuple[int, int], str]) -> str: 
    # -> Union[Tuple[int, int], str, Fraction[int, int]] type hint doesn't recognize Fraction
    """ find the Calkin-Wilf path string corresponding to a fraction by a binary search 
        moving from the frac value up to the root
    Args:
        frac: a fraction as a tuple (numerator: int, denominator: int) as (3,8) 
        or a string as '3/8' or a fraction as Fraction(3,8)
    Returns:
        the path string S:
    Example:
        CWpathDemo(3/8) -> 'LRLL'        
    Demo: this version print the intermediate results of the binary search   
    """
    if type(frac) is tuple:
        num,den = frac
    else:
        frac = Fraction(frac)
        num,den = frac.numerator,frac.denominator
    S = ''
    print('{}/{}'.format(num,den),end=': ')
    while num != den:
        if num > den:
            print('{} > {} coming from right ->'.format(num,den),end=' ')
            S = 'R' + S
            num -= den
            print('{}/{}: {}'.format(num,den,S))
        else:
            print('{} < {} coming from left  ->'.format(num,den),end=' ')   
            S = 'L' + S
            den -= num  
            print('{}/{}: {}'.format(num,den,S))      
        print('{}/{}'.format(num,den),end=': ')
    print('path_str({}) == {}'.format(str(frac),S))
    return S

# %%
CWpathDemo(3/8)

# %% [markdown]
# Et sans l'impression pour la "demo", Nous définissons la fonction `CWpath(frac)`:

# %%
def CWpath(frac: Union[Tuple[int, int], str]) -> str: 
    # -> Union[Tuple[int, int], str, Fraction[int, int]] type hint doesn't recognize Fraction
    """ find the Calkin-Wilf path string corresponding to a fraction by a binary search 
        moving from the frac value up to the root
    Args:
        frac: a fraction as a tuple (numerator: int, denominator: int) as (3,8) 
        or a string as '3/8' or a fraction as Fraction(3,8)
    Returns:
        the path string S:
    Example:
        CWpathDemo(3/8) -> 'LRLL'          
    """
    if type(frac) is tuple:
        num,den = frac
    else:
        frac = Fraction(frac)
        num,den = frac.numerator,frac.denominator
    S = ''
    while num != den:
        if num > den:
            S = 'R' + S
            num -= den
        else:
            S = 'L' + S
            den -= num
    return S    

# %%
CWpath(3/8)

# %% [markdown]
# ##  Arbre de Stern-Brocot (deuxième version)
# Nous remarquons que le calcul du chemin par la fonction `SBpathDemo` est bien plus coûteuse que le calcul du chemin dans l'arbre de Calkin-Wilf par la fonction `CWpathDemo`. Or nous savons que:  
# Pour chaque chemin `S`: `SBfrac(S) == CWfrac(S[::-1])`   
# et donc que:  
# Pour chaque fraction `frac` on a : `SBpath(frac) == CWpath(frac)[::-1]`  et donc nous pouvons rechercher le chemin d'une fraction dans l'arbre de Calkin-Wilf et le renverser pour trouver le chemin correspondant de Stern-Brocot.  
# De cette façon nous pouvons définir `SBpath(frac)` par:

# %%
def SBpath(frac: Union[Tuple[int, int], str]) -> str: 
    # -> Union[Tuple[int, int], str, Fraction[int, int]] type hint doesn't recognize Fraction
    """ find the Calkin-Wilf path string S corresponding to a fraction by a binary search 
        moving from the frac value up to the root and return the reverse path S[::-1],
        which is the corresponding Stern-Brocot path
    Args:
        frac: a fraction as a tuple (numerator: int, denominator: int) as (3,8) 
        or a string as '3/8' or a fraction as Fraction(3,8)
    Returns:
        the path string S:
    Example: SBpath(3/8) -> CWpath(3/8)[::-1] == 'LRLL'[::-1] == 'LLRL'
    """
    return CWpath(frac)[::-1]

# %%
SBpath(3/8)

# %% [markdown]
# Ou en construisant directement le chemin renversé en utilisant le code de `CWpath(frac)`:

# %%
def SBpath(frac: Union[Tuple[int, int], str]) -> str: 
    # -> Union[Tuple[int, int], str, Fraction[int, int]] type hint doesn't recognize Fraction
    """ find the Stern-Brocot path string S corresponding to a fraction by a binary search 
        moving from the frac value up to the root on Calkin-Wilf and build on each move 
        the corresponding Stern-Brocot path string S

    Args:
        frac: a fraction as a tuple (numerator: int, denominator: int) as (3,8) 
        or a string as '3/8' or a fraction as Fraction(3,8)
    Returns:
        the path string S:
    Example: SBpath(3/8) -> 'LLRL'
    """
    if type(frac) is tuple:
        num,den = frac
    else:
        frac = Fraction(frac)
        num,den = frac.numerator,frac.denominator
    S = ''
    while num != den:
        if num > den:
            S += 'R'
            num -= den
        else:
            S += 'L'
            den -= num
    return S    

# %%
print(SBpath(3/8) == CWpath(3/8)[::-1] == 'LLRL')

# %% [markdown]
# # Approximation d'un nombre réel  par une fraction  
# Nous pouvons utiliser ce dernier algorithme pour obtenir une approximation d'un nombre réel $x$ par un chemin de l'arbre de Stern-Brocot de longueur $n$ et la fraction correspondante:

# %%
def SBrealpath(x: float, n: int) -> str:
    """ return a path string of length n as an approximation of infinite path string of x
    
    Args:
        x: (float) the float representation of the real number x
        n: (int) length of the desired path string
    Returns:
        a path string of length n
    """
    S = ''
    for k in range(n):
        if x < 1:
            S += 'L'
            x = x/(1-x)
        else:
            S += 'R'
            x = x - 1
    return S

# %% [markdown]
# Par exemple une approximation du nombre d'Euler $e = 2.718281828459045\ldots$par un chemin allant au $20^{ième}$ niveau nous donne une approximation de  $e$ par la fraction 2721/1001 valide sur 6 décimales: 

# %%
SB_e20 = SBrealpath(e,20)

# %%
SB_e20

# %%
frac_e20 = SBfrac(SB_e20)
print('{} ~ {}'.format(frac_e20, frac_e20.numerator/frac_e20.denominator))

# %% [markdown]
# On peut aussi sur ce modèle utiliser `SBfrac` et définir une nouvelle fonction `SBrealfrac(x,n)` pour obtenir une suite de fractions approximant un nombre réel `x`, où `n` peut être un nombre entier  auquel cas la fonction retourne la suite des fractions correspondant à tous les chemins de longueur 1 jusqu'à `n` ou bien un 'slice Python' comme `slice(12,41,2)`, ce qui signifie que la fonction retourne seulement les fractions correspondant aux chemins de 12 jusqu'à 40 caractères inclus par pas de 2, c'est à dire les chemins dont le nombre de caractères est dans la liste `[12,14,16,...,40]`:

# %%
def SBrealfrac(x: float, n: Union[int, slice]) -> Tuple[int, int]:  
    # -> Fraction[int, int] type hint doesn't recognize Fraction:
    """ return a list of n successive fractions approximating the real number x
    
    Args:
        x: (float) the float representation of the real number x
        n: (int) number of successive rational approximations of x
           or a slice defining the subset of desired rational approximations of x
    Returns:
        a list of n fractions approximating x
    """
    if isinstance(n, slice):
        SBpath = SBrealpath(x,n.stop)
        return [SBfrac(SBpath[:k]) for k in range(n.stop)[n]]
    SBpath = SBrealpath(x,n)
    return [SBfrac(SBpath[:k]) for k in range(n)]



# %%
print(SBrealfrac(e,20))

# %%
print(SBrealfrac(e,slice(5,21,5)))

# %% [markdown]
# Ou une version retournant une liste 'pretty print' de chaînes de caractères formatés:

# %% {"code_folding": [0]}
def prettySBrealfrac(x, n: Union[int, slice] = 10, prec: str ='.10f') -> List[str]: 
    """ return a list of n successive fractions approximating the real number x
    
    Args:
        x: (float) the float representation of the real number x
        n: (int) number of successive rational approximations of x
           or a slice defining the subset of desired rational approximations of x
        prec: a string describing the desired float precision of the result
    Returns:
        a list of n formated strings each string describing an approximation of x and consists of:
        - position of each selected approximation in the list of max length (n.stop when n is a slice)
        - the fraction approximation
        - the float value of this fraction with the precision defined by prec
    """

    fracs = SBrealfrac(x,n)
    fmt = '{}:{}={:'+prec+'}'
    if isinstance(n, slice):
        return [fmt.format(n.start+n.step*k,frac,frac.numerator/frac.denominator) 
                for (k,frac) in enumerate(fracs)]
    return [fmt.format(k,frac,frac.numerator/frac.denominator) for (k,frac) in enumerate(fracs)]

# %%
print(prettySBrealfrac(e,slice(12,40,2)))

# %% [markdown]
# avec la même méthode, une suite d'approximations of $\pi$ = 3.141592653589793…

# %%
SB_pi =  SBrealpath(pi,400) 
SB_pi

# %%
print(prettySBrealfrac(pi,slice(21,401,20)))

# %% [markdown]
# # L'arbre de Stern-Brocot dans $\mathbb{N^2}$

# %% {"code_folding": [0]}
# click to read the code
def gridticks(ax,xmajticks=(0,11,5),xminticks=None,ymajticks=None,yminticks=None,alphamaj=0.5,alphamin=0.2):
    """ plot a grid: see matplotlib.axes.Axes.set_xticks,matplotlib.axes.Axes.grid
    """
    if ymajticks == None:
        ymajticks = xmajticks
    if yminticks == None:
        yminticks = xminticks
    xmajor_ticks = np.arange(*xmajticks)
    ymajor_ticks = np.arange(*ymajticks)
    ax.set_xticks(xmajor_ticks)
    ax.set_yticks(ymajor_ticks)
    if xminticks != None:
        xminor_ticks = np.arange(*xminticks)
        ax.set_xticks(xminor_ticks, minor=True)
        ax.grid(which='minor', alpha=alphamin)
    if yminticks != None:
        yminor_ticks = np.arange(*yminticks)
        ax.set_yticks(yminor_ticks, minor=True)
        ax.grid(which='minor', alpha=alphamin)
    ax.grid(which='major', alpha=alphamaj)

def plot_points(ax, pts: np.array, lw: float = 1.2, colors:List[str] = ['red'], marker: str = 'o') -> None:
    """ plot a list of points: see matplotlib plot, fig.add_subplot
    Args:
        ax: axes as result of fig.add_subplot(X,X,X)
        pts: a np.array of points (x:float,y:float)
        lw: (float) line width
        colors: a list of matplot_lib colors, here string of named colors, 
                coloring the points and cycling according to the length of colors
        marker: (str) a one char string, see matplotlib.markers
    Returns: None
    """
    ptsT = pts.T
    x_points, y_points = ptsT
    size = len(x_points)
    for i in range(size):
        ax.plot(x_points[i], y_points[i], color=colors[i%len(colors)], marker=marker)

def plot_pt2pts(ax, pt: Tuple[float,float], pts: np.array, 
                lw: float = 1.2, colors=['green'], ls: str = '-') -> None:
    """ Plot lines from pt to each point in pts
    
    Args: 
        ax: axes as result of fig.add_subplot(X,X,X)
        pt: (float,float) a point
        pts: a np.array of points (x:float,y:float)
        lw: (float) line width
        colors: a list of matplot_lib colors, here string of named colors, 
                coloring the points and cycling according to the length of colors
        ls: (str) linestyle  one or two chars string,
            https://matplotlib.org/1.5.3/api/pyplot_api.html#matplotlib.pyplot.plot
    """
    ptsT = pts.T
    x_points, y_points = ptsT
    size = len(x_points)
    for i in range(size):
        ax.plot([pt[0], x_points[i]], 
                 [pt[1], y_points[i]], 
                 color=colors[i%len(colors)], linestyle=ls, linewidth=lw)

# %% [markdown]
# Il est intéressant de remarquer que dans $\mathbb{N^2}$ les paires `(n,d)` correspondant aux fractions `n/d` de l'arbre de Stern-Brocot, que ces paires donc, forment elles aussi un arbre binaire.  
# Ce qui n'est pas le cas pour les paires `(n,d)` correspondant aux fractions `n/d` de l'arbre de Calkin-Wilf. Les arcs correspondants aux liens père-fils sont horizontaux ou verticaux.  
# Ci-dessous les sept premiers niveaux, avec les arcs de différentes couleurs dans les deux cas.

# %% {"code_folding": [0]}
#click to see  the code 
xmax = 21
plt.rcParams["figure.figsize"] =  [14.0, 6.0]
fig = plt.figure()
sub1 = fig.add_subplot(1,2, 1)
tree = SBpairs(7)
for k in range(0,len(tree)-1):
    level_colors=['blue','brown','green','navy','goldenrod','turquoise']
    for i in range(len(tree[k])):
        pt = np.array(tree[k][i])
        pts = np.array([tree[k+1][2*i],tree[k+1][2*i+1]])             
        plot_pt2pts(sub1,pt,pts,colors=[level_colors[k%len(level_colors)]])
        plot_points(sub1,pt.reshape(1,2))
        plot_points(sub1,pts)
gridticks(sub1,xmajticks=(0,xmax+1,1))
sub1.set_xlabel('Stern-Brocot: SBpairs(7)')

#plt.rcParams["figure.figsize"] =  [6.0, 14.0]
#fig = plt.figure()
sub4 = fig.add_subplot(1,2, 2)
tree = CWpairs(7)
for k in range(0,len(tree)-1):
    level_colors=['blue','brown','green','navy','goldenrod','turquoise']
    for i in range(len(tree[k])):
        pt = np.array(tree[k][i])
        pts = np.array([tree[k+1][2*i],tree[k+1][2*i+1]])             
        plot_pt2pts(sub4,pt,pts,colors=[level_colors[k%len(level_colors)]])
        plot_points(sub4,pt.reshape(1,2))
        plot_points(sub4,pts)    
gridticks(sub4,xmajticks=(0,xmax+1,1))
sub4.set_xlabel('Calkin-Wilf: CWpairs(7)')
plt.show()

# %% [markdown]
# Dans une grille de taille donnée, on peut placer tous les couples d'entiers copremiers présents dans cette grille
# We can put more nodes in the grid, plotting all the relatively prime integer's pairs present in the grid 

# %% {"code_folding": []}
def rel_prime(a: int, b: int) -> bool:
    """ return True if a and b are relatively prime
    
    Args:
        a: int
        b: int
    Returns:
        True or False
    """
    if a == b == 1:
        return False
    return gcd(a,b) == 1

# %%
rel_prime(3,8)

# %% {"code_folding": [0]}
#click to see the plot's code 
xmax = 22
plt.rcParams["figure.figsize"] =  [6.0, 6.0]
fig = plt.figure()
sub = fig.add_subplot(1,1, 1)
for lin in range(1,xmax+1):
    xpts = [col for col in range(1,xmax+1) if rel_prime(col,lin)]
    ypts = len(xpts)*[lin]
    for i in range(len(xpts)):
        sub.plot(xpts[i], ypts[i], color='red', marker='o')
gridticks(sub,xmajticks=(0,xmax+1,1))
sub.set_xlabel('Stern-Brocot: coprimes pairs')

# %% [markdown]
# et visualiser un plus grand nombre de branches dans la grille, en dessinant tous les arcs liant les noeud visibles 
# aux noeuds de leur père.  
# On définit facilement une fonction `SBfather(frac)` retournant la fraction que porte le père d'un noeud `(n,d)`:

# %% {"code_folding": [0]}
def SBfather(frac: Union[Tuple[int, int], str]) -> Tuple[int, int]: 
    # frac: Union[Tuple[int, int], Fraction[int, int], str] 
    # -> Union[Tuple[int, int], Fraction[int, int]] type hint doesn't recognize Fraction
    """ find the father node of a Stern-Brocot node from the pair or fraction value or string fraction value
    
    Args:
        frac: a fraction as a tuple (numerator: int, denominator: int) as (3,8) 
        or a string as '3/8' or a fraction as Fraction(3,8)
    Returns:
        the pair value of the father of frac if the frac parameter value was a pair
        the fraction value of the father of frac if the frac parameter value was a fraction or a string fraction
    """
    father = SBfrac(SBpath(frac)[:-1])
    if type(frac) is tuple:
        return father.numerator,father.denominator
    return father

# %%
SBfather((3,8))

# %%
SBfather('3/8')

# %% [markdown]
# Ou encore plus de branches en dessinant les liens reliant chaque noeud visible à ses fils gauche et droit. 
# or again with more branches plotting the branches linking a node to his left and right sons. 

# %% {"code_folding": [0]}
def SBsons(frac: Union[Tuple[int, int], str]) -> Tuple[Tuple[int, int],Tuple[int, int]]: 
    # frac: Union[Tuple[int, int], Fraction[int, int], str] 
    # -> Union[Tuple[int, int], Fraction[int, int]] type hint doesn't recognize Fraction
    """ find the (left son, right son) nodes pair of a Stern-Brocot tree node
    
    Args:
        frac: a fraction as a tuple (numerator: int, denominator: int) as (3,8) 
        or a string as '3/8' or a fraction as Fraction(3,8)
    Returns:
        a pair of pair's values for the sons of frac if the frac parameter value was a pair
        a pair of fraction values for the sons of frac if the frac parameter value was a fraction or a string
    """
    path = SBpath(frac)
    sons = [SBfrac(path + 'L'),SBfrac(path + 'R' )]
    if type(frac) is tuple:
        return ((sons[0].numerator,sons[0].denominator),(sons[1].numerator,sons[1].denominator))
    return sons

# %%
SBsons((2, 5))

# %% [markdown]
# Dans le cas où les noeuds ne sont pas dans le rectangle défini par la grille il faut couper les arcs correspondants (clipping).

# %% {"code_folding": [0]}
def xylinefrom(*pts: np.array) -> Tuple[Tuple[float,float],Tuple[float,float]]:
    """ compute 2 pairs (x0,x1),(y0,y1) to be plotted by matlib.plot
    
    Args:
        pts : a np.array of 2 or 3 points in the plane
        the 2 pts pts[0] and pts[1] verifying:
           0 < pts[0][0] < pts[1][0]
           0 < pts[0][1] < pts[1][1]
        if present, the third point defines a clipping
        window (0,xmax=pts[2][0]),(0,ymax=pts[2][1])
        to the segment defined by pts[0]to pts[1].
    Returns:
        two pairs (x0,x1),(y0,y1) defining the segment
        ready to be plotted by matlib.plot
    """
    xs = [pt[0] for pt in pts]
    ys = [pt[1] for pt in pts]
    if len(pts) == 2:
        return xs,ys
    else:
        m = (ys[1]-ys[0])/(xs[1]-xs[0])
        if xs[1] > xs[2]:
            xs[1] = xs[2]
            ys[1] = ys[0] + m*(xs[1]-xs[0])
        if ys[1] > ys[2]:
            ys[1] = ys[2]
            xs[1] = xs[0] + (xs[1]-xs[0])/m  
    return xs[:2],ys[:2]   

# %% {"code_folding": [0]}
#click to see the plot's code 
xmax = 22
plt.rcParams["figure.figsize"] =  [14.0, 6.0]
fig = plt.figure()

sub2 = fig.add_subplot(1,2, 1)
for lin in range(2,xmax+1): 
    xpts = [col for col in range(lin+1,xmax+1) if rel_prime(col,lin)]
    ypts = len(xpts)*[lin]
    for i in range(len(xpts)):
        xpair,ypair = np.array((SBfather((xpts[i],lin)),(xpts[i],lin))).T
        sub2.plot(xpair, ypair, color='green', linewidth=1.2)
        sub2.plot(ypair, xpair, color='green', linewidth=1.2)
for lin in range(1,xmax+1):
    xpts = [col for col in range(1,xmax+1) if rel_prime(col,lin)]
    ypts = len(xpts)*[lin]
    for i in range(len(xpts)):
        sub2.plot(xpts[i], ypts[i], color='red', marker='o')
gridticks(sub2,xmajticks=(0,xmax+1,1))
sub2.set_xlabel('Stern-Brocot: coprimes pairs and SBfathers')


#xmax = 22
#plt.rcParams["figure.figsize"] =  [6.0, 6.0]

sub3 = fig.add_subplot(1,2, 2)
for lin in range(1,xmax+1): 
    xpts = [col for col in range(lin+1,xmax+1) if rel_prime(col,lin)]
    ypts = len(xpts)*[lin]
    for i in range(len(xpts)):
        left,right = SBsons((xpts[i],lin))
        #xpair,ypair = pts2xys(SBfather((xpts[i],lin)),(xpts[i],lin))
        xpair,ypair = xylinefrom((xpts[i],lin),left,(xmax,xmax))
        sub3.plot(xpair, ypair, color='green', linewidth=1.2)
        sub3.plot(ypair, xpair, color='green', linewidth=1.2)
        xpair,ypair = xylinefrom((xpts[i],lin),right,(xmax,xmax))
        sub3.plot(xpair, ypair, color='green', linewidth=1.2)
        sub3.plot(ypair, xpair, color='green', linewidth=1.2)
for lin in range(1,xmax+1):
    xpts = [col for col in range(1,xmax+1) if rel_prime(col,lin)]
    ypts = len(xpts)*[lin]
    for i in range(len(xpts)):
        sub3.plot(xpts[i], ypts[i], color='red', marker='o')
gridticks(sub3,xmajticks=(0,xmax+1,1))
sub3.set_xlabel('Stern-Brocot: coprimes pairs and SBsons')
plt.show()

# %%

