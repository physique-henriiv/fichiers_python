import sys
import numpy as np
import re
from time import *
from IPython.display import display, clear_output

# Import optionnel de ipywidgets pour éviter de bloquer le démarrage
try:
    import ipywidgets as widgets
except ImportError:
    widgets = None

import matplotlib.pyplot as plt
from pylab import *

# Réglages graphiques standards
rcParams['figure.figsize'] = [12, 6]
rcParams['font.size'] = 12

# Accès à l'espace de noms du notebook
main = sys.modules['__main__']

def tableurVersVariables(fichier, delimiter=','):
    tableau = np.genfromtxt(fichier, delimiter=delimiter, skip_header=0, names=True)
    for i in tableau.dtype.names:
        setattr(main, i, tableau[i])

def Modele(expression, x, y, contraintes):
    from lmfit.models import ExpressionModel
    modele = ExpressionModel(expression)
    parametres = modele.make_params()
    for i in parametres:
        modele.set_param_hint(i, value=1)
    for j in contraintes:
        if j[0] in parametres:
            modele.set_param_hint(j[0], value=j[1], vary=j[2], min=j[3], max=j[4])
    parametres = modele.make_params()
    resultat = modele.fit(y, parametres, x=x)
    valeurs = ""
    for key in resultat.params:
        if resultat.params[key].stderr is not None:
            valeurs += f"{key} = {resultat.params[key].value:.3g} ; incertitude : {resultat.params[key].stderr:.2g}\n"
        else:
            valeurs += f"{key} = {resultat.params[key].value:.3g} ; incertitude : ?\n"
    return (modele, resultat.params, valeurs, expression)

def Calcul_modele(abscisse_name, ordonnee_name, equation, debut, fin, debutCourbe, finCourbe, contraintes):
    ord_val = ordonnee_name
    eq_val = equation
    equation_mod = re.sub(r"\b" + abscisse_name + r"\b", "x", equation)
    abscisse = getattr(main, abscisse_name)
    ordonnee = getattr(main, ordonnee_name)
    if debutCourbe is None:
        debutCourbe = min(abscisse)
    if finCourbe is None:
        finCourbe = max(abscisse)
    xMod = np.linspace(debutCourbe, finCourbe, 30)
    modele, parametres, valeurs, expression = Modele(equation_mod, abscisse[debut:fin], ordonnee[debut:fin], contraintes)
    yMod = modele.eval(parametres, x=xMod)
    for key in parametres:
        setattr(main, key, parametres[key].value)
    return (xMod, yMod, f"{ord_val} = {eq_val}", valeurs, abscisse, ordonnee, modele, parametres)
