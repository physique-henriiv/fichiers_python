import sys

# Installation automatique des modules sous JupyterLite (Pyodide)
if 'pyodide' in sys.modules:
    import piplite
    await piplite.install(['numpy', 'scipy', 'matplotlib', 'lmfit', 'pandas', 'seaborn', 'ipympl'])

from pylab import *
from scipy import interpolate
from scipy.optimize import curve_fit
from lmfit import minimize, Parameters, Parameter, report_fit
from lmfit.models import ExpressionModel
from time import *
import re
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output

# Accès à l'espace de noms du notebook
main = sys.modules['__main__']

def tableurVersVariables(fichier, delimiter=','):
    """Importe un CSV et crée les variables directement dans le notebook."""
    tableau = np.genfromtxt(fichier, delimiter=delimiter, skip_header=0, names=True)
    for i in tableau.dtype.names:
        setattr(main, i, tableau[i])

def Modele(expression, x, y, contraintes):
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
    expression = f"{ord_val} = {eq_val}"
    yMod = modele.eval(parametres, x=xMod)
    
    for key in parametres:
        setattr(main, key, parametres[key].value)
    
    return (xMod, yMod, expression, valeurs, abscisse, ordonnee, modele, parametres)
