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
import sys

# Installation automatique pour JupyterLite
if 'pyodide' in sys.modules:
    try:
        import piplite
    except ImportError:
        pass

def tableurVersVariables(fichier, delimiter=','):
    tableau = genfromtxt(fichier, delimiter=delimiter, skip_header=0, names=True)
    for i in tableau.dtype.names:
        globals()[i] = tableau[i]

def Modele(expression, x, y, contraintes):
    modele = ExpressionModel(expression)
    parametres = modele.make_params()
    for i in parametres :
        modele.set_param_hint(i, value = 1)
    for j in contraintes :
        if j[0] in parametres :
            modele.set_param_hint(j[0], value = j[1], vary = j[2], min = j[3], max = j[4])
    parametres = modele.make_params()
    resultat = modele.fit(y, parametres, x = x)
    valeurs = ""
    for key in resultat.params:
        # On garde votre f-string préférée
        # On s'assure juste que stderr n'est pas None pour le formatage .2g
        if resultat.params[key].stderr is not None:
            valeurs += f"{key} = {resultat.params[key].value:.3g} ; incertitude : {resultat.params[key].stderr:.2g}\n"
        else:
            valeurs += f"{key} = {resultat.params[key].value:.3g} ; incertitude : ?\n"
    return(modele, resultat.params, valeurs, expression)

def Calcul_modele(abscisse_name, ordonnee_name, equation, debut, fin, debutCourbe, finCourbe, contraintes):
    ord_val = ordonnee_name
    eq_val = equation
    equation_mod = re.sub(r"\b"+abscisse_name+r"\b","x", equation)
    
    import __main__
    abscisse = getattr(__main__, abscisse_name)
    ordonnee = getattr(__main__, ordonnee_name)
    
    if debutCourbe == None :
        debutCourbe = min(abscisse)
    if finCourbe == None :
        finCourbe = max(abscisse)
    
    xMod = linspace(debutCourbe, finCourbe, 30)
    modele, parametres, valeurs, expression = Modele(equation_mod, abscisse[debut:fin], ordonnee[debut:fin], contraintes)
    expression = f"{ord_val} = {eq_val}"
    yMod = modele.eval(parametres, x = xMod)
    
    for key in parametres:
        setattr(__main__, key, parametres[key].value)
    
    return(xMod, yMod, expression, valeurs, abscisse, ordonnee, modele, parametres)
