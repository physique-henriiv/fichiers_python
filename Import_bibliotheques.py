from numpy import *
from numpy.random import random, normal
from lmfit.models import ExpressionModel
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'iframe'


def tableurVersVariables(fichier, delimiter=','):
    """Importe un fichier CSV et crée une variable par colonne."""
    tableau = genfromtxt(fichier, delimiter=delimiter, names=True)
    for nom in tableau.dtype.names:
        globals()[nom] = tableau[nom]


def _barres_erreur(valeurs, incertitude):
    """Convertit une incertitude (nombre ou liste) en format Plotly."""
    if incertitude is None:
        return {}
    try:
        arr = float(incertitude) * ones(len(valeurs))
    except (TypeError, ValueError):
        arr = array(incertitude)
    return dict(array=arr)


def Modele(expression, x, y, contraintes=[]):
    """Calcule un modèle par régression. Renvoie (modele, parametres, valeurs, expression)."""
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
    """Calcule le modèle et renvoie les données pour le tracé."""
    eq_val = equation
    equation = re.sub(r"\b" + abscisse_name + r"\b", "x", equation)
    abscisse = globals()[abscisse_name]
    ordonnee = globals()[ordonnee_name]
    if debutCourbe is None:
        debutCourbe = min(abscisse)
    if finCourbe is None:
        finCourbe = max(abscisse)
    xMod = linspace(debutCourbe, finCourbe, 100)
    modele, parametres, valeurs, expression = Modele(equation, abscisse[debut:fin], ordonnee[debut:fin], contraintes)
    yMod = modele.eval(parametres, x=xMod)
    for key in parametres:
        globals()[key] = parametres[key].value
    return (xMod, yMod, f"{ordonnee_name} = {eq_val}", valeurs, abscisse, ordonnee, modele, parametres)
