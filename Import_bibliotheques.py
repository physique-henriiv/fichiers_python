import sys
import numpy as np
import re
from time import *
from IPython.display import display, clear_output
import ipywidgets as widgets
from IPython import get_ipython

# Accès à l'espace de noms du notebook
main = sys.modules['__main__']

# Activation sécurisée des graphiques interactifs
ip = get_ipython()
if ip:
    try:
        # 'widget' est l'alias recommandé pour ipympl
        ip.run_line_magic('matplotlib', 'widget')
    except Exception:
        # Repli sur le mode standard en cas d'erreur (évite de bloquer le notebook)
        ip.run_line_magic('matplotlib', 'inline')
    
    from pylab import rcParams
    rcParams['figure.figsize'] = [16, 8]
    rcParams['font.size'] = 15
    rcParams['lines.markersize'] = 15
    rcParams['lines.markeredgewidth'] = 2

def tableurVersVariables(fichier, delimiter=','):
    """Importe un CSV et crée les variables directement dans le notebook."""
    tableau = np.genfromtxt(fichier, delimiter=delimiter, skip_header=0, names=True)
    for i in tableau.dtype.names:
        setattr(main, i, tableau[i])

def Modele(expression, x, y, contraintes):
    try:
        from lmfit.models import ExpressionModel
    except ImportError:
        print("Erreur : lmfit n'est pas encore prêt. Relancez la cellule.")
        return None
        
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
    res = Modele(equation_mod, abscisse[debut:fin], ordonnee[debut:fin], contraintes)
    if res is None: return None
    
    modele, parametres, valeurs, expression = res
    expression = f"{ord_val} = {eq_val}"
    yMod = modele.eval(parametres, x=xMod)
    
    for key in parametres:
        setattr(main, key, parametres[key].value)
    
    return (xMod, yMod, expression, valeurs, abscisse, ordonnee, modele, parametres)
