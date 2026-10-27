# Guide de présentation : basket optimizer

## 1. La thèse
La question n'est pas "quel optimiseur est le meilleur" mais "quelle décision déplace vraiment
le Sharpe". Réponse mesurée : le vol targeting (+0,2 de Sharpe, drawdown divisé par deux)
compte plus que le choix des poids (écart de 0,4 entre six schémas), et un plafond sur les
Mag 7 ne coûte rien aux schémas fondés sur le risque parce qu'ils les sous-pondèrent déjà.

## 2. Les concepts

### Pourquoi pas de rendements attendus
Les erreurs d'estimation sur les rendements font exploser les poids (Michaud 1989). Les six
schémas n'utilisent que la covariance : equal weight, inverse vol, min variance, risk parity
(ERC), max diversification (Choueifaty 2008), HRP (Lopez de Prado 2016).

### Poids ≠ risque
La contribution au risque d'un nom est wᵢ (Σw)ᵢ / (w'Σw). 10 % dans un nom à 50 de vol
corrélé à tout, c'est un pari bien plus gros que 10 % dans un staple à 15 de vol. Le
**nombre effectif de paris** est l'entropie des contributions : 30 pour l'equal weight, 48 pour
l'ERC, moins de 10 pour la cap-pondération.

### La covariance est le produit
50 noms sur 252 jours = 1 275 paramètres pour 252 observations. Nombre de conditionnement de la
matrice échantillon : plusieurs milliers. **Ledoit-Wolf** : on tire la matrice vers une cible à
corrélation constante avec une intensité calculée analytiquement (ici 0,06). C'est pour ça que
le min variance tient 30 noms et pas 5.

### Vol targeting
Levier = cible / prévision de vol, plafonné. La vol s'agglutine et monte avant les pires
rendements : se désendetter quand la vol réalisée monte évite une partie de chaque krach,
se réendetter dans le calme récolte la tendance. Version VIX-scaled : réalisée × VIX /
réalisée de l'indice, réagit plus vite. Les changements de levier sont facturés.

### Survivorship, dit d'abord
Le top 50 d'aujourd'hui, ce sont les gagnants. Tout panier bâti dessus bat SPY : 26 % par an
pour l'equal weight contre 15 % pour SPY. RSP, l'ETF S&P equal-weight, a fait 12 % : la
différence est le biais, mesuré. Les comparaisons entre schémas sont justes (même univers),
celles avec l'indice ne le sont pas.

## 3. Ce que fait le code
- `covariance.py` : échantillon, EWMA, Ledoit-Wolf (1/T, intensité analytique).
- `optimizers.py` : SLSQP pour min variance, ERC, max diversification avec bornes par nom et
  contraintes de groupe dans le solveur ; HRP par bissection récursive sur l'arbre de
  distances de corrélation, sans inversion de matrice.
- `risk.py` : contributions, ENB, ratio de diversification, clusters, tracking error.
- `backtest.py` : poids walk-forward (covariance sur la fenêtre glissante finissant la
  veille), simulation avec dérive des positions entre rebalancements et turnover facturé,
  overlay de vol targeting, stats.

## 4. Questions probables
**Pourquoi le min variance tourne-t-il à 20 % par mois ?** Il poursuit les noms les plus
calmes, qui changent ; HRP et ERC tournent à 7 %. Le coût est dans les chiffres.
**Pourquoi l'ERC par optimisation et HRP par récursion ?** Les deux visent un risque équilibré ;
HRP n'inverse jamais la covariance, donc reste stable quand elle est mal conditionnée.
**Que vaut le Sharpe de 1,5 ?** Rien en absolu (survivorship). La hiérarchie entre schémas et
l'apport du vol targeting sont ce qui est robuste.
**Comment enlever le biais ?** Avec les compositions historiques de l'indice, qui ne sont pas
gratuites. C'est la première ligne du "Next".
