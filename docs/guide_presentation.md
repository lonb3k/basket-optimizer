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
- `covariance.py` : échantillon, EWMA, Ledoit-Wolf (1/T, intensité analytique), avec une cible externe possible.
- `implied_corr.py` : prévisions de corrélation à partir du COR1M, covariance à niveau implicite, tests de prévision de risque.
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

## 5. La covariance qui écoute le marché des options (onglet "Forward-looking risk")

C'est le pont avec le projet dispersion. Là-bas on a montré que la corrélation implicite contient une
prime. Ici on se demande si, une fois corrigée de cette prime, elle aide à construire un portefeuille.

### 5.1 Le problème

Une matrice de covariance estimée sur l'année passée suppose que la corrélation du mois prochain
ressemblera à celle de l'an dernier. C'est faux juste avant les crises : la corrélation monte d'un coup,
et la matrice historique ne l'apprend qu'après. Le marché des options, lui, publie chaque jour son
anticipation, le COR1M.

### 5.2 Pourquoi on ne branche pas le COR1M tel quel

Deux raisons :
- **Il contient une prime.** Il est en moyenne 12 points au-dessus de la corrélation réalisée ensuite :
  c'est ce que paient les acheteurs de protection. Le brancher tel quel ferait voir du risque qui
  n'arrive jamais.
- **Ce n'est pas la même définition.** Le COR1M se définit par rapport à l'indice (vol du SPX contre vols
  des 50 titres). Une matrice de covariance a besoin de la corrélation moyenne entre paires de titres.

On le transforme donc d'abord en prévision. Chaque mois, une régression estimée uniquement sur les cinq
années précédentes relie le COR1M et la corrélation du dernier trimestre à la corrélation réalisée le
mois suivant. La régression corrige à la fois la prime et la différence de définition. Les cinq ans
glissants lui permettent de s'adapter au régime de corrélation très basse depuis 2024. Estimée sur tout
l'historique, elle prévoyait 20 % aujourd'hui contre 12,5 % avec cinq ans, alors que le réalisé tourne
autour de 7 %.

### 5.3 Première question : qui prévoit le mieux la corrélation ?

| Prévision | Erreur (RMSE, points) | Corrélation avec le réalisé |
|---|---|---|
| Dernière année, ce que suppose l'optimiseur | 17,3 | 0,33 |
| COR1M brut | 16,6 | 0,57 |
| COR1M + dernier trimestre, 5 ans glissants | 13,9 | 0,56 |

La corrélation de l'an dernier est la pire des sept prévisions testées. Le COR1M bouge avant que la
baisse n'apparaisse dans les rendements. Une fois corrigé, il réduit l'erreur d'environ un cinquième.

### 5.4 Deuxième question : est-ce que la covariance prévoit mieux le risque ?

On garde les mêmes vols et la même structure (quels titres bougent ensemble), et on ne change que le
niveau moyen de corrélation. Chaque mois, la matrice prévoit la vol d'un portefeuille, qu'on compare
ensuite à la vol réalisée le mois suivant. On mesure l'écart avec la QLIKE, la fonction de perte
standard pour les prévisions de variance : plus elle est basse, mieux c'est.

- **Panier équipondéré :** QLIKE 0,494 contre 0,575 pour Ledoit-Wolf, soit 14 % de mieux.
- **Variance minimale :** 0,552 contre 0,643, soit 14 % de mieux aussi.
- **Avec le COR1M brut, non corrigé :** 0,840, la pire prévision. C'est la preuve que l'étape de
  correction est indispensable.

### 5.5 Le résultat honnête : ça aide à mesurer le risque, pas à choisir les titres

Le portefeuille de variance minimale ne change presque pas : sa vol réalisée reste à 13,5 % et son Sharpe
passe de 1,06 à 1,09, ce qui n'est pas significatif. Ses poids dépendent de quels titres sont
*relativement* peu risqués, pas du niveau moyen de corrélation. Le marché des options te dit combien de
risque tu portes, pas quels titres porter.

La suite logique : utiliser cette prévision dans le vol targeting, là où le niveau de risque est
exactement ce qui compte. C'est noté dans le "Next" du README.

### 5.6 Questions probables

**"Pourquoi une régression et pas le COR1M directement ?"** À cause de la prime et de la différence de
définition. Le test le montre : le COR1M brut donne la pire prévision de risque.

**"Comment savez-vous qu'il n'y a pas de look-ahead ?"** La régression du mois k n'utilise que des mois
dont la fenêtre de réalisation est close à la date k. Un test modifie toutes les valeurs à partir du mois
k et vérifie que les prévisions jusqu'à k ne bougent pas.

**"Pourquoi QLIKE et pas l'erreur quadratique ?"** Parce que la QLIKE pénalise davantage la sous-estimation
du risque, celle qui coûte cher, et qu'elle reste fiable quand la vol réalisée est bruitée. C'est la
perte de référence dans la littérature sur les prévisions de variance (Patton, 2011).

**"Pourquoi le min variance ne s'améliore-t-il pas ?"** Parce que décaler le niveau moyen de corrélation
change peu le classement relatif des titres. C'est un résultat, pas un échec : il dit où l'information
des options sert, et où elle ne sert pas.
