# Préconisations pour la suite — Défi IA 2021

Document rédigé en français, à l'inverse du reste de `reports/` : c'est une note
de décision destinée au propriétaire du projet, pas un journal technique.

Écrit le 2026-07-19, après la première soumission réelle sur Kaggle.

---

## 1. Où on en est, avec un vrai chiffre

| | Macro-F1 |
|---|---:|
| **Notre soumission (public)** | **0,82166** |
| notre soumission (seconde colonne affichée) | 0,82403 |
| 3ᵉ du classement — Cameroun ENSPY, Fred | 0,82733 |
| 2ᵉ — France UJM, BravoNils | 0,82932 |
| 1ᵉʳ — France UR2, WeTried | 0,84247 |

**Écart au podium : 0,0057. Écart au premier : 0,0208.**

Ce n'est pas « loin ». C'est un écart du même ordre que les gains que produisent
les techniques listées en §3, dont plusieurs n'ont jamais été essayées ici.

Point de départ de la session, pour situer le chemin parcouru : 0,7643
(modèle classique) et 0,8035 (référence transformeur héritée).

## 2. La leçon la plus importante : notre estimation hors-ligne ment de 1,1 point

| | valeur |
|---|---:|
| estimation sur notre jeu de validation | 0,8329 |
| score public réel | 0,82166 |
| **surestimation** | **+0,0112** |

C'est un écart considérable : il vaut **le double** de la distance qui nous
sépare du podium. Toute décision prise sur la seule foi d'un chiffre hors-ligne
est donc suspecte.

**Trois causes plausibles, par ordre de probabilité :**

1. **Les seuils par classe ne se transportent pas entièrement.** Ils sont réglés
   sur nos 32 580 lignes de validation ; le jeu de test en compte 54 300, tirées
   d'une distribution voisine mais pas identique. Le gain mesuré (+0,019) est
   probablement plus faible en réalité.
2. **Le biais de sélection du choix final.** La configuration retenue avait
   gagné une comparaison à quatre options sur les mêmes lignes qui la notaient.
   Je l'avais signalé, sans le chiffrer.
3. **Un seul découpage, un seul seed.** Tous les chiffres transformeur reposent
   sur un unique partage entraînement/validation.

**Ce qu'il faut en faire — action n° 0, gratuite et immédiate :**

Soumettre `roberta_large_6ep_ensemble.csv` (estimation 0,8288) **et** une version
sans aucun post-traitement. En comparant les trois scores publics, on saura
lequel des trois facteurs domine. Sans cette mesure, on optimise à l'aveugle.

C'est un usage rentable des soumissions quotidiennes : les équipes de tête en ont
utilisé 9, 44 et 14. Une soumission est une mesure réelle, elle vaut plus que
n'importe quelle estimation.

## 3. Les leviers à activer, classés par rapport gain/coût

Durées mesurées sur la RTX 4060 Ti de la machine : un roberta-large complet
(6 époques + inférence) prend **environ 3 h**.

### Niveau 1 — fiable, à faire en premier

**A. Moyenner plusieurs graines aléatoires du même modèle** — *3 × 3 h*
Entraîner roberta-large trois fois avec des graines différentes et moyenner
leurs scores. C'est la technique la plus régulière des compétitions : chaque
modèle se trompe différemment, la moyenne annule une partie du bruit. Gain
habituel **+0,005 à +0,015**, soit à lui seul de quoi viser le podium.
Bénéfice secondaire : on obtient enfin une barre d'erreur sur nos chiffres,
ce qui manque cruellement aujourd'hui.

**B. Mélanger des architectures différentes** — *3 à 6 h*
Moyenner roberta-large avec un modèle d'une autre famille (DeBERTa-v3-large,
ELECTRA-large) rapporte davantage que moyenner des graines, parce que les
erreurs sont encore moins corrélées. Gain habituel **+0,010 à +0,020**.
Prérequis : DeBERTa-v3 n'a jamais tourné ici. La question ouverte du ROADMAP —
est-ce que le format bf16, absent de la carte Kaggle mais présent sur la 4060 Ti,
stabilise son entraînement ? — n'a toujours pas de réponse. Une sonde de 2 h
suffit à trancher.

**C. Ré-étiqueter le jeu de test et réentraîner dessus** *(pseudo-labeling)* — *4 h*
Le jeu de test fait 54 300 biographies, soit **25 % de données en plus** que
l'entraînement, et il est fourni. On prédit ses étiquettes, on garde celles où
le modèle est très sûr, on les ajoute à l'entraînement, on réentraîne. C'est
autorisé par le règlement. Gain habituel **+0,005 à +0,010**.
Piège à éviter : ne garder que les prédictions confiantes renforce les biais du
modèle sur les classes rares. Filtrer par classe, pas globalement.

### Niveau 2 — moins cher, gain plus modeste

**D. Allonger le texte lu par le modèle** — *3 h*
Il ne lit aujourd'hui que 192 unités de texte. Le 95ᵉ centile des biographies
tourne autour de 150, donc la troncature touche peu de monde — mais elle touche
peut-être les biographies longues, souvent les plus informatives. Tester 256.
Gain estimé **+0,002 à +0,005**.

**E. Régler la pondération des classes** — *3 h*
Question ouverte jamais tranchée : faut-il pondérer les classes par l'inverse
exact de leur fréquence, par sa racine carrée, ou pas du tout ? Le Macro-F1
compte les 28 métiers à poids égal, donc ce réglage compte. Gain estimé
**+0,003 à +0,008**.

**F. Perturbation adverse pendant l'entraînement** *(FGM)* — *4 h*
On perturbe légèrement la représentation interne des mots à chaque étape pour
forcer le modèle à ne pas dépendre de détails fragiles. Classique en compétition,
gain habituel **+0,003 à +0,005**.

### Niveau 3 — plus lourd, à garder pour la fin

**G. Empilement par validation croisée** *(stacking k-fold)* — *15 h*
Découper l'entraînement en 5 parts, entraîner 5 modèles, et apprendre un
méta-modèle sur leurs prédictions croisées. C'est la version aboutie du
mélange, et accessoirement le seul protocole qui donnerait une estimation
hors-ligne fiable — ce qui règlerait le problème du §2. Cher mais complet.

## 4. Ce qu'il ne faut PAS refaire — c'est mesuré

Ces pistes ont été testées sérieusement et ne rapportent rien. Les rouvrir
coûterait du temps pour zéro gain.

| piste | verdict mesuré |
|---|---|
| Réglage des hyperparamètres du modèle classique | 11 configurations, **aucune** ne bat le défaut |
| Masquage des prénoms par reconnaissance d'entités | **aucun effet** (−0,012 ± 0,073 sur le DI, 3 seeds) |
| Augmenter artificiellement les classes rares | **dégrade** (−0,003), et les classes rares ne sont pas le goulot |
| Empiler seuils + mélange naïvement | **0,8295 contre 0,8329** pour les seuils seuls, deux fois de suite |
| Calage de seuils sur le modèle classique | gain **nul** (entre −0,003 et +0,003) |

⚠️ Nuance importante sur la dernière ligne : le calage de seuils ne vaut rien sur
le modèle classique mais **+0,019 sur le transformeur**. C'est l'illustration la
plus nette d'une règle générale : **une technique doit être mesurée sur le modèle
qui part en soumission, jamais sur un substitut bon marché.**

## 5. Où le modèle perd encore des points

D'après `reports/error_analysis.md` et `reports/per_class_comparison.md` :

Le Macro-F1 se perd sur la **confusion entre métiers voisins**, pas sur les
classes rares. `professor` (32 % des données) déborde sur `teacher`,
`psychologist`, `physician` ; `architect` et `software_engineer` s'échangent ;
`nurse` glisse vers `physician`. Les métiers les plus rares — `rapper`, `dj`,
`yoga_teacher` — sont parmi les mieux classés.

Deux conséquences pour la suite :

- Si on refait de l'augmentation de données, il faut viser **ces paires-là**,
  pas les classes peu nombreuses.
- Le modèle classique **bat encore** roberta-large sur `surgeon`, `professor`,
  `dentist` et `physician`. Cette complémentarité est réelle et sous-exploitée :
  le mélange actuel utilise un poids global unique, alors qu'un poids **par
  classe** exploiterait exactement cette structure. Piste peu coûteuse et, à ma
  connaissance, jamais tentée ici.

## 6. La piste équité

| soumission | Macro-F1 | DI |
|---|---:|---:|
| `roberta_counterfactual_fairness.csv` | 0,8018 | 3,41 |
| `classical_counterfactual_fairness.csv` | 0,7522 | 3,28 |

L'entraînement contrefactuel — montrer au modèle chaque biographie avec les
genres inversés — coûte **0,0009 de Macro-F1 pour 0,70 de DI** sur le
transformeur. C'est quasi gratuit.

**Action évidente jamais faite : l'appliquer à roberta-large.** Sur roberta-base
il ne coûte rien ; sur large il donnerait vraisemblablement une soumission
équité **au-dessus de 0,82**, c'est-à-dire compétitive sur les deux axes à la
fois. Coût : 3 h.

⚠️ **Piège à ne jamais oublier** : la métrique d'équité note un métier prédit
pour un seul genre comme *parité parfaite*. Retirer toutes les femmes d'un
métier **améliore** le score. Toute optimisation visant le DI trouvera cette
faille. Publier systématiquement `count_single_gender_jobs` à côté de tout DI.

## 7. Le plan que je recommande

**Étape 1 — aujourd'hui, gratuit.** Soumettre les deux autres fichiers déjà
prêts pour mesurer d'où vient l'écart de 1,1 point du §2. Sans ça, on optimise
à l'aveugle.

**Étape 2 — une nuit, ~9 h.** Trois roberta-large avec des graines différentes,
moyennés (levier A). C'est le gain le plus sûr et il donne enfin des barres
d'erreur.

**Étape 3 — une nuit, ~6 h.** Sonde DeBERTa-v3 en bf16 (2 h) ; si elle tient,
un entraînement complet à ajouter au mélange (levier B).

**Étape 4 — une nuit, ~4 h.** Ré-étiquetage du jeu de test (levier C).

Cumulés, les leviers A à C valent en général **+0,02 à +0,04**. L'écart au
premier est de 0,021. Le podium est atteignable ; la première place est
plausible.

**Et en parallèle, gratuit :** roberta-large contrefactuel pour la piste équité
(§6), qui n'entre en concurrence avec rien puisqu'il vise l'autre classement.

## 8. Les règles de méthode à garder

Établies en mesurant, souvent en se trompant d'abord. Elles ont chacune corrigé
un chiffre faux sur ce projet.

1. **Tout ce qui est réglé doit être noté sur des données qui ne l'ont pas
   réglé.** Trois chiffres publiés ici étaient mesurés sur leurs propres données
   de calage — le gain des seuils était surestimé d'un facteur 2,2.
2. **Une technique se mesure sur le modèle qui part en soumission**, pas sur un
   substitut. Les seuils : zéro sur le classique, +0,019 sur le transformeur.
3. **Jamais lire une époque sans son calendrier de taux d'apprentissage.** Un
   palier en milieu d'entraînement n'est pas une convergence : roberta-large
   semblait plafonné à l'époque 4, il a gagné 1,8 point à l'époque 5. Vérifier
   avec `scripts/check_convergence.py`.
4. **Un test à blanc (`--smoke`) avant tout run long.** Il a intercepté un bug
   qui aurait fait planter l'entraînement GPU *après* plusieurs heures, au
   moment de sauvegarder.
5. **Après avoir tué un processus GPU, vérifier que la mémoire vidéo est
   réellement rendue.** Une allocation fantôme de 7 Go a coûté un facteur 2,9 de
   ralentissement et ressemblait exactement à un problème de modèle.
6. **Une soumission réelle vaut mieux que n'importe quelle estimation.** L'écart
   de 1,1 point du §2 en est la démonstration.
