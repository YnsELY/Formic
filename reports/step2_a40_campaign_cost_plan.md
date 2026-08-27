# SPEC-02 — Plan final consolidé de la session A40 (protocole v3)

**État : plan révisé après les diagnostics A40 r3–r6, le crossover équilibré
r2 et le run `a40-2026-08-26-r1`
(`reports/step2_a40_run_2026-08-26_diagnostic.md`). Le preflight affiche une
estimation et la session enchaîne automatiquement, sans plafond horaire ni
coupe-circuit budgétaire.**

## Décisions verrouillées

- Option B : calibration et gate CI sur **8 frames** de décodage.
- Deux prompts par classe pour les prefills ; un prompt épinglé par classe pour
  les chemins de décodage.
- Sonde d'accumulation obligatoire : **64 frames**, logits seulement — profil
  de capture logits-only de bout en bout, `short_error_assertion` et
  `medium_cache_regression`.
- Recalcul complet absent en classe longue ; segmentations longues limitées à
  médiane et quarts.
- Trois répétitions pour construire une tolérance ; deux mesures exactes pour
  les gates qui ne construisent pas de tolérance.
- Six chauffes par ensemble de formes exactes et par processus, **un registre
  de chauffes par endpoint**, jamais de capture d'état pendant une chauffe.
- **Burn-in mesuré-jeté après chaque bloc de chauffe non vide** : 4
  paires-traces sur le chemin de mesure exact pour les gates par paires, 1
  répétition pour les mesures cross-path, l'inertie de trace et
  snapshot/restore ; enregistré dans l'artefact, exclu de tout critère
  bloquant et des statistiques de tolérance (fenêtre transitoire observée :
  2 paires-traces ; marge ×2).
- Écriture atomique après chaque cas ; reprise uniquement à hashes de protocole
  identiques.
- Identité wrapper : gate ABBA à quatre traitements RR/NN/RN/NR, stabilité des
  contrastes appariés ; les empreintes brutes prises à des ordinaux processus
  différents restent diagnostiques.
- Tolérances numériques : référence stock canonique contre chemin candidat
  Formic ; prefill segmenté contre prefixes complets et decode cached contre
  recalcul complet pour court/moyen.

ADR-0005 reste `PROPOSED` tant que les tolérances réelles ne sont pas mesurées,
mais les sous-décisions ci-dessus y sont enregistrées comme validées par Yanis.

## Corpus gelé

Le corpus canonique est `configs/reference_prompts.yaml`, schéma v2 :

```text
corpus_sha256 = 482e63d88a53d2850fe87db648f7d6fe2414ca5ee64b1a307de7cb3501c1f3c0
tokenizer revision = 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
change_policy = ADR_REQUIRED
```

Il contient le texte rendu exact, les IDs exacts et leurs hashes séparés. Les
longueurs sont `4, 5, 20, 24, 86, 86` pour la continuité legacy, puis
`26, 25, 310, 331, 2 437, 2 542` pour la calibration. Toute mutation échoue en
CI et invalide les verdicts antérieurs.

## Séquence finale et coût

L'« équivalent EXP-0008 » applique seulement l'ancre historique
`3,077994 s/forward` obtenue sur une forme cachée de quatre tokens. Ce n'est pas
la durée prévue de la classe ; après le preflight, cette colonne est remplacée
par les chronométrages réels.

Les forwards incluent les chauffes par endpoint et le burn-in mesuré-jeté ;
l'évidence admise par cas est inchangée par rapport au plan v2.

| Ordre | Poste | Forwards | Transfert mesuré | Équiv. historique | Durée après preflight théorique |
|---:|---|---:|---:|---:|---|
| 1 | Preflight : 18 candidats + 12 références distinctes, 1 dry + 2 chronométrés | 333 | 0 | 0,28 h | `T_preflight` observé |
| 2 | Gate d'inertie du traceur, corpus complet | 144 | 4,22 GiB | 0,12 h | `E_trace` |
| 3 | Continuité legacy exacte, Latin ABBA horizon 8 | 3 872 | 1,57 GiB | 3,31 h | `E_legacy` |
| 4 | Plancher alterné RR/NN/RN, 3 prompts, horizon 8 | 752 | 0,26 GiB | 0,64 h | `E_noise` |
| 5 | Snapshot/restore réel, continuité horizon 8 | 64 | 4,81 GiB | 0,05 h | `E_snapshot` |
| 6 | Continuations de référence, greedy + 3 seeds | 96 | négligeable | 0,08 h | `E_continuations` |
| 7 | Classe courte | 808 | 18,93 GiB | 0,69 h | `E_short` |
| 8 | Classe moyenne | 808 | 35,23 GiB | 0,69 h | `E_medium` |
| 9 | Classe longue | 488 | 18,87 GiB | 0,42 h | `E_long` |
| 10 | Sonde cached/recalcul 64, court + moyen | 2 560 | 0,47 GiB | 2,19 h | `E_probe64` |
| | **Total** | **9 925** | **84,36 GiB** | **8,49 h non calibrées** | `T_preflight + ΣE` |

La sonde 64 tourne à trois répétitions mesurées comme toutes les autres
mesures (burn-in + rép 0 couvrent la fenêtre transitoire de deux traces ;
l'assertion last-two compare les répétitions 1 et 2). Les artefacts dérivés
(mesures brutes, tolérances candidates, adjudication snapshot, verdict
candidat) sont écrits dès que leurs données existent, avant la sonde ; les
gates finales sont jugées ensemble en fin de session.

Le preflight couvre les chemins candidats et leurs références canoniques
distinctes, capture désactivée :

| Classe | Chemins chronométrés | Frames par trio dry/mesures | Forwards |
|---|---|---:|---:|
| courte | full, 4 segmentations, cached-8, recompute-8 | `3 × (1+2+2+2+4+8+8)` | 81 |
| moyenne | full, 4 segmentations, cached-8, recompute-8 | `3 × (1+2+2+2+4+8+8)` | 81 |
| longue | full, médiane, quarts, cached-8 | `3 × (1+2+4+8)` | 45 |
| **Total** | 18 chemins | | **207** |

Pour chaque chemin, le plus lent des deux passages chronométrés alimente les
estimations `E_*`. Le temps de chargement, l'inventaire A12 et le hash streaming
font partie de `T_preflight`. Le preflight mesure aussi le débit de copie
GPU→CPU sur des buffers représentatifs, sans conserver d'état d'identité. Les
coûts de capture utilisent ce débit et le volume réel du poste ; la sonde 64 est
extrapolée depuis les temps par frame cached mesurés, puis couverte par la marge
globale de 30 %.

## Détail de la calibration principale

Forwards par cas = chauffes des deux endpoints + burn-in jeté + mesures ; les
transferts comptent le burn-in (capture réelle).

| Classe | Chemin | Segment | Formes exactes | Forwards | Transfert |
|---|---|---|---|---:|---:|
| court | prefill | complet | `26`; `25` | 40 | 1,28 GiB |
| court | prefill | précoce | `1/25`; `1/24` | 80 | 2,48 GiB |
| court | prefill | médiane | `13/13`; `12/13` | 80 | 2,50 GiB |
| court | prefill | tardive | `25/1`; `24/1` | 80 | 2,53 GiB |
| court | prefill | quarts | `7/7/7/5`; `7/7/7/4` | 160 | 4,95 GiB |
| court | decode cached | — | `26 + 7×1` vs `26…33` | 160 | 4,60 GiB |
| court | decode recompute | — | `26…33` | 208 | 0,59 GiB |
| moyen | prefill | complet | `310`; `331` | 40 | 2,34 GiB |
| moyen | prefill | précoce | `1/309`; `1/330` | 80 | 3,53 GiB |
| moyen | prefill | médiane | `155/155`; `165/166` | 80 | 3,89 GiB |
| moyen | prefill | tardive | `309/1`; `330/1` | 80 | 4,25 GiB |
| moyen | prefill | quarts | `78/78/78/76`; `83/83/83/82` | 160 | 7,01 GiB |
| moyen | decode cached | — | `310 + 7×1` vs `310…317` | 160 | 8,47 GiB |
| moyen | decode recompute | — | `310…317` | 208 | 5,74 GiB |
| long | prefill | complet | `2 437`; `2 542` | 40 | 3,62 GiB |
| long | prefill | médiane | `1 218/1 219`; `1 271/1 271` | 80 | 4,83 GiB |
| long | prefill | quarts | `610/610/610/607`; `636/636/636/634` | 160 | 7,24 GiB |
| long | decode cached | — | `2 437 + 7×1` | 208 | 3,18 GiB |
| | **Total calibration** | | | **2 104** | **73,03 GiB** |

## Estimation automatique après preflight

Le preflight écrit un JSON strict avec son temps écoulé, le temps de chargement
réel et les estimations des neuf postes restants. Le rapporteur calcule :

```text
total_estimated_hours = (T_preflight + ΣE_poste) / 3600
```

La commande est :

```text
python scripts/step2_budget_gate.py \
  --preflight artifacts/step2/preflight/estimate.json
```

Le plan de forwards est codé en dur. Le rapport détaille le nombre de processus
modèle, le chargement, le preflight, chaque poste restant et le total. Il ne
reçoit plus de budget, n'émet pas de verdict GO/NO-GO et retourne toujours 0 :
l'estimation est informative et la session enchaîne automatiquement.

La session est un seul processus et un seul chargement complet du modèle. Le
temps affiché est celui réellement mesuré dans `model_load_seconds`; l'historique
A40 disponible situe ce chargement entre environ 205 et 366 secondes, sans en
faire une promesse pour le pod courant.

## Gates et reprise

1. La gate d'inertie compare trace OFF/ON sur les douze prompts. Le chemin est
   `None` par défaut dans `runner.py`. Premier delta : arrêt immédiat.
2. La continuité legacy fait tourner les quatre traitements RR/NN/RN/NR dans
   chacun des quatre slots de configuration ; les contrastes référence/runner
   doivent être exacts et leurs deux signatures mesurées identiques.
3. Le plancher de bruit utilise le calendrier alterné r6 et conserve RR et NN
   comme contrôles bruts, logits seulement. Le maximum RR est propagé dans
   chaque ligne logits candidate et interdit une tolérance plus serrée que la
   référence. L'assertion last-two bloquante porte sur RR et NN (les paires
   qui produisent le plancher) ; la paire mixte RN reste mesurée mais est un
   diagnostic non bloquant (oscillation de période 2 mesurée par
   `a40-2026-08-27-r1`).
4. Snapshot/restore capture `audit_echo` en continu et via snapshot à mi-chemin.
   Les métriques brutes sont écrites avant les classes ; après création des
   tolérances, elles sont adjudicées sans nouveau forward. Instabilité ou erreur
   structurelle arrête immédiatement.
5. Les classes écrivent un artefact atomique par `(prompt, chemin,
   segmentation, répétition)`.
6. La sonde 64 compare recalcul complet et cached, puis écrit ses huit étapes
   de calibration et les 56 étapes
   supplémentaires séparément. Une croissance est rapportée telle quelle,
   sans conclusion causale et sans élargissement silencieux des tolérances.

La sonde conserve ses six chauffes sur le chemin complet de 64 frames. Un
chemin 8 frames ne chauffe que les caches jusqu'à `n+6`; les frames 9 à 64
introduisent les longueurs `n+7` à `n+62`, qui sont des formes d'entrée/cache
distinctes et peuvent sélectionner un autre chemin backend. Les réduire à 8
retirerait donc la stabilisation requise avant les deux mesures retenues.

Chaque bloc de chauffe non vide est suivi de son burn-in mesuré-jeté (voir
« Décisions verrouillées ») avant la première trace admise ; l'évidence du
burn-in reste dans l'artefact sous `burn_in` et n'entre dans aucun critère.

Chaque manifeste de reprise épingle commit, config, corpus, tokenizer,
backbone, protocole et futur hash de `tolerances.json`. Après interruption, les
cas terminés restent valides ; un nouveau processus refait ses chauffes sans
capturer d'état, puis leurs burn-ins. Le passage au protocole v3 change le hash
de config résolue : les runs antérieurs ne sont pas reprenables, la prochaine
session exige un nouveau run-id.
