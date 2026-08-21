# SPEC-02 — Plan final consolidé de la session A40

**État : plan validé. Le preflight affiche une estimation et la session
enchaîne automatiquement, sans plafond horaire ni coupe-circuit budgétaire.**

## Décisions verrouillées

- Option B : calibration et gate CI sur **8 frames** de décodage.
- Deux prompts par classe pour les prefills ; un prompt épinglé par classe pour
  les chemins de décodage.
- Sonde d'accumulation obligatoire : **64 frames**, logits seulement,
  `short_error_assertion` et `medium_cache_regression`.
- Recalcul complet absent en classe longue ; segmentations longues limitées à
  médiane et quarts.
- Trois répétitions pour construire une tolérance ; deux mesures exactes pour
  les gates qui ne construisent pas de tolérance.
- Six chauffes par ensemble de formes exactes et par processus, jamais de
  capture d'état pendant une chauffe.
- Écriture atomique après chaque cas ; reprise uniquement à hashes de protocole
  identiques.

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

| Ordre | Poste | Forwards | Transfert mesuré | Équiv. historique | Durée après preflight théorique |
|---:|---|---:|---:|---:|---|
| 1 | Preflight : 18 chemins × 1 dry + 2 chronométrés | 207 | 0 | 0,18 h | `T_preflight` observé |
| 2 | Gate d'inertie du traceur, corpus complet | 60 | 1,82 GiB | 0,05 h | `E_trace` |
| 3 | Continuité legacy exacte, horizon 8 | 480 | 0,09 GiB | 0,41 h | `E_legacy` |
| 4 | Plancher de bruit, 3 prompts, horizon 8 | 576 | 0,20 GiB | 0,49 h | `E_noise` |
| 5 | Snapshot/restore réel, continuité horizon 8 | 48 | 3,61 GiB | 0,04 h | `E_snapshot` |
| 6 | Continuations de référence, 3 prompts × 3 seeds | 72 | négligeable | 0,06 h | `E_continuations` |
| 7 | Classe courte | 552 | 18,10 GiB | 0,47 h | `E_short` |
| 8 | Classe moyenne | 552 | 29,35 GiB | 0,47 h | `E_medium` |
| 9 | Classe longue | 312 | 10,90 GiB | 0,27 h | `E_long` |
| 10 | Sonde d'accumulation 64, court + moyen | 1 280 | 0,24 GiB | 1,09 h | `E_probe64` |
| | **Total** | **4 139** | **64,30 GiB** | **3,54 h non calibrées** | `T_preflight + ΣE` |

Le preflight couvre les chemins complets, capture désactivée :

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

| Classe | Chemin | Segment | Formes exactes | Forwards | Transfert |
|---|---|---|---|---:|---:|
| court | prefill | complet | `26`; `25` | 24 | 0,96 GiB |
| court | prefill | précoce | `1/25`; `1/24` | 48 | 1,86 GiB |
| court | prefill | médiane | `13/13`; `12/13` | 48 | 1,86 GiB |
| court | prefill | tardive | `25/1`; `24/1` | 48 | 1,87 GiB |
| court | prefill | quarts | `7/7/7/5`; `7/7/7/4` | 96 | 3,67 GiB |
| court | decode cached | — | `26 + 7×1` | 144 | 7,38 GiB |
| court | decode recompute | — | `26…33` | 144 | 0,50 GiB |
| moyen | prefill | complet | `310`; `331` | 24 | 1,75 GiB |
| moyen | prefill | précoce | `1/309`; `1/330` | 48 | 2,65 GiB |
| moyen | prefill | médiane | `155/155`; `165/166` | 48 | 2,76 GiB |
| moyen | prefill | tardive | `309/1`; `330/1` | 48 | 2,88 GiB |
| moyen | prefill | quarts | `78/78/78/76`; `83/83/83/82` | 96 | 4,79 GiB |
| moyen | decode cached | — | `310 + 7×1` | 144 | 9,60 GiB |
| moyen | decode recompute | — | `310…317` | 144 | 4,92 GiB |
| long | prefill | complet | `2 437`; `2 542` | 24 | 2,72 GiB |
| long | prefill | médiane | `1 218/1 219`; `1 271/1 271` | 48 | 2,72 GiB |
| long | prefill | quarts | `610/610/610/607`; `636/636/636/634` | 96 | 2,73 GiB |
| long | decode cached | — | `2 437 + 7×1` | 144 | 2,72 GiB |
| | **Total calibration** | | | **1 416** | **58,35 GiB** |

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
2. La continuité legacy reste exacte et hors tolérances.
3. Le plancher de bruit conserve les trois couples référence/référence,
   runner/runner et runner/référence, logits seulement.
4. Snapshot/restore capture `audit_echo` en continu et via snapshot à mi-chemin.
   Les métriques brutes sont écrites avant les classes ; après création des
   tolérances, elles sont adjudicées sans nouveau forward. Instabilité ou erreur
   structurelle arrête immédiatement.
5. Les classes écrivent un artefact atomique par `(prompt, chemin,
   segmentation, répétition)`.
6. La sonde 64 écrit ses huit étapes de calibration et les 56 étapes
   supplémentaires séparément. Une croissance est rapportée telle quelle,
   sans conclusion causale et sans élargissement silencieux des tolérances.

La sonde conserve ses six chauffes sur le chemin complet de 64 frames. Un
chemin 8 frames ne chauffe que les caches jusqu'à `n+6`; les frames 9 à 64
introduisent les longueurs `n+7` à `n+62`, qui sont des formes d'entrée/cache
distinctes et peuvent sélectionner un autre chemin backend. Les réduire à 8
retirerait donc la stabilisation requise avant les deux mesures retenues.

Chaque manifeste de reprise épingle commit, config, corpus, tokenizer,
backbone, protocole et futur hash de `tolerances.json`. Après interruption, les
cas terminés restent valides ; un nouveau processus refait ses chauffes sans
capturer d'état.
