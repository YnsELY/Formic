# SPEC-02 — Diagnostic du run `a40-2026-08-28-r1`

**Statut : FAIL analysé ; sept phases complètes, dont snapshot/restore validé
sur le vrai checkpoint ; les deux derniers verrous structurels sont désormais
mesurés et amendés (protocole v4). Aucune tolérance promue, aucun verdict
d'identité officiel.**

## Identité du run

| Champ | Valeur |
|---|---|
| Run | `artifacts/step2/runs/a40-2026-08-28-r1/` (commité par le pod, fe89e27) |
| Commit source | `9fd9bb7` (worktree propre), protocole `SPEC-02-h8-option-b-balanced-v3` |
| Durée | 2026-08-28, échec à 15:34:12Z (~5 h) |
| Mémoire | pic 39 968 172 544 o (37,2 GiB), 7,27 GiB libres à l'échec — aucun OOM |

## Ce qui est passé (7 phases, 60 cas commités)

`preflight`, `trace_inertness`, `legacy_continuity` (6/6), `noise_floor`
(3/3), `snapshot_restore`, `reference_continuations`, classe `short`
entière — puis 12 cas `medium` (les cinq prefills des deux prompts, et
`decode_recompute` greedy + échantillonné).

Deux risques résiduels listés au run précédent sont **levés** :

- **Plancher de bruit sur short et medium** (jamais mesurés sous calendrier
  alterné) : RR et NN stables sur les trois prompts. Planchers RR mesurés :
  `audit_echo` 16,09375 ; `short_error_assertion` **19,5625** ;
  `medium_cache_regression` 11,59375. La paire mixte RN, non bloquante depuis
  le correctif du 27/08, est stable sur short et medium et instable seulement
  sur `audit_echo` — cohérent avec l'observation précédente.
- **Snapshot/restore sur le vrai checkpoint** : PASS, avec une stabilité
  parfaite (les trois répétitions produisent des comparaisons identiques,
  `max_abs_delta` 12,40625 constant). La primitive fonctionne.

## L'échec

`calibration__medium_cache_regression__decode_cached__none__greedy` :
`InvalidMeasurement: last two measured traces are unstable;
first_changed_repetition=1; changed_side=candidate`.

Empreintes du cas (burn-in + 3 répétitions admises) :

| Exécution | Référence (recompute) | Candidat (cached) |
|---|---|---|
| burn-in | `d433a343…` | `25fe8e51…` |
| rép 0 | `d433a343…` | `d590149d…` |
| rép 1 | `d433a343…` | `39e10cd1…` |
| rép 2 | `d433a343…` | `c82b23c9…` |

**La référence est bit-stable sur quatre exécutions ; le candidat produit
quatre empreintes toutes différentes.** Localisation de la divergence
inter-répétitions (152 mesures par répétition) : steps 0 et 1 bit-identiques,
puis à partir du step 2 uniquement dans les groupes tardifs et les logits —
rép 1 → rép 2 : 33 mesures changent, réparties en `logits` (6), `POST_G16`
(6), `G15_G16` (6), `G13_G14` (5), `G14_G15` (5), `G12_G13` (3), `G11_G12`
(2). Aucun changement avant G11.

Ce n'est **ni un transitoire** (pas de convergence : le burn-in a déjà tourné
et les trois répétitions suivantes diffèrent encore), **ni une oscillation
périodique** (pas de motif A/B/A). C'est une variabilité continue du chemin
cached medium **sous capture complète des frontières**. Contrôles du même
run : le même chemin cached medium sans capture lourde (plancher de bruit,
logits-only) est stable ; le chemin cached **short** sous capture complète est
stable (3 répétitions identiques) ; le chemin **recompute** medium est stable.

## Le verrou suivant, désormais un fait mesuré

Sur des cas **stables et déjà commités**, l'accord top-1 référence-vs-candidat
vaut : **3/8** (short cached), **1/8** (medium cached), **2/8** (snapshot).
Le verdict candidat traitait tout désaccord top-1 en échec dur : aucune
relance ne pouvait donc jamais atteindre `CALIBRATION_COMPLETE`, quelle que
soit la correction en amont.

Contrôle décisif du même run : `medium_cache_regression / decode_recompute /
greedy` — c'est-à-dire Formic contre le stock sur le **même** chemin — est
**exact 8/8, delta 0,0**, empreintes identiques aux trois répétitions. Les
désaccords top-1 comparent donc cached contre recompute **sur le backend
stock** (la référence est du Hugging Face pur) : ils mesurent l'écart entre
deux chemins d'exécution du moteur GPU, pas un défaut du wrapper.

## Interprétation (étiquetée comme telle, sans attribution de cause racine)

Les deux critères en échec exigeaient d'un chemin candidat des propriétés que
le backend n'a pas, et que le protocole est précisément conçu pour mesurer :
la reproductibilité bit-à-bit d'un chemin dont on veut quantifier la
variabilité (via 3 répétitions et un seuil 2×max), et l'accord top-1 entre
deux chemins d'exécution différents (dont le plancher RR mesuré, jusqu'à
19,56, montre l'ampleur). Le mécanisme physique n'est pas attribué.

## Corrections apportées (protocole v4, commit local, tests hors GPU)

1. **Critère de stabilité des mesures de tolérance** : le critère bloquant
   devient « les empreintes de la référence canonique sont identiques sur
   toutes les répétitions » (`_assert_reference_stable`). La variabilité du
   candidat est enregistrée dans l'artefact
   (`stability.blocking_criterion`, `candidate_stability_is_diagnostic`,
   empreintes des deux côtés conservées). Les gates d'**exactitude** —
   inertie de trace, continuité legacy, plancher RR/NN, snapshot/restore —
   gardent la stricte assertion last-two des deux côtés.
2. **Top-1 cross-position** : compté et rapporté au lieu d'échouer, dans le
   verdict candidat (`top1_disagreements` : total, par cas, premier cas) et
   dans l'adjudication snapshot. Il reste bloquant dans `verdict.evaluate`
   (gate CI, protocole aligné), et **chaque ligne concernée reste
   `bounded`/`REVIEW_REQUIRED`** : la justification humaine à la promotion est
   inchangée. Le seuil delta planché de l'adjudication reste bloquant.
3. Budget **inchangé** : 9 925 forwards, ~8,5 h. Aucun forward ajouté.

## Vérification locale

- Suite weight-free complète et gardes A11/A12/inertes : vertes.
- `formic verify` : PASS ; `identity-check --toy` : PASS.
- Nouveaux tests : critère référence-stable (candidat variable accepté,
  référence dérivante refusée, y compris à travers `measure_forced`) ;
  `candidate_verdict` comptant les flips sans échouer ; adjudication snapshot
  enregistrant les flips tout en gardant le seuil delta bloquant ; un flip
  top-1 force toujours une ligne `bounded`/`REVIEW_REQUIRED`.

## Ce que ce rapport n'établit pas

Aucune cause racine backend n'est attribuée. Aucune tolérance n'est proposée
ni promue. Aucun verdict d'identité officiel n'est rendu. Étape suivante :
relancer la campagne (runbook `docs/runbooks/step2_pod_campaign.md`, nouveau
run-id), puis revue humaine des lignes bornées — qui seront nombreuses et
larges (plancher jusqu'à 19,56, flips top-1 documentés) — avant toute
promotion.
