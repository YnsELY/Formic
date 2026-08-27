# SPEC-02 — Diagnostic du run `a40-2026-08-27-r1`

**Statut : FAIL analysé ; la gate legacy v3 est passée pour la première fois ;
correctif du plancher de bruit implémenté et testé hors GPU. Aucune tolérance
mesurée, aucun verdict d'identité.**

## Identité du run

| Champ | Valeur |
|---|---|
| Run | `artifacts/step2/runs/a40-2026-08-27-r1/` (commité par le pod, ca404b2) |
| Commit source | `04f18ed9353fd4d0aee070aa60e33f0384aee050` (worktree propre) |
| Protocole | `SPEC-02-h8-option-b-balanced-v3`, seed 0, attempt_000 |
| Config résolue | `742dabcabb9c597c276f32ea448fd3b0fd2535d3d0035957855268c3db07488b` (cap 35 GiB) |
| Durée | 2026-08-27 15:48:10Z → 19:11:59Z (~3 h 24) |
| Préflight | chargement 303,5 s ; 1 158 s ; estimation totale ~8,5 h pour 9 669 forwards |

## Constats mesurés

1. **Preflight : PASS. Trace inertness : PASS.**
2. **Gate de continuité legacy : PASS 6/6 — première fois.** Tous les
   contrastes appariés exacts (`matched_endpoint_exact` et
   `matched_contrast_last_two_exact` vrais partout), et les empreintes brutes
   par répétition sont identiques (rép 0 ≡ rép 1) sur **16/16 slots pour les
   six cas**, y compris `instruction_scope` (0 chauffe, 0 burn-in). Le
   mécanisme v3 est visible dans l'artefact : les 4 paires de burn-in du tout
   premier cas (`audit_echo`) portent 8 empreintes toutes distinctes — le
   transitoire de première exécution absorbé dans le burn-in — puis, dès le
   deuxième cas, les empreintes de burn-in se recoupent exactement par
   (endpoint, côté) : régime stationnaire atteint avant chaque mesure admise.
   L'échec du 26/08 ne s'est pas reproduit.
3. **Échec terminal** : `RuntimeError: noise-floor last two traces are
   unstable`, au premier cas du plancher de bruit
   (`noise__audit_echo__alternating` ; `warmup_pair_traces: 0`, burn-in non
   exécuté — formes déjà chauffées par la gate legacy).
4. Détail du cas fautif (3 répétitions par paire, calendrier alterné) :
   - `reference_reference` : **stable 3/3** (empreintes identiques aux trois
     répétitions) — sans chauffe ni burn-in : le passage ABBA→alterné n'a
     créé aucun transitoire ;
   - `runner_runner` : **stable 3/3** ;
   - `reference_runner` : **instable aux trois répétitions**, avec un motif
     précis : **rép 0 ≡ rép 2 bit-identiques aux steps 1–5** des deux côtés,
     rép 1 appartient à une autre réalisation (steps 6–7 : légère dérive
     supplémentaire de fin de trace). C'est une **oscillation de période 2**
     entre deux réalisations (A, B, A), pas un transitoire : l'assertion
     « deux dernières traces identiques » compare rép 1 (B) à rép 2 (A) et ne
     peut jamais être satisfaite ; davantage de répétitions (A,B,A,B…) ou un
     burn-in (décalage de parité) n'y changeraient rien.
5. Plancher réellement mesuré (left vs right au sein d'une même paire,
   positions d'exécution adjacentes) : `max_abs_delta` **16,09375** pour RR
   comme pour NN, 21/24 comparaisons non exactes, top-1 différents, KL
   jusqu'à ~6. Le step 0 (prefill) est exact partout.
6. **Mémoire : aucune anomalie.** 34 550 975 488 octets alloués constants,
   pic 38 618 827 776 (35,97 GiB), 7,56 GiB libres à l'échec.
7. Reporting : `run_metadata.json` affiche les flags backend par défaut de
   torch (`cudnn_deterministic: false`, `flash_sdp: true`, …). C'est un
   artefact d'ordre d'écriture introduit par la robustesse v3 (le rapport est
   produit avant `load_backbone`, qui applique la politique) ; l'exécution
   mesurée était conforme, comme pour tous les runs précédents.

## Interprétation (étiquetée comme telle, sans attribution de cause racine)

La paire mixte reference_runner sous calendrier alterné n'alimente **aucun
chiffre** : le plancher injecté dans les tolérances n'utilise que RR (NN sert
de contrôle), et l'identité wrapper est décidée par la gate ABBA appariée —
qui vient de passer 6/6. Le critère bloquant exigeait donc la reproductibilité
brute d'une comparaison sans consommateur, sur un backend dont il est établi
que les sorties brutes dépendent de la position et de la séquence d'exécution.
C'est la même erreur de conception que le crossover r2 avait identifiée pour
la gate legacy (stabilité ordinale brute ≠ critère d'identité), corrigée pour
la gate, pas pour le plancher. Le mécanisme physique de l'oscillation de
période 2 n'est pas attribué au-delà des mesures ; on note qu'elle n'apparaît
ni dans les paires homogènes (RR, NN — chemins d'appel identiques des deux
côtés), ni sous ABBA (16/16 slots reproductibles pour les paires mixtes dans
la gate legacy du même processus), ni dans la matrice r6 (qui chauffait chaque
configuration sous son propre calendrier).

## Risque aval documenté (décision : mesurer d'abord)

Le plancher mesuré (~16, top-1 divergents) quantifie l'effet de position sur
ce backend. La calibration cross-path (cached vs recompute, segmenté vs
préfixes pleins) compare **par construction** des positions non appariées ;
EXP-0008 avait mesuré cached-vs-recompute CUDA à delta 14,8 avec 1/8 accords
top-1 (mesure alors confondue avec la frontière de première exécution). Il est
donc plausible que la calibration produise des désaccords top-1, que le
verdict candidat traite en échec dur, et que l'adjudication snapshot
(interrompu vs restauré) rencontre le même effet. Décision explicite de
Yanis : **ne pas amender ces critères sans données** — relancer, laisser la
campagne tout mesurer (les mesures brutes, `tolerances.candidate.json` et les
artefacts snapshot sont tous écrits avant le verdict), et statuer ensuite
chiffres en main. Un `FAIL` terminal au verdict top-1 serait alors un résultat
informatif complet, pas une session perdue.

## Corrections apportées (commit local, tests hors GPU)

1. **Plancher de bruit** : l'assertion last-two bloquante porte désormais sur
   les paires qui produisent le plancher (`reference_reference`,
   `runner_runner`) ; la paire mixte `reference_runner` reste mesurée et
   enregistrée mais devient un diagnostic non bloquant
   (`blocking_pairs`, `mixed_pair_stability_is_diagnostic_only` dans
   l'artefact ; sous-protocole `SPEC-02-alternating-noise-floor-h8-v3`).
   Budget inchangé : 9 669 forwards.
2. **Reporting** : le lanceur applique `configure_determinism` avant la
   campagne, si bien que `run_metadata.json` reflète les flags réellement
   utilisés.
3. Tests weight-free : un endpoint factice oscillant à période 2 sur la paire
   mixte fait passer le plancher (oscillation visible en diagnostic) ; la
   même oscillation sur une paire bloquante (NN) fait toujours échouer.

## Vérification locale

- Suite weight-free complète + gardes A11/A12/inertes : vertes (voir CI).
- `formic verify` : PASS ; `identity-check --toy` : PASS.
- Budget : 9 669 inchangé (`scripts/estimate_step2_campaign.py` ==
  `budget.EXPECTED_PHASE_FORWARDS`).

## Ce que ce rapport n'établit pas

Aucune cause racine backend n'est attribuée. Aucune tolérance n'est proposée.
Aucun verdict d'identité n'est rendu. Étape suivante : relancer la campagne
(runbook `docs/runbooks/step2_pod_campaign.md`, nouveau run-id — tout commit
invalide la reprise), puis analyse des mesures complètes.
