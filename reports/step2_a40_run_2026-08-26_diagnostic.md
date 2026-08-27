# SPEC-02 — Diagnostic du run `a40-2026-08-26-r1`

**Statut : FAIL analysé ; correctifs de protocole v3 implémentés et testés
hors GPU. Aucune tolérance mesurée, aucun verdict d'identité.**

## Identité du run

| Champ | Valeur |
|---|---|
| Run | `artifacts/step2/runs/a40-2026-08-26-r1/` (commité par le pod, c565085) |
| Commit source | `f7794071624bbe3309df4857c081f04b071eccd4` (worktree propre) |
| Protocole | `SPEC-02-h8-option-b-balanced-v2` |
| Config résolue | `b0b2ca19b553ea06f41b0cf4f876107bfd843ad08d0ccf5c281123ce3c7965b5` (cap 35 GiB) |
| Corpus | `482e63d88a53d2850fe87db648f7d6fe2414ca5ee64b1a307de7cb3501c1f3c0` |
| Backbone | `74e1813c29b065406f4b772ed7c9059b8455428bff9aa6e572645cf09743c662` (851 tenseurs) |
| Environnement | A40 47,7 Go, torch 2.4.1+cu124, transformers 5.8.0, accelerate 1.14.0, Python 3.11.10 ; cudnn_deterministic, math-SDP seul, TF32 off ; aucun fast-path (FLA/flash-attn/causal-conv1d absents) |
| Durée | 2026-08-26 20:49:44Z → 21:48:38Z (~59 min) |

## Constats mesurés

1. **Preflight : PASS.** 333 forwards, chargement 223,6 s, estimation totale
   7,14 h pour 8 549 forwards.
2. **Trace inertness : PASS.** 12 prompts, tout bit-exact, KL = 0.
3. **Échec terminal :** `RuntimeError: balanced endpoint identity comparison
   diverged` sur le premier cas de la gate legacy, `legacy__audit_echo`.
4. Sur les 256 contrastes appariés du cas :
   - les **128 contrastes `*_companion_runner` sont exacts 128/128** (delta 0,
     KL 0) ;
   - **19 contrastes `*_companion_reference` échouent**, tous à
     `configuration_ordinal 0` : 5 en répétition 0 (steps 5–7), 14 en
     répétition 1 (steps 1–7) ;
   - les échecs impliquent tous la **paire `reference_reference` du round 0**,
     c'est-à-dire les **32 premiers forwards mesurés du processus** (ordinaux
     processus 96–127, immédiatement après les 96 forwards de chauffe
     `capture=False`) ;
   - la même paire RR aux rounds 1–3 (ordinaux 224+) est **100 % exacte**.
5. Amplitudes des contrastes échoués : `max_abs_delta` 5,0–15,2, KL 0,49–6,04,
   désaccords top-1 — la même famille d'amplitudes que le basculement de
   réalisation d'EXP-0008 (delta 14,15625, KL 3,4712985).
6. Le `raw_control_floor` (left/right d'une même paire, diagnostic non
   bloquant) est non exact à 112/128 pour RR **et** NN, avec des KL identiques
   entre répétitions (ex. NN step 1 : KL 2,4553784178989333 aux répétitions 0
   et 1) : l'effet de position est massif mais reproductible.
7. **Mémoire : aucune anomalie.** 23 mesures, 34 550 975 488 octets alloués
   constants du preflight au point d'échec ; aucun OOM.

## Recoupement avec les artefacts antérieurs

- Sonde du 24/08 (`legacy-audit-2026-08-24`) : `first_changed_repetition=1`,
  `changed_side=both_reference_and_runner`.
- Matrice r5 : les trois configurations séquentielles mesurées ont
  `first_changed_repetition=1` **et** `last_two_exact=true` (répétition 1 ≡
  répétition 2) ; chaque configuration refait ses six chauffes et re-transite
  à sa répétition 1.
- Crossover r2 : 1 536/1 536 contrastes appariés exacts sur 8 rounds ;
  336/384 groupes ordinaux bruts changent.
- Le step 0 (prefill) est bit-exact dans toutes les mesures disponibles ;
  l'instabilité naît au premier step de décodage caché.

## Interprétation (étiquetée comme telle, sans attribution de cause racine)

Les constats sont compatibles avec une seule lecture d'ensemble : le
basculement déterministe de réalisation numérique de première exécution,
documenté par EXP-0008 et intégré par ADR-0004, **déborde de la chauffe dans
les ~2 premières paires-traces mesurées**, parce que la chauffe
(`capture=False`) n'exécute pas le chemin de mesure exact (pas de copie CPU
des logits à chaque step). La première paire mesurée après un bloc de chauffe
peut donc appartenir à la réalisation pré-transitoire, tandis que tous ses
compagnons de slot, mesurés plus tard, appartiennent à la réalisation
stationnaire — d'où l'échec du critère `matched_endpoint_exact` concentré sur
la première paire du processus. Le côté runner étant exact 128/128 et le
crossover r2 ayant validé 1 536/1 536, **rien dans ce run n'indique une
non-identité Formic↔Qwen**. Le mécanisme physique du basculement n'est pas
attribué au-delà de ces mesures.

## Corrections apportées (protocole v3, commit local, tests hors GPU)

1. **Burn-in mesuré-jeté** après chaque bloc de chauffe non vide : 4
   paires-traces sur le chemin de mesure exact pour les gates par paires
   (fenêtre observée : 2 ; marge ×2), 1 répétition pour les mesures
   cross-path, l'inertie de trace et snapshot/restore. Enregistré dans les
   artefacts (`burn_in`, `excluded_from_blocking_criteria: true`), exclu de
   tout critère bloquant et des statistiques de tolérance. Épinglé :
   `identity.burn_in_pair_traces = 4`, `identity.burn_in_repetitions = 1`.
2. **Chauffe par endpoint** : un registre de chauffes par endpoint ; le runner
   est désormais chauffé pour `prefill_full`, `decode_recompute` et le
   décodage caché long (le registre partagé le laissait froid, et sa
   répétition 0 pouvait gonfler les maxima de tolérance).
3. **Sonde 64 en vrai logits-only** : le profil de capture est forcé à
   `LOGITS_ONLY` de bout en bout (auparavant seule la sérialisation filtrait ;
   la capture retenait ~6 Go d'états de frontière par trace, pic estimé
   ~39,5 Go).
4. **Adjudication snapshot planchérisée** : seuil = max(ligne candidate
   short/cached/logits, plancher RR mesuré) — cohérent avec la règle « jamais
   sous le plancher référence/référence » déjà appliquée aux lignes logits.
5. **Robustesse** : un verdict candidat FAIL termine la campagne en FAIL
   (exit 1) ; `reference_continuations` est résumable cas par cas ;
   `run_metadata.json` et l'évidence mémoire s'empilent par tentative au lieu
   d'être écrasés ; la reprise refuse un run déjà `CALIBRATION_COMPLETE` ; les
   scripts de diagnostic appliquent `configure_determinism` avant
   `environment_report()` (les rapports de la sonde et de r5 affichaient les
   flags par défaut de torch parce qu'ils étaient écrits avant
   `load_backbone` ; l'exécution, elle, appliquait la politique épinglée).
6. **Budget v3 : 8 549 → 9 669 forwards** (inertie 144, legacy 3 872, plancher
   752, snapshot 64, short/medium 808 chacun, long 488, sonde 2 304 ;
   preflight 333 et continuations 96 inchangés). L'augmentation est
   intégralement de la chauffe et du burn-in jeté ; l'évidence admise par cas
   est inchangée. Équivalent historique : ~8,27 h (au lieu de 7,31 h).
   Le protocole passe à `SPEC-02-h8-option-b-balanced-v3` et le hash de config
   résolue change : **toute reprise d'un run antérieur est volontairement
   invalidée ; la prochaine session exige un nouveau run-id.**

## Vérification locale

- Suite weight-free complète : **373 passed** (dont `tests/test_burn_in.py`
  qui rejoue la forme exacte de l'échec contre un endpoint factice à
  réalisation basculante : la gate échoue sans burn-in, passe avec, et
  l'artefact contient l'évidence du burn-in).
- `formic verify` : PASS ; `identity-check --toy` : PASS.
- `scripts/estimate_step2_campaign.py` : total 9 669, identique à
  `budget.EXPECTED_PHASE_FORWARDS` et au plan validé par
  `campaign_plan.validate()`.

## Ce que ce rapport n'établit pas

Aucune cause racine backend n'est attribuée. Aucune tolérance n'est proposée.
Aucun verdict d'identité n'est rendu. La seule étape restante est une nouvelle
campagne A40 avec le lanceur v3 (runbook :
`docs/runbooks/step2_pod_campaign.md`), suivie de la revue humaine des lignes
bornées, de la promotion et de l'acceptation d'ADR-0005.
