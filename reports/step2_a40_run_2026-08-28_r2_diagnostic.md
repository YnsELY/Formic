# SPEC-02 — Diagnostic du run `a40-2026-08-28-r2`

**Statut : FAIL analysé ; huit phases complètes, les deux amendements v4
validés en conditions réelles ; l'échec est un bug structurel localisé
(profil de capture des références segmentées longues), corrigé et testé hors
GPU. Aucune tolérance promue, aucun verdict d'identité officiel.**

## Identité du run

| Champ | Valeur |
|---|---|
| Run | `artifacts/step2/runs/a40-2026-08-28-r2/` (commité par le pod, 63c4326) |
| Commit source | `7a4bea5` (worktree propre), protocole `SPEC-02-h8-option-b-balanced-v3` |
| Durée | 2026-08-28 17:01:31Z → 22:19:32Z (~5 h 18) |
| Environnement | flags backend conformes dans `run_metadata.json` (`cudnn_deterministic: true`, `flash_sdp: false`, `mem_efficient_sdp: false`, `cudnn_allow_tf32: false`) — le correctif de reporting du 27/08 fonctionne |

## Ce qui est passé (8 phases, 63 cas commités)

`preflight`, `trace_inertness`, `legacy_continuity`, `noise_floor`,
`snapshot_restore`, `reference_continuations`, **`short`** et **`medium`**
— la classe medium complète pour la première fois. La classe `long` a démarré
et son premier cas (`long_resume_incidents / prefill_full`) est passé.

### Les deux amendements v4 sont validés par la mesure

Le cas qui avait tué le run r1, `medium_cache_regression / decode_cached /
greedy`, est **passé** :

- `blocking_criterion: reference_fingerprints_identical` ;
- `reference_fingerprints_identical: true` (la référence recompute est
  bit-stable sur les trois répétitions) ;
- `last_two_exact: false` conservé comme diagnostic — la variabilité du
  candidat est enregistrée, non bloquante.

Observation notable : les trois empreintes candidates de r2
(`d590149d…`, `39e10cd1…`, `c82b23c9…`) sont **identiques à celles de r1**.
La « variabilité » du chemin cached est donc elle-même déterministe d'un
processus à l'autre : c'est une séquence de réalisations reproductible, pas
un aléa. Constat brut, sans attribution de cause.

## L'échec

`calibration__long_resume_incidents__prefill_segmented__median` :
`TraceStructureError: model-attached state registry differs`
(`formic/science/identity/comparison.py:174`), levée dès la première paire
(le burn-in), avant toute observation — le diagnostic du cas ne contient donc
que la structure d'échec.

### Cause : deux profils de capture désalignés entre les côtés

`_capture_profile(length_class, is_final)` (`executor.py`) rend, pour la
classe longue, `FINAL_STATE_ONLY` sur la frame finale et `LOGITS_ONLY` avant.

- **Candidat** (segmenté long) : une seule trace de deux frames
  (1 218 / 1 219 tokens) → frame 0 en `LOGITS_ONLY` (registre `model_state`
  vide), frame 1 en `FINAL_STATE_ONLY`.
- **Référence** : chaque préfixe complet est exécuté comme une trace
  `PREFILL_FULL` **séparée d'une seule frame** → `is_final=True` à chaque
  fois → `FINAL_STATE_ONLY` partout, y compris pour le préfixe 0.

À la frame 0, la comparaison structurelle voit donc un registre
`model_state` vide d'un côté et peuplé (`rope_deltas`) de l'autre, et refuse
la paire.

Pourquoi personne ne l'avait vu : en short/medium, les deux côtés sont
`FULL_BOUNDARIES` (le profil ne dépend pas de `is_final`) ; en long
`prefill_full`, la frame unique est finale des deux côtés ; la sonde 64 force
`LOGITS_ONLY` des deux côtés. **Seul le cas long + segmenté mélange les
profils entre les côtés** — et aucun run n'était allé aussi loin.

C'était de surcroît une non-conformité à ADR-0005 (« long : logits et état
final seulement ») : la référence sur-capturait un état final complet à
chaque préfixe, y compris intermédiaire.

## Correction (commit local, tests hors GPU)

1. **`executor.py`, `execute_reference_for_candidate`, branche segmentée** :
   les segments sont matérialisés (`parts`) et chaque préfixe reçoit
   explicitement le profil de la frame candidate qu'il double —
   `_capture_profile(length_class, step == len(parts) - 1)` — au lieu de
   laisser chaque trace mono-frame se résoudre en « finale ». Un
   `capture_profile` explicite (sonde 64) reste prioritaire.
   Effet : short/medium inchangés ; long → préfixes intermédiaires en
   `LOGITS_ONLY`, préfixe final en `FINAL_STATE_ONLY`.
2. **Modèle de coût** : le transfert des références segmentées longues suit
   la même règle. Transferts de la classe longue **18,87 → 14,07 GiB**
   (médiane 4,83 → 3,63 ; quarts 7,24 → 3,64) ; calibration principale
   73,03 → 68,23 GiB ; total planifié 84,36 → **79,56 GiB**. **Forwards
   inchangés : 9 925.**
3. Tests weight-free : un cas toy long + segmenté reproduit exactement
   l'erreur du pod sans le correctif (`TraceStructureError: model-attached
   state registry differs`) et passe avec, en vérifiant que les frames non
   finales des deux côtés n'ont ni frontières ni registre d'état, et que la
   frame finale en a des deux côtés ; un test de non-régression vérifie que
   le segmenté short garde `FULL_BOUNDARIES` partout.

## Ce qui reste devant

Après ce cas : les quarts longs et le second prompt long (mêmes chemins que
le cas corrigé), le decode cached long (référence cached — mêmes frames et
mêmes profils des deux côtés par construction), la sonde 64 (profil forcé
identique des deux côtés) et les adjudications finales v4 (dont les critères
viennent d'être validés sur short et medium).

## Vérification locale

- Suite weight-free complète et gardes A11/A12/inertes : vertes.
- `formic verify` : PASS ; `identity-check --toy` : PASS.
- `scripts/estimate_step2_campaign.py` : 9 925 forwards, transferts
  régénérés et propagés dans `preflight.py` et le plan de coût.

## Ce que ce rapport n'établit pas

Aucune cause racine backend n'est attribuée (notamment pour la
reproductibilité inter-runs de la séquence de réalisations candidates).
Aucune tolérance n'est proposée ni promue. Aucun verdict d'identité officiel
n'est rendu.
