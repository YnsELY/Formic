# Runbook — campagne de calibration SPEC-02 sur le pod A40

Procédure opératoire complète pour l'agent (humain ou de codage) qui lance la
campagne sur le pod. Elle n'accorde aucune latitude : tout écart est un
incident à rapporter, pas à résoudre en séance.

## 1. Pré-requis du pod

| Élément | Valeur attendue |
|---|---|
| GPU | exactement **une** NVIDIA A40 visible (44,42 GiB) |
| Checkout | `/workspace/formic` |
| Checkpoint | `/workspace/Qwen3.8-27B` (~55 Go, 18 shards) |
| Interpréteur | venv **`/workspace/formic-venv`** — torch 2.4.1+cu124, transformers 5.8.0, accelerate 1.14.0, safetensors 0.8.0, Python 3.11.x |

Le venv n'est PAS créé par ce runbook ; s'il est absent ou incomplet,
**s'arrêter et le signaler** (interdiction d'installer ou modifier des
paquets).

## 2. Synchronisation

```bash
source /workspace/formic-venv/bin/activate
cd /workspace/formic
git status          # DOIT être propre : le lanceur refuse un worktree sale
git pull
git rev-parse HEAD  # DOIT être le commit annoncé par Yanis pour cette session
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

## 3. Lancement

```bash
python scripts/step2_a40_campaign.py \
  --run-id a40-YYYY-MM-DD-r1 \
  --sampled-continuation-seed 0
```

- `--gpu-max-memory` garde son défaut `35GiB` (validé par r6/r2). Ne pas le
  changer : toute autre valeur change le hash de config résolue et invalide
  la reprise.
- Le run est entièrement automatique : preflight (~20 min, estimation
  affichée), puis enchaînement sans gate budgétaire. Plan v3 : **9 669
  forwards**, ~8,3 h en équivalent historique — l'estimation post-preflight
  fait foi.
- Le lanceur n'arrête jamais le pod et ne demande pas de l'arrêter.

## 4. Reprise après un FAIL

```bash
python scripts/step2_a40_campaign.py \
  --run-id <même-run-id> --sampled-continuation-seed 0 --resume
```

Conditions strictes : même commit, même config, même corpus, même backbone —
tout écart est refusé (`resume identity differs`). Un run terminé
`CALIBRATION_COMPLETE` refuse la reprise. Après un changement de code ou de
protocole : **nouveau run-id**, jamais `--resume`.

## 5. Fin de session — à rapporter

1. Code de sortie du lanceur et dernière ligne `IDENTITY CHECK: ...`.
2. `artifacts/step2/runs/<run-id>/terminal.json` (intégral).
3. Présence/contenu de `tolerances.candidate.json` et
   `verdict.candidate.json`.
4. Pic mémoire : dernier bloc de `memory/cuda_memory.json`.
5. En cas de FAIL : `diagnostics/<cas>.json` du cas fautif.

Puis committer et pousser les artefacts :

```bash
git add -f artifacts/step2/runs/<run-id>/ artifacts/step2/preflight/
git commit -m "chore: add SPEC-02 A40 campaign artifacts (<run-id>)"
git push origin main
```

## 6. Interdits en séance

- Modifier du code, une config, un prompt gelé, `tolerances*` ou une version
  de paquet.
- Conclure sur une cause (« c'est le cumsum », « c'est l'allocateur ») —
  les artefacts se rapportent, l'analyse se fait hors pod.
- Relancer un run-id existant sans `--resume`, ou « reprendre » un run
  `CALIBRATION_COMPLETE`.
- Supprimer ou réécrire des artefacts.
- Promouvoir des tolérances : `scripts/step2_promote_calibration.py` est une
  étape humaine séparée, hors pod, après revue des lignes bornées.

## 7. Résultat attendu d'un run réussi

`terminal.json` : `status: CALIBRATION_COMPLETE`, message
`CALIBRATION COMPLETE — PROMOTION REQUIRED`. Ce n'est **pas** un PASS
officiel : la promotion, le rapport, l'enregistrement de gouvernance et
l'acceptation d'ADR-0005 restent des actions humaines après analyse.
