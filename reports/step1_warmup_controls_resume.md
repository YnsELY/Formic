# SPEC-01 — Reprise des contrôles de chauffe

**État :** interrompu, sans verdict et sans modification d'ADR.

## Point d'arrêt

La mesure intra-processus a terminé et a écrit :

- `artifacts/step1/warmup_controls/intra.json`
- `artifacts/step1/warmup_controls/intra.pt`

Le contrôle inter-processus HF/HF de référence a été interrompu pendant le
prompt `instruction_scope`, à la chauffe `4/6`. Il n'a pas atteint les deux
traces mesurées et n'a écrit ni `hf_baseline.json` ni `hf_baseline.pt`.

Le GPU était ensuite libre et aucun processus Python de cette mesure ne
tournait. Les résultats partiels affichés avant l'arrêt ne sont pas des
artefacts de mesure et ne doivent pas être interprétés.

## Protocole à reprendre

Le harnais est `scripts/step1_warmup_controls.py`. Il applique la configuration
numérique versionnée, utilise une continuation forcée de 16 tokens et exige que
les deux dernières traces mesurées soient exactes au sein de chaque processus.

Les étapes restantes sont indépendantes de la mesure intra-processus déjà
terminée :

1. Refaire intégralement le contrôle HF de référence :

   ```bash
   PYTHONPATH=$PWD python -u scripts/step1_warmup_controls.py --stage hf-baseline
   ```

2. Exécuter le contrôle HF dans un nouveau processus avec l'historique
   d'allocation CUDA de 2 GiB avant chargement :

   ```bash
   PYTHONPATH=$PWD python -u scripts/step1_warmup_controls.py --stage hf-perturbed
   ```

3. Comparer les deux dernières traces forcées mesurées :

   ```bash
   PYTHONPATH=$PWD python -u scripts/step1_warmup_controls.py --stage hf-compare
   ```

La dernière étape produit
`artifacts/step1/warmup_controls/hf_compare.json`, avec les métriques par
prompt et par étape: égalité exacte, delta absolu maximal, KL, accord top-1 et
première divergence.

## Garde-fous

- Ne pas relancer `--stage all`: il referait inutilement la mesure
  intra-processus déjà disponible.
- Ne pas utiliser le résultat incomplet du contrôle de référence.
- Ne pas modifier `ADR-0004`, `STATUS.md`, les kernels, le checkpoint, les
  paquets installés ou la version de torch à partir de ces contrôles.
- SPEC-01 reste à `8/9`; SPEC-02 ne démarre pas.
