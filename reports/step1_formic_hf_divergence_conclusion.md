# Clôture de l'analyse de divergence entre Formic et Hugging Face

**Date :** 2026-08-20  
**Périmètre :** SPEC-01, Qwen3.8-27B text-only, batch 1, BF16, décodage avec cache  
**Décision recommandée :** continuer le développement de Formic, sans considérer la divergence observée comme la preuve d'un défaut du wrapper  
**Réserve :** l'identité bit à bit entre processus CUDA indépendants n'est pas établie et ne doit pas être revendiquée

## 1. Résumé exécutif

Le problème initial était une divergence entre la génération de Formic et celle
du modèle Hugging Face direct. Le prefill était parfaitement identique, mais les
générations avec cache exécutées dans des processus CUDA séparés ne l'étaient
pas. Comme l'invariant de SPEC-01 demande que Formic, avec tous ses mécanismes
désactivés, reproduise le modèle d'origine, cette divergence devait être traitée
comme un blocage potentiel.

Après les diagnostics réalisés, le scénario de loin le plus probable est le
suivant :

1. Formic ne modifie pas les calculs des cellules Qwen et ne charge pas des poids
   différents.
2. Le chemin CUDA du modèle stock, en particulier le chemin GDN utilisé pour le
   cache récurrent, dépend d'un état d'initialisation ou d'un ordinal
   d'exécution propre au processus/backend.
3. Deux exécutions placées au même ordinal produisent les mêmes résultats dans
   Formic et HF, bit pour bit.
4. Deux exécutions placées à des ordinaux ou dans des historiques différents
   peuvent diverger fortement, y compris lorsque les deux exécutions utilisent
   uniquement HF ou uniquement Formic.
5. Les comparaisons Formic/HF non alignées mélangent donc une comparaison de
   wrappers et une comparaison d'historiques CUDA. Elles ne démontrent pas un
   défaut propre à Formic.

La confiance d'ingénierie dans ce scénario est **élevée**, estimée de manière
qualitative à environ **85-90 %**. Ce chiffre n'est pas une probabilité
statistique issue du protocole interrompu; il synthétise la force convergente
des observations disponibles.

La recommandation pratique est donc :

- **ne pas arrêter ni réécrire Formic à cause de cette divergence;**
- **continuer le développement;**
- conserver comme limitation connue l'absence d'identité CUDA bit à bit entre
  processus indépendants;
- ne pas présenter SPEC-01 comme ayant formellement réussi 9/9 tant qu'une
  décision humaine n'a pas accepté cette limitation ou adapté le critère;
- traiter la reproductibilité numérique et les tolérances comme une question de
  protocole/backend, normalement du ressort de SPEC-02, et non comme un défaut
  démontré du wrapper.

## 2. Problématique de départ

Formic enveloppe le modèle Hugging Face `Qwen3_5ForCausalLM`. À ce stade du
projet, tous les mécanismes propres à Formic sont désactivés. L'objectif de
SPEC-01 est de vérifier que cette intégration est structurellement correcte et
qu'elle ne modifie pas le comportement du modèle de base.

Les premiers contrôles ont établi que :

- les logits de prefill sont identiques sur les six prompts figés;
- les 851 tenseurs textuels attendus sont chargés sans manque, ajout, erreur de
  forme ou erreur de dtype;
- le renommage des clés est bijectif;
- la tour vision et le MTP sont exclus conformément au périmètre;
- les 17 hooks de frontière no-op sont bit-inertes;
- aucune cellule Qwen n'est copiée, réimplémentée, sous-classée ou patchée.

Malgré cela, le contrôle initial de génération avec cache a échoué :

| Contrôle initial | Résultat exact Formic/HF |
|---|---:|
| Boucle greedy explicite avec cache | 0/4 exécutions |
| `generate()` greedy natif | 0/6 prompts |
| `generate()` échantillonné natif | 0/3 prompts |

Le premier token, issu du prefill, était identique. Les différences apparaissaient
ensuite pendant le décodage token par token avec cache.

Deux hypothèses principales étaient alors possibles :

- **hypothèse Formic :** le wrapper, le runner, le cache, les arguments ou le
  chargement de Formic modifient réellement le calcul;
- **hypothèse backend :** les deux chemins exécutent le même calcul logique,
  mais la sortie CUDA dépend de l'ordinal d'exécution ou d'un état global du
  runtime qui n'appartient pas au modèle.

## 3. Contexte numérique important

Le modèle utilise des couches GDN hybrides. Dans le fallback stock disponible
sur ce pod, PyTorch signale qu'un `cumsum` CUDA du chemin GDN ne possède pas
d'implémentation déterministe avec Torch 2.4 :

```text
RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation
```

Ce fait prouve qu'une opération non déterministe existe dans le chemin stock.
Il ne prouve pas, à lui seul, que cette opération est la cause exacte de toutes
les divergences mesurées. Aucun kernel et aucune cellule n'ont été remplacés,
car cela aurait changé le modèle de référence et violé la contrainte A11.

Environnement principal des diagnostics :

| Élément | Valeur |
|---|---|
| GPU | NVIDIA A40 |
| Python | 3.11.10 |
| Torch | 2.4.1+cu124 |
| CUDA runtime | 12.4 |
| Transformers | 5.8.0 |
| Accelerate | 1.14.0 |
| Safetensors | 0.8.0 |
| Dtype | BF16 |
| Attention | eager |
| Batch | 1 |
| Périmètre | text-only |

Les contrôles avancés ont également fixé :

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
cudnn.benchmark=False
cudnn.deterministic=True
cudnn.allow_tf32=False
cuda.matmul.allow_tf32=False
flash_sdp=False
mem_efficient_sdp=False
math_sdp=True
```

Même avec ces réglages, `torch.use_deterministic_algorithms(True)` ne peut pas
rendre déterministe le `cumsum` CUDA stock concerné.

## 4. Diagnostics effectués avant la campagne interrompue

### 4.1 Prefill, structure et chargement

Les six prompts de référence donnent exactement les mêmes logits de prefill :

| Mesure | Résultat |
|---|---:|
| SHA-256 des logits identique | 6/6 |
| Delta absolu maximal | 0 |
| KL moyen et maximal | 0 |
| Accord top-1 | 6/6 |

La divergence n'est donc pas une divergence générale du modèle chargé. Elle est
spécifique au chemin de décodage récurrent avec cache ou à son contexte
d'exécution.

### 4.2 Comparaison à ordinal CUDA aligné

Le diagnostic `EXP-0008` a comparé Formic et HF dans des processus séparés, mais
en alignant leur ordinal d'exécution :

| Comparaison | Exactitude |
|---|---:|
| Formic run 1 / HF run 1 sur CUDA | 8/8 logits exacts |
| Formic run 2 / HF run 2 sur CUDA | 8/8 logits exacts |
| Formic / HF sur CPU | 3/3 logits exacts |

C'est l'indice le plus fort contre l'hypothèse d'un défaut logique Formic. Un
wrapper qui modifie les entrées, les poids, le cache ou la séquence d'appels ne
devrait pas redevenir bit-identique simplement parce que les ordinaux sont
alignés.

### 4.3 Effet du premier passage dans un processus

Trois traces successives ont été exécutées dans le même processus :

| Comparaison | Logits exacts | Première différence |
|---|---:|---:|
| Run 1 / run 2 | 1/8 | étape 1 |
| Run 2 / run 3 | 8/8 | aucune |
| Run 1 / run 3 | 1/8 | étape 1 |

Le même profil a été observé avec Formic seul et avec HF stock seul. Trois
processus Formic indépendants exécutant chacun une seule trace étaient en outre
mutuellement exacts et égaux au run 1 du processus à trois traces.

Ce résultat démontre un effet de premier passage déterministe : la première
trace d'un processus appartient à une réalisation stable, les traces suivantes
à une autre réalisation stable. Ce n'est ni un bruit aléatoire simple, ni un
phénomène propre à Formic.

### 4.4 Recherche d'un état persistant dans le modèle

Le modèle a été fingerprinté avant et après les traces :

- 851/851 paramètres vérifiés;
- 2/2 buffers enregistrés vérifiés;
- attributs tensoriels directs de tous les modules vérifiés;
- 51 emplacements publics initialement à `None`, dont les emplacements pouvant
  contenir `rope_deltas`, vérifiés;
- aucune entrée modifiée.

Les diagnostics ne soutiennent donc pas l'hypothèse d'un poids, buffer ou état
tensoriel persistant du modèle muté par Formic. Un état global du runtime, du
backend ou d'un kernel reste en revanche possible et est l'explication la plus
cohérente.

### 4.5 Chauffe par forme

Une chauffe unique n'était pas suffisante pour toutes les longueurs de prompt.
Le protocole candidat a donc utilisé six traces de chauffe et deux traces
mesurées pour chaque forme de prompt/cache.

Après cette chauffe :

- Formic est localement stable sur 6/6 prompts;
- HF est localement stable sur 6/6 prompts;
- les deux dernières traces de chaque chemin sont exactes, 16/16 étapes;
- la génération Formic et la génération native exécutées dans le même modèle
  chargé donnent les mêmes séquences sur 6/6 prompts;
- une comparaison HF/HF entre un processus normal et un processus ayant subi
  une allocation CUDA préalable de 2 GiB est exacte sur 96/96 logits.

Le simple historique d'allocation CUDA de 2 GiB ne reproduit donc pas la
divergence. Cela réduit le champ des causes possibles, sans identifier le
mécanisme backend exact.

`reports/step1_warmup_controls_resume.md` décrit l'arrêt initial de cette série
de contrôles. Il s'agit d'un point de reprise historique : les artefacts
`hf_baseline`, `hf_perturbed` et `hf_compare` ont été produits ultérieurement et
sont présents dans `artifacts/step1/warmup_controls/`.

La comparaison intra-processus Formic contre boucle native après chauffe a varié
avec le calendrier :

| Prompt | Logits exacts | Top-1 identiques |
|---|---:|---:|
| `audit_echo` | 3/16 | 14/16 |
| `plain_text` | 16/16 | 16/16 |
| `code_completion` | 9/16 | 12/16 |
| `code_bugfix` | 16/16 | 16/16 |
| `instruction_short` | 16/16 | 16/16 |
| `instruction_scope` | 16/16 | 16/16 |
| **Total** | **76/96** | **90/96** |

Une campagne ultérieure avec un autre calendrier a donné **67/96 logits exacts
et 72/96 top-1**, tandis qu'une campagne antérieure avait donné **40/96 logits
exacts**. Cette dépendance forte au calendrier est difficilement compatible avec
un bug déterministe simple dans le calcul Formic.

### 4.6 Arguments d'appel et instrumentation

Un hook read-only placé sur le CausalLM de premier niveau a enregistré les
appels sans modifier les sorties.

- Gate observateur : 288/288 logits instrumentés identiques aux logits nus.
- Gate capture d'état : 16/16 exacts sur chacun des six prompts, pour Formic et
  pour la boucle HF explicite.
- Formic et la boucle HF explicite présentent zéro différence dans les arguments
  enregistrés sur 96 forwards : présence des arguments, valeurs effectives,
  formes, dtypes, strides, contenu des tenseurs, type du cache et longueurs des
  couches du cache.

L'instrumentation elle-même est donc innocente, et aucune différence d'interface
du forward n'a été trouvée entre le runner Formic et la boucle HF explicite.

`generate()` suit en revanche une convention différente de la boucle explicite :
cache précréé, masque et positions explicites, `logits_to_keep=1` et
`return_dict=True`. Ses divergences dès le prefill ne doivent pas être utilisées
pour accuser spécifiquement Formic, car il s'agit d'un autre chemin d'appel.

### 4.7 Première divergence d'état observée

Sur la campagne instrumentée :

- quatre prompts restent exacts sur 16/16 étapes;
- `audit_echo` diverge à `after_forced_1`;
- `plain_text` diverge à `after_forced_0`;
- dans les deux cas, la première différence d'état trouvée est
  `recurrent_states` dans la couche GDN 49;
- la première divergence d'état arrive au même boundary que la première
  divergence de logits, jamais avant;
- toutes les composantes des couches 0 à 48 sont encore exactes à ce boundary;
- `rope_deltas` est absent de l'entrée CausalLM text-only observée.

Cette localisation montre où les réalisations numériques commencent à se
séparer, mais pas pourquoi. Elle est compatible avec une amplification tardive
d'une différence issue du chemin GDN/CUDA. Elle ne révèle pas une mutation
antérieure du cache par Formic.

## 5. Objectif de la campagne du 20 août 2026

Le dernier protocole devait répondre aux deux questions encore ouvertes :

1. Quel est le plancher de divergence du même chemin, en comparant
   `reference/reference` puis `runner/runner` dans un calendrier partagé ?
2. Quelle est la sensibilité de `runner/reference` au nombre de chauffes et à
   l'ordre initial d'exécution ?

Le plancher même-chemin utilisait quatre réalisations logiques dans un même
processus :

- `runner_a` et `runner_b`;
- `explicit_a` et `explicit_b`;
- modes `naked` et `state_captured`;
- ordre des quatre réalisations tournant à chaque cycle;
- ordre des deux modes alternant à chaque cycle;
- six chauffes puis deux traces mesurées;
- six prompts et une continuation forcée de 16 tokens;
- cache neuf à chaque trace.

La grille ordinale devait ensuite exécuter, dans des processus neufs :

- `N=0`, `N=3` et `N=6` chauffes;
- ordre initial `runner-first` ou `explicit-first`;
- alternance des deux chemins;
- deux traces mesurées par chemin et par prompt.

Cette grille aurait permis de savoir si les écarts cross-path étaient du même
ordre que les écarts même-chemin, et si leur profil suivait directement
l'ordinal d'exécution.

## 6. Pourquoi les processus n'ont pas terminé

Les premières tentatives ont d'abord rencontré un problème périphérique avec
`pip freeze` et l'installation éditable locale. Ce problème est survenu avant
les mesures et n'apporte aucune information numérique sur le modèle.

Les tentatives qui ont réellement commencé la mesure ont ensuite été arrêtées
par la limite de temps ou par l'interruption de la session :

- une tentative a fonctionné pendant environ quatre heures et a atteint le
  cinquième prompt, pendant les chauffes;
- une autre a atteint le premier prompt, également pendant les chauffes;
- aucun OOM, crash CUDA ou traceback du modèle n'a été observé;
- après les interruptions, le GPU était libre et aucun processus de mesure ne
  restait actif.

Le protocole était beaucoup trop coûteux pour le budget du pod :

```text
4 instances × 2 modes × 8 cycles × 6 prompts = 384 traces
384 traces × 16 forwards = 6 144 forwards du modèle 27B
```

Le mode `state_captured` hachait l'état de chacune des 64 couches à chacune des
16 frontières, y compris pendant les chauffes qui n'étaient pas conservées.
Cela représente approximativement 393 216 copies/hachages de composantes de
cache GPU vers CPU pour la campagne complète, en plus des forwards. Ces copies
introduisent des synchronisations très coûteuses.

Le modèle lui-même se chargeait correctement en quelques minutes. Le goulot
d'étranglement principal était la capture et le hachage exhaustifs des états
pendant toutes les chauffes, pas un blocage du modèle.

## 7. Pourquoi aucun résultat partiel n'a été récupéré

Le harnais construisait les métriques en mémoire et n'écrivait
`same_path_floor.json` et `same_path_floor.pt` qu'après les six prompts. Il ne
créait :

- aucun checkpoint par prompt;
- aucun JSON intermédiaire;
- aucun fichier de tenseurs intermédiaire;
- aucun journal persistant des métriques calculées en mémoire.

Les lignes de progression affichées dans la console indiquaient seulement quel
prompt, quelle instance, quel mode et quel cycle venaient de finir. Elles
n'affichaient pas les comparaisons finales. La mort du processus a donc supprimé
les traces et états déjà présents en mémoire.

La grille ordinale n'a jamais commencé. Il n'existe donc aucun résultat nouveau
pour `N=0`, `N=3`, `N=6`, `runner-first` ou `explicit-first`.

Les seules informations tirées de ces exécutions incomplètes sont :

- le protocole pouvait charger le modèle et exécuter les quatre réalisations;
- les modes nu et instrumenté démarraient sans erreur immédiate;
- aucune panne fonctionnelle Formic/HF n'est apparue;
- le coût du protocole était incompatible avec la durée disponible du pod;
- la conception sans checkpoint rendait une interruption intégralement
  destructive pour les résultats partiels.

Ces informations sont opérationnelles, mais elles ne constituent pas un nouveau
résultat numérique sur le plancher de bruit.

## 8. Ce qui est établi, probable et inconnu

### Établi par mesure

- Le prefill Formic/HF est bit-identique sur 6/6 prompts.
- Le chargement textuel est strict et bijectif sur 851/851 tenseurs.
- Les hooks Formic no-op et les observateurs sont bit-inertes.
- Formic et HF sont bit-identiques sur CPU pour le diagnostic court.
- Formic et HF sont bit-identiques sur CUDA lorsque les ordinaux comparés sont
  alignés.
- Le changement entre première trace et traces suivantes existe dans Formic et
  dans HF stock avec le même profil.
- Les runs 2 et 3 d'un même processus sont bit-identiques.
- Trois premières traces issues de trois processus indépendants sont
  bit-identiques entre elles.
- Aucun état tensoriel persistant du modèle inspecté ne change.
- Après six chauffes, chaque chemin est localement stable sur les six formes.
- Les arguments enregistrés du runner Formic et de la boucle HF explicite sont
  identiques.
- Les divergences instrumentées commencent dans l'état GDN de la couche 49 au
  même instant que les logits, sans divergence d'état antérieure observée.

### Scénario le plus probable

Un état global du backend CUDA, une initialisation de kernel, une sélection de
réalisation numérique ou un effet d'ordinal propre au chemin GDN fait basculer
le calcul entre plusieurs réalisations stables. Formic et HF utilisent la même
réalisation lorsqu'ils occupent le même ordinal, mais les comparaisons exécutées
dans des calendriers ou processus indépendants peuvent associer des réalisations
différentes.

Le `cumsum` CUDA non déterministe est un suspect naturel, mais la causalité
précise n'est pas démontrée. Il peut aussi s'agir d'un autre état runtime lié au
même chemin. La conclusion importante pour Formic ne dépend pas de
l'identification exacte du kernel : l'effet est reproduit avec HF stock et
disparaît dans les comparaisons ordinales alignées.

### Toujours inconnu

- Le plancher complet `runner/runner` et `reference/reference` dans le nouveau
  calendrier partagé.
- La matrice complète de sensibilité aux chauffes `N=0/3/6`.
- Le mécanisme CUDA exact responsable de la sélection entre réalisations.
- Une garantie générale pour toutes les formes, tous les GPU ou les futures
  versions de Torch.
- L'identité bit à bit entre deux processus CUDA indépendants après chauffe.

## 9. Évaluation de l'hypothèse d'un bug Formic

### Éléments qui seraient attendus en présence d'un bug Formic

Un défaut du wrapper devrait typiquement produire au moins un des symptômes
suivants : arguments différents, poids différents, mutation du cache, divergence
au prefill, divergence CPU, divergence constante au même endroit, échec de HF à
reproduire le phénomène, ou impossibilité pour Formic et HF de redevenir exacts
à ordinal aligné.

Aucun de ces symptômes n'a été observé.

### Éléments effectivement observés

- égalité structurelle et des poids;
- égalité du prefill;
- égalité CPU;
- égalité CUDA à ordinal aligné;
- effet ordinal identique dans HF stock;
- stabilité locale après chauffe;
- résultats cross-path variables selon le calendrier;
- mêmes arguments de forward;
- aucun état modèle muté trouvé.

La meilleure conclusion n'est donc pas « Formic est mathématiquement prouvé
parfait ». La meilleure conclusion est :

> **Il n'existe actuellement aucun indice positif d'un défaut logique propre à
> Formic, tandis que plusieurs mesures indépendantes attribuent la divergence au
> contexte d'exécution CUDA partagé par Formic et HF.**

Le risque résiduel d'un effet subtil du runner sur l'historique du backend n'est
pas nul. Même dans ce cas, il s'agirait vraisemblablement d'une interaction de
calendrier avec le backend stock, et non d'une implémentation erronée des cellules,
du chargement ou du cache par Formic.

## 10. Décision recommandée pour la suite du projet

### Décision technique

**Oui, le développement de Formic peut continuer.** Les données disponibles ne
justifient ni l'abandon du projet, ni une réécriture du runner, ni une modification
des cellules Qwen. Continuer à bloquer tout développement uniquement pour obtenir
une causalité CUDA parfaite aurait un coût disproportionné par rapport au risque
Formic restant.

### Décision de gouvernance

La continuation ne doit cependant pas falsifier les résultats :

- le contrôle strict de génération inter-processus reste rouge;
- SPEC-01 reste factuellement à 8/9 dans les documents actuels;
- ADR-0004 reste `PROPOSED` tant qu'une personne responsable ne l'accepte pas;
- aucune affirmation d'identité CUDA inter-processus générale ne doit être faite;
- une validation humaine peut décider d'accepter SPEC-01 avec cette limitation
  documentée et d'autoriser la suite;
- SPEC-02 devra définir un protocole de comparaison aligné et, si nécessaire,
  des tolérances numériques versionnées.

La recommandation de ce rapport est que cette validation humaine **autorise la
continuation**, avec le problème classé comme **limitation de reproductibilité du
backend CUDA, non comme bug Formic démontré**.

### Conditions minimales à conserver

- Garder les tests de chargement strict A12 et d'absence de réimplémentation A11.
- Garder le contrôle de prefill bit-exact.
- Garder les hooks désactivés par défaut et leur preuve d'inertie.
- Pour les mesures cached-decode importantes, stabiliser chaque forme et vérifier
  les deux dernières traces du même chemin.
- Comparer si possible les chemins au même ordinal ou dans le même processus.
- Enregistrer config, environnement, ordre d'exécution et politique de chauffe.
- Ne pas patcher silencieusement le GDN ou le `cumsum` pour rendre un test vert.

## 11. Verdict final

Le phénomène observé est réel : des générations CUDA avec cache peuvent produire
des logits et parfois des top-1 différents. Il ne faut donc pas le minimiser en
le qualifiant de simple erreur d'affichage ou de différence négligeable.

En revanche, la question déterminante est de savoir si ce phénomène est introduit
par Formic. Les preuves disponibles indiquent majoritairement que **non** :

- HF stock présente le même effet d'ordinal;
- Formic et HF sont exacts lorsqu'ils sont correctement alignés;
- le CPU est exact;
- le prefill est exact;
- les poids, modules, hooks et arguments observés sont équivalents;
- aucun état modèle fautif n'a été trouvé.

Le scénario le plus probable est donc un effet déterministe mais dépendant de
l'historique du backend CUDA/GDN. La dernière campagne aurait renforcé ou nuancé
ce diagnostic, mais son absence de résultat ne renverse pas les preuves déjà
acquises.

**Conclusion de clôture : Formic ne présente pas, sur la base des mesures
disponibles, de défaut suffisamment probable pour empêcher la poursuite du
développement. Il faut continuer, documenter la limitation numérique CUDA et
reporter la formalisation des tolérances et du protocole d'identité à l'étape
appropriée.**

## 12. Références et artefacts

- `STATUS.md`
- `reports/step1_report.md`
- `reports/step1_decode_diagnostics.md`
- `reports/step1_runner_state_diagnostics.md`
- `reports/step1_warmup_controls_resume.md`
- `docs/adr/ADR-0004-deterministic-cached-decode-warmup.md`
- `scripts/step1_decode_diagnostics.py`
- `scripts/step1_warmup_controls.py`
- `scripts/step1_runner_state_diagnostics.py`
- `scripts/step1_ordinal_noise_controls.py`
- `artifacts/step1/decode_diagnostics/`
- `artifacts/step1/warmup_controls/`
- `artifacts/step1/runner_state_diagnostics/`

Le répertoire `artifacts/step1/ordinal_noise_controls/` n'existe pas, car la
campagne correspondante n'a atteint aucune écriture finale.
