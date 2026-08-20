# Formic

**Formic** est un modèle d'exécution spécialisé pour l'ingénierie logicielle.
Il constitue le composant neural d'exécution du projet **Uly Code**, un système
agentique conçu pour réaliser des tâches de software engineering de manière
fiable, contrôlée et vérifiable.

Formic n'est pas conçu, dans un premier temps, comme un assistant conversationnel
généraliste autonome. Il est conçu pour recevoir une tâche déjà comprise,
diagnostiquée et cadrée par le système Uly Code, puis pour la transformer en une
suite d'actions logicielles correctes, limitées et vérifiées.

## Uly Code et Formic

Uly Code est le système agentique global. Il sépare deux responsabilités qui
sont souvent mélangées dans les agents de programmation actuels :

```text
Modèle d'orchestration
    comprendre le problème
    enquêter dans le dépôt
    diagnostiquer la cause
    définir la stratégie globale
    découper le travail
              |
              v
Tâche précise, contrainte et vérifiable
              |
              v
Formic
    comprendre la tâche cadrée
    lire les preuves utiles
    décider localement
    produire une action typée
    vérifier et corriger
    terminer avec des preuves
```

Le modèle d'orchestration est responsable de l'investigation ouverte et de la
stratégie générale. **Formic est le modèle d'exécution** : il reçoit un objectif
déjà défini, les contraintes applicables, le contexte nécessaire et les critères
de réussite, puis il agit avec discipline.

Cette séparation est le principe fondateur du projet :

- le modèle d'orchestration cherche et comprend le problème;
- Formic exécute la tâche avec fidélité;
- le système Uly Code conserve l'état, applique les règles et vérifie les
  résultats;
- aucune déclaration du modèle ne suffit à elle seule à considérer une tâche
  comme terminée.

Formic est donc conçu, dans un premier temps, **spécifiquement pour le système
agentique Uly Code**. Ses interfaces, ses contraintes et son architecture sont
définies autour de ce rôle d'exécuteur.

## Le problème à résoudre

Les agents de programmation actuels sont généralement construits autour d'un
modèle conversationnel entouré d'une boucle de prompts et d'outils. Cette
approche est utile, mais elle présente des faiblesses structurelles sur les
tâches longues ou sensibles :

- les instructions importantes se mélangent avec le contenu du dépôt et les
  sorties d'outils;
- les contraintes peuvent être oubliées, diluées ou contredites au fil du
  contexte;
- l'état du travail reste souvent implicite dans l'historique conversationnel;
- le raisonnement peut devenir trop long, redondant ou hors sujet;
- les modifications sont produites sous forme de texte libre, sans garantie
  structurelle sur leur portée;
- le modèle peut déclarer qu'il a terminé sans preuve externe suffisante;
- les échecs, les reprises et les décisions précédentes sont difficiles à
  représenter de manière fiable.

Uly Code et Formic cherchent à remplacer cette logique de conversation ouverte
par une logique d'exécution structurée : une tâche cadrée, une action limitée,
une validation déterministe et un état versionné.

## L'objectif de Formic

L'objectif est de construire un modèle capable de réaliser une tâche de
software engineering bien spécifiée avec :

- une forte fidélité aux instructions;
- une stricte adhérence au périmètre demandé;
- des modifications minimales et justifiées;
- une compréhension suffisante du code et des dépendances locales;
- une capacité de vérification et de correction;
- une gestion explicite des erreurs et des reprises;
- une détection fiable de la fin réelle de la tâche;
- un raisonnement proportionné à la difficulté;
- une exécution reproductible et contrôlable;
- une latence et une consommation de calcul raisonnables.

Le terme **executor** ne signifie pas que Formic est dépourvu de raisonnement.
Formic doit comprendre la tâche, analyser les conséquences d'une modification,
raisonner sur les contraintes, choisir une action, vérifier son travail et
revenir sur une décision si de nouvelles preuves le nécessitent.

La différence est que ce raisonnement doit servir directement l'exécution. Il ne
doit pas devenir une investigation globale sans fin ni une conversation qui
remplace l'action.

## Le modèle d'exécution visé

Le projet ne cherche pas simplement à ajouter un prompt spécialisé à un grand
modèle existant. La cible, appelée **CAPE-R** (*Contract-Aware Progressive
Executor, Revised*), change l'unité fondamentale de travail :

```text
Contrat d'instructions fiable
        +
État versionné et typé
        +
Preuves sélectionnées du dépôt
        |
        v
Décision neural bornée
        |
        v
Une action typée ou une proposition d'état
        |
        v
Validation déterministe et preuves externes
        |
        +--> commit atomique
        |
        +--> rejet sans effet de bord
```

Une tâche longue devient une séquence de transactions d'exécution, et non un
unique décodage conversationnel qui accumule tout l'historique.

Chaque transaction doit :

1. lire un contrat d'instructions immuable et versionné;
2. lire un état externe et typé;
3. recevoir uniquement les éléments de preuve nécessaires;
4. produire une action principale bornée;
5. faire valider cette action par les règles du système;
6. l'appliquer dans un environnement contrôlé;
7. vérifier le résultat;
8. committer un nouvel état ou rejeter l'action sans effet de bord.

La complétion n'est donc pas uniquement une phrase produite par le modèle. Elle
est une prédiction qui doit être confirmée par des preuves déterministes : tests,
analyse syntaxique, vérification de périmètre, état du dépôt et autres contrôles
applicables.

## Les quatre plans de l'architecture

La cible CAPE-R organise le système en quatre plans complémentaires.

### Plan de contrôle

Le plan de contrôle contient le contrat d'instructions (`ContractIR`) : objectifs,
contraintes, autorité, périmètre, critères de réussite et règles applicables.
Un moniteur de référence non neural vérifie que les actions proposées respectent
ce contrat. Il peut refuser une action, mais il n'invente pas lui-même une
solution.

### Plan d'état

Le plan d'état, appelé **State Fabric**, conserve les informations durables :

- snapshot du dépôt;
- fichiers, symboles et relations importantes;
- graphe des obligations de la tâche;
- preuves collectées;
- échecs et tentatives précédentes;
- transitions d'état;
- journal append-only des décisions et validations.

L'état durable ne dépend pas du contenu fragile d'un cache neural ou d'une longue
conversation.

### Plan d'exécution

Le plan d'exécution réutilise le tronc Qwen conservé dans Formic. Il reçoit le
contrat, l'état et les preuves nécessaires, puis produit une décision structurée
au moyen d'une grammaire et de sorties typées plutôt qu'une simple réponse libre.

À terme, l'architecture pourra contrôler la profondeur de calcul à des frontières
naturelles du modèle. Cette capacité est volontairement différée jusqu'à ce que
la base complète soit mesurée et validée.

### Plan de commit

Le plan de commit vérifie la forme, le périmètre, les hashes, les tests, les
parseurs, les types et les autres critères applicables. Il applique la
modification dans un environnement contrôlé, puis committe atomiquement ou
rejette l'action.

## Pourquoi partir de Qwen3.8-27B ?

Formic ne préentraîne pas un modèle depuis zéro. Le projet utilise le checkpoint
Qwen3.8-27B comme substrat neural, parce qu'il fournit une base forte pour le
code, le raisonnement et les usages agentiques.

Le checkpoint possède notamment :

- environ 27 milliards de paramètres;
- 64 couches de décodeur;
- 16 groupes hybrides;
- une alternance vérifiée de trois couches Gated DeltaNet et d'une couche de
  full attention par groupe;
- un contexte natif très large;
- une architecture adaptée à l'étude du calcul adaptatif et de l'état
  séquentiel.

Le projet ne considère toutefois pas ces propriétés comme des hypothèses. Le
checkpoint a été audité avant l'implémentation, et les décisions d'architecture
doivent respecter ce que l'audit a effectivement établi.

## Discipline de réutilisation des poids

La règle fondamentale de Formic est :

```text
tous les mécanismes Formic désactivés == comportement du Qwen d'origine
```

Cela implique notamment :

- les cellules Qwen ne sont pas réimplémentées;
- les cellules Qwen ne sont pas copiées puis modifiées;
- les équations, poids, normalisations et ordres résiduels sont conservés;
- le chargement des tenseurs est strict et vérifié dans les deux directions;
- la tour vision n'est pas construite dans le chemin text-only;
- les hooks et frontières sont désactivés par défaut;
- toute nouvelle capacité est ajoutée autour du tronc et placée derrière une
  configuration désactivée par défaut;
- une mesure citée doit conserver son environnement, sa configuration, son
  seed, son commit et son identifiant d'expérience.

Cette discipline permet de distinguer deux choses :

- le comportement du modèle de base;
- les capacités ajoutées par l'architecture Formic.

## Comment le projet sera construit

Le développement suit huit étapes strictement ordonnées. Une étape ne doit pas
être considérée comme validée sans tests, rapport, état documenté et validation
humaine lorsque cela est requis.

### Partie 1 : construire les fondations

1. **Fondation Formic et intégration du backbone** : dépôt, configuration,
   registre scientifique, chargement strict et vue testable des groupes Qwen.
2. **Identité et tolérances numériques** : baseline bloquante, mesure de la
   reproductibilité et primitives de snapshot/restore.
3. **Évaluation et baselines** : suites de tests figées, métriques, seeds et
   baselines reproductibles.
4. **Moteur transactionnel full-depth** : `ContractIR`, State Fabric, moteur de
   transactions et moniteur de référence.
5. **Actions logicielles typées** : édition, appel d'outil, mise à jour d'état,
   question, abstention et complétion, avec validation et sandbox.
6. **Première boucle de bout en bout et FORMIC-M0** : exécuter de petites tâches
   logicielles complètes et mesurer le gain du runtime seul.
7. **Production d'épisodes** : industrialiser les épisodes d'exécution annotés
   qui serviront à l'entraînement.
8. **Premiers sidecars neuraux et FORMIC-M1** : ajouter de petits composants
   spécialisés, garder le tronc gelé et vérifier chaque gain par ablation.

### Partie 2 : capacités avancées, volontairement différées

La partie 2 n'est pas commencée. Elle viendra seulement après une base mesurée
et validée. Elle pourra inclure :

- sorties à profondeur progressive;
- routage appris;
- budgets de réflexion et de scratch;
- décision apprise « continuer à raisonner ou agir »;
- décodage spéculatif;
- rollback contrôlé des états GDN;
- intégration MTP;
- tâches longues et entraînement outcome/preference;
- reinforcement learning;
- serving multi-GPU et chemin vision.

Ces capacités ne sont pas supprimées du projet. Elles sont séquencées après les
fondations pour éviter de construire des mécanismes avancés sur une baseline
non mesurée.

## Où en est le projet ?

Le dépôt est actuellement dans la première étape de la partie 1.

### Éléments déjà disponibles

- structure du dépôt et conventions scientifiques;
- schéma de configuration strict;
- registre d'expériences `EXP-...`;
- chargement text-only BF16 du checkpoint;
- inventaire strict des tenseurs;
- mapping bijectif des poids;
- vue des 16 groupes hybrides;
- 17 frontières inertes;
- runner de génération native;
- contrôles de déterminisme et rapports d'expérience;
- tests weight-free et garde-fous A11/A12.

### État de la validation du backbone

La vérification préliminaire de SPEC-01 est à **8/9** :

- le prefill Formic/HF est bit-identique sur les six prompts;
- le chargement strict et la structure du modèle sont validés;
- les hooks no-op sont bit-inertes;
- Formic et HF sont exacts dans plusieurs contrôles alignés;
- une divergence de décodage CUDA avec cache reste observée entre certaines
  exécutions indépendantes.

Les diagnostics disponibles indiquent que cette divergence est très probablement
liée à un effet déterministe d'ordinal ou d'initialisation du backend CUDA/GDN,
et non à un défaut logique démontré du wrapper Formic. Le même phénomène est
observable avec le modèle Hugging Face stock, et Formic et HF redeviennent
bit-identiques lorsque leurs conditions d'exécution sont correctement alignées.

Cette conclusion permet de poursuivre le développement de Formic, mais ne
transforme pas automatiquement la vérification 8/9 en validation formelle 9/9.
La reproductibilité inter-processus CUDA et la définition des tolérances restent
documentées comme une question ouverte.

Voir [`STATUS.md`](STATUS.md) et le [rapport de clôture de l'analyse de
divergence](reports/step1_formic_hf_divergence_conclusion.md) pour le détail des
mesures, des limites et de la décision recommandée.

## Démarrage rapide

Les commandes suivantes vérifient le projet sans charger les poids, et prennent
quelques secondes :

```bash
export PYTHONPATH=$PWD

python -m formic.cli verify
python -m formic.cli structure
python -m formic.cli inventory
python -m formic.cli config
python -m formic.cli env
python -m pytest tests/ -q
```

Les commandes qui chargent le checkpoint nécessitent l'environnement et le
fichier de poids local :

```bash
python -m formic.cli load
python -m formic.cli generate --prompt "..." --chat
python scripts/step1_acceptance.py --stage all
```

Le chargement du modèle nécessite plusieurs dizaines de gigaoctets de mémoire
et peut prendre plusieurs minutes. Les mesures décisives doivent utiliser le
checkpoint, la configuration et l'environnement documentés dans les rapports.

## Organisation du dépôt

```text
formic/
├── formic/                package Python principal
│   ├── backbone/          checkpoint, groupes, frontières, runner
│   ├── config/            schéma et chargement de configuration
│   ├── science/           déterminisme, environnement, registre d'expériences
│   ├── runtime/           moteur transactionnel à venir
│   ├── contracts/         ContractIR et compilateur à venir
│   ├── state/             State Fabric à venir
│   ├── actions/           actions typées à venir
│   ├── validation/        moniteur, sandbox et validations à venir
│   ├── eval/              évaluations et baselines à venir
│   ├── episodes/          production d'épisodes à venir
│   └── sidecars/          composants neuraux à venir
├── configs/               configurations YAML et prompts figés
├── docs/adr/              décisions d'architecture
├── tests/                 tests, weight-free par défaut
├── scripts/               outils d'acceptance et expériences
├── experiments/           registre append-only des expériences
├── reports/               rapports de développement et de mesure
├── artifacts/             sorties d'exécution, non versionnées
├── PROJECT.md             contexte complet du projet
└── STATUS.md              tableau de statut courant
```

## Règles de contribution

Avant toute modification importante, lire [`docs/conventions.md`](docs/conventions.md).
Les principes essentiels sont :

- l'audit du checkpoint est l'autorité technique principale;
- les étapes du projet sont strictement ordonnées;
- une décision d'architecture ne doit pas être inventée implicitement;
- tout nouveau comportement est derrière un flag désactivé par défaut;
- le mode « tout désactivé » doit préserver le comportement Qwen;
- toute mesure doit être reproductible et accompagnée de ses métadonnées;
- les cellules Qwen ne doivent pas être réécrites;
- les contraintes d'audit A1 à A12 doivent être respectées;
- aucune étape suivante ne doit être lancée avant la validation de la précédente.

## Documentation principale

- [`PROJECT.md`](PROJECT.md) : contexte, motivation, audit, architecture CAPE-R et
  plan complet;
- [`STATUS.md`](STATUS.md) : état courant des briques et des étapes;
- [`docs/conventions.md`](docs/conventions.md) : règles de travail et contraintes
  A1-A12;
- [`docs/adr/`](docs/adr/) : décisions d'architecture;
- [`reports/`](reports/) : résultats techniques et rapports d'étape;
- [`experiments/REGISTRY.md`](experiments/REGISTRY.md) : registre des expériences.

## Résumé

Uly Code est le système agentique qui comprend, cadre et pilote les tâches de
software engineering. Formic est le modèle d'exécution conçu pour transformer
ces tâches cadrées en actions logicielles précises, validées et vérifiables.

Le projet commence volontairement par une fondation stricte : préserver le
comportement du modèle Qwen de base, mesurer l'environnement, construire l'état
et les transactions, puis ajouter progressivement les capacités spécialisées.
La performance future de Formic ne sera pas évaluée uniquement par la qualité
de son texte, mais par sa capacité à produire des changements corrects,
respecter les contraintes, récupérer des erreurs et terminer avec des preuves.
