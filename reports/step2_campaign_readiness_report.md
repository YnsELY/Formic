# SPEC-02 — Readiness de la campagne A40 après diagnostics

**Statut : implémentation locale prête à vérifier ; calibration A40 et verdict
officiel non exécutés. ADR-0005 reste `PROPOSED`.**

## Résultats A40 pris en compte

Le diagnostic schedule-matrix r6 a terminé sans OOM. Ses six configurations
ont trois répétitions complètes, avec les deux dernières exactes. Le calendrier
alterné rend RR, NN et RN stables dans cet artefact ; le calendrier séquentiel
conserve un changement brut RR. L'allocation revient à 34 550 975 488 octets
après chauffe et après matrice, avec un pic à 37 258 788 864 octets. Aucune
cause n'est attribuée.

Le balanced crossover r2 a ensuite mesuré :

- 1 536/1 536 comparaisons endpoint à slot relatif apparié exactes, delta et KL
  nuls ;
- 336/384 groupes dont l'empreinte brute change avec la position ordinale ;
- 612/768 inversions RN/NR brutes exactes ;
- aucun OOM, pic alloué 37 258 330 112 octets et retour à
  34 550 975 488 octets après chaque round.

Ces chiffres sont rapportés sans conclusion causale. Ils démontrent seulement
que les empreintes prises à des ordinaux processus distincts ne peuvent pas
être confondues avec le contraste endpoint apparié.

Références fournies par la campagne pod : commit source
`ccb2b3642147ec85c5df14938011744ee86b8bb6`, analyse r6
`a17bafb3b2e6224d66fa78f62a7bcab5d131c50e3c29a37089162663f73517ff`,
mémoire r6
`a952899e05f5dc3697e5e7fe3c566935a4e756544c5731588e7c865115979168`,
analyse r2
`cb61e8d988be3a36a26c76d37196130ebf745d1d1af5e0e0c3d78988cdfb24df`
et mémoire r2
`7fd3d542f2f8934f478b650857da6ac8d990e33124ced88edd8267cbab656b5c`.

## Corrections intégrées

1. La gate legacy n'utilise plus la succession endpoint séquentielle. Elle
   exécute quatre rounds Latin ABBA : RR, NN, RN et NR occupent chacun les
   quatre ordinaux de configuration. Seuls les contrastes endpoint appariés et
   leurs signatures last-two sont bloquants. Les empreintes ordinales brutes
   restent dans l'artefact, non bloquantes. La signature compare les invariants
   de différence endpoint, jamais l'ID top-1 absolu ni une empreinte brute.
2. Le plancher de bruit est RR/NN/RN sous le calendrier alterné r6, logits
   seulement, trois prompts et trois répétitions. Son maximum RR est réellement
   injecté dans chaque ligne logits de `tolerances.candidate.json`, puis conservé
   à la promotion ; l'ancien zéro codé en dur a été supprimé.
3. Le protocole de calibration mesure maintenant les effets demandés : prefill
   segmenté contre prefixes complets stock, et decode cached contre recalcul
   complet stock pour court/moyen. Le long reste cached/cached conformément au
   plan validé qui exclut son recalcul complet.
4. La sonde 64 compare réellement cached et recalcul complet. Ses formes 9–64
   sont chauffées séparément ; une chauffe limitée à huit ne couvre pas les
   longueurs de séquence/cache rencontrées.
5. Les continuations greedy et les trois continuations échantillonnées sont
   générées une fois par la référence puis forcées. Le seed n'entre pas dans
   les forwards mesurés.
6. Snapshot/restore est mesuré avant calibration, vérifie la stabilité de ses
   deux dernières traces, puis est adjudicé contre le seuil candidat short
   cached sans nouveau forward. Une reprise recharge correctement son artefact
   déjà commité.
7. Le hash de contenu du backbone A40 est commité et comparé avant toute mesure :
   `74e1813c29b065406f4b772ed7c9059b8455428bff9aa6e572645cf09743c662`,
   851 tenseurs et 53 791 996 928 octets.
8. Le placement A40 est plafonné à 35 GiB par défaut. Les artefacts restent
   atomiques par cas et le lanceur ne stoppe ni ne demande de stopper le pod.

## Coût et ordre final

L'ordre demeure : preflight, inertie du traceur, continuité legacy, plancher de
bruit, snapshot/restore réel, continuations de référence, classes courte,
moyenne, longue, puis sonde 64. Le plan contient 8 549 forwards et environ
65,47 GiB de transferts mesurés. La projection historique est 7,31 h ; après le
preflight, les chronométrages du pod produisent l'estimation par poste qui
remplace cette projection sans bloquer l'enchaînement.

## Traitement des contraintes d'audit

| Contrainte | Traitement |
|---|---|
| A1 | Aucun cache n'est protégé par `use_cache=False`; le recalcul ne reçoit aucun cache et chaque chemin cached possède un cache frais. |
| A2 | Tous les `DynamicCache` explicites reçoivent `model.config`. |
| A3 | Aucun `crop()` n'est utilisé dans la campagne ou snapshot/restore. |
| A4 | Chaque restauration deep-clone l'état ; snapshot, branche A et branche B sont vérifiés par identité de stockage puis par mutation. |
| A5 | Aucune convention RMSNorm ou gated norm n'est réimplémentée ou uniformisée. |
| A6 | L'état attaché au modèle est capturé séparément ; l'absence texte-only de `rope_deltas` reste explicite. |
| A7 | Le point d'entrée reste text-only ; aucune tour vision n'est construite. |
| A8 | Batch 1 strict et aucun padding. |
| A9 | K/V sont capturés tels que stockés, sans renormalisation ni réinterprétation. |
| A10 | MTP reste exclu et inactif. |
| A11 | Seuls les modules HF intacts, le runner existant et des observateurs inertes sont utilisés. |
| A12 | Inventaire strict avant chargement, égalité post-load et hash 851 tenseurs obligatoire. |

## Condition de sortie restante

La vérification locale de cette implémentation rapporte 358 tests weight-free,
182 gardes A11/A12/frontières, `formic verify` PASS et
`identity-check --toy` PASS. Le hash de configuration résolue avec le placement
35 GiB est
`b0b2ca19b553ea06f41b0cf4f876107bfd843ad08d0ccf5c281123ce3c7965b5`,
identique au crossover r2 réussi.

Le développement hors GPU ne peut pas produire les tolérances réelles ni le
verdict officiel. Après passage des gates locales, la seule étape restante est
une nouvelle campagne A40 avec ce lanceur. Elle doit produire les mesures
incrémentales, `tolerances.candidate.json`, l'adjudication snapshot et
`verdict.candidate.json`. La promotion, le rapport final et l'acceptation de
l'ADR restent des actions humaines distinctes après analyse des nombres.
