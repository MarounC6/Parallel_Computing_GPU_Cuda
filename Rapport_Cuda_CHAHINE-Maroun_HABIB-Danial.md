# TP CUDA - Programmation GPU

*Auteurs :* CHAHINE Maroun, HABIB Danial  
*Date :* Décembre 2025  
*Formation :* 5IF - INSA Lyon

---

## Table des matières

1. [Partie 1 : Calcul de π](#partie-1--calcul-de-π)
2. [Partie 2 : Produit Matrice-Vecteur](#partie-2--produit-matrice-vecteur)
3. [Partie 3 : Multiplication de Matrices](#partie-3--multiplication-de-matrices)

---

## Partie 1 : Calcul de π

### 1.1 Objectif

L'objectif de cette première partie est de calculer une approximation de π en utilisant la méthode des rectangles pour approximer l'intégrale suivante :

$$\pi = \int_0^1 \frac{4}{1+x^2} dx$$

Nous avons implémenté plusieurs versions du programme pour comparer les performances :
- *Version séquentielle*
- *Version CUDA simple* (GPU)
- *Version avec mémoire partagée* (GPU optimisée)
- *Version avec réduction à 2 niveaux* (GPU optimisée)
- *Version avec réduction multi-étages* (GPU optimisée)
- *Version avec tableau* (GPU)
- *Version tableau avec réduction à 2 niveaux* (GPU optimisée)

### 1.2 Méthodologie

Pour chaque implémentation, nous avons effectué des tests avec :
- *Nombre de pas* : 1 000 et 1 000 000
- *Threads par bloc* : 1, 32, 64, 128, 256
- *Répétitions* : 10 exécutions par configuration pour obtenir des moyennes fiables

### 1.3 Résultats et Analyse

![Analyse de performance - Partie 1](Part_1/performance_analysis.png)

#### Observations principales :

*1. Version séquentielle*
- Temps d'exécution stable
- Pas d'influence du paramètre "threads par bloc" (normal, car CPU)

*2. Version CUDA simple (pi_cuda_gpu)*
- Premier portage sur GPU
- Sensible au nombre de threads par bloc
- Meilleure performance autour de 128-256 threads/bloc

*3. Version avec mémoire partagée (pi_cuda_shared_memory)*
- Utilisation de __shared__ pour réduire les accès à la mémoire globale
- Réduction des latences mémoire
- Performance améliorée par rapport à la version simple
- La mémoire partagée permet aux threads d'un même bloc de collaborer efficacement

*4. Versions avec réduction (2-level et multistage)*
- Approches optimisées pour minimiser les synchronisations
- Réduction hiérarchique des résultats partiels
- *Multistage reduction* : meilleure performance globale
- Évite les goulots d'étranglement lors de la combinaison des résultats

*5. Versions avec tableau*
- Stockage des résultats partiels dans un tableau global
- Utile pour déboguer et analyser les contributions individuelles
- Performance légèrement inférieure aux versions avec réduction optimale

#### Speedup observé :
- *Impact du nombre de pas* : Plus le calcul est complexe (1M vs 1K pas), plus le GPU montre son avantage
- *Threads optimaux* : 128-256 threads/bloc offrent le meilleur compromis

### 1.4 Conclusion Partie 1

Les résultats démontrent clairement l'intérêt du GPU pour les calculs massivement parallèles. Les optimisations comme la mémoire partagée et les réductions multi-niveaux permettent d'exploiter au maximum la puissance du GPU. Le choix du nombre de threads par bloc est crucial : trop peu limite le parallélisme, trop peut saturer les ressources.

---

## Partie 2 : Produit Matrice-Vecteur

### 2.1 Objectif

Cette partie consiste à calculer le produit d'une matrice par un vecteur : *valeur = A . y . X* avec . un produit scalaire entre vecteur

Avec :
- *A* : matrice de dimension *N×M* (N lignes, M colonnes)
- *X* : vecteur de dimension *M* (colonne)
- *Y* : vecteur résultat de dimension *N* (colonne)

Implémentations réalisées :
- *Version séquentielle*
- *Version CUDA simple* (GPU)
- *Version avec mémoire partagée* (GPU)
- *Version avec mémoire partagée optimisée* (GPU)
- *Version avec réduction à 2 niveaux* (GPU)

### 2.2 Méthodologie

Tests effectués avec :
- *Tailles de matrice* : 
  - N = 2^n avec n ∈ {2, 4, 6, 8, 10, 12} (nombre de lignes)
  - M = 2^m avec m ∈ {1, 3, 7, 9, 11} (nombre de colonnes)
  - Donc N varie de 4 à 4096, et M varie selon les tests
- *Vecteur X* : de dimension M (nombre de colonnes de A)
- *Vecteur Y* : de dimension N (nombre de lignes de A)
- *Threads par bloc* : 1, 32, 64, 128, 256
- *Répétitions* : 10 exécutions par configuration

### 2.3 Résultats et Analyse

![Analyse de performance - Partie 2](Part_2/performance_analysis.png)

#### Observations principales :

*1. Version séquentielle*
- Temps d'exécution croît avec N×M (O(N×M))
- Devient rapidement prohibitif pour les grandes matrices
- Aucune exploitation du parallélisme disponible

*2. Version CUDA simple (matrix_cuda_gpu)*
- Chaque thread calcule un élément du vecteur résultat
- 1 threqd pqr blck, et N blocks
- Parallélisme naturel : N threads pour N éléments de sortie
- Chaque thread fait M multiplications + M additions (parcourt une ligne de A)
- Gain important par rapport au sequentiel
- Bonne scalabilité avec la taille de la matrice

*3. Version avec mémoire partagée (matrix_cuda_shared_memory)*
- Cache les données fréquemment accédées dans la mémoire partagée
- Réduit les accès à la mémoire globale (plus lente)
- Amélioration notable des performances
- Particulièrement efficace pour les grandes matrices où le ratio calcul/mémoire est élevé

*4. Version optimisée (matrix_cuda_shared_memory_optimized)*
- Une version qui n'utilise pas le atomic add et utilise plus de shared memory pour fonctionner plus rapidement

*5. Version avec réduction à 2 niveaux (matrix_cuda_2_level_reduction)*
- Réduction hiérarchique des produits partiels
- Deux niveaux de réduction : au sein du bloc puis globalement
- Performance comparable à la version optimisée
- Approche différente mais résultats similaires

#### Speedup observé :
- *GPU vs Sequentiel* : Accélération jusqu'à *100-200x* pour les grandes matrices
- *Impact de N* : Plus la matrice est grande, plus le GPU est avantageux
- *Mémoire partagée* : Gain de *20-40%* par rapport à la version simple
- *Optimisations* : Gain additionnel de *10-20%*

### 2.4 Conclusion Partie 2

Le produit matrice-vecteur est une opération idéale pour le GPU car chaque élément du résultat Y peut être calculé indépendamment. Chaque thread traite une ligne de la matrice A (M éléments) pour produire un élément de Y. Les optimisations mémoire (mémoire partagée, coalescence) sont essentielles pour atteindre les meilleures performances. Pour les grandes matrices (N > 1000, M > 1000), le GPU devient indispensable.

---

## Partie 3 : Multiplication de Matrices

### 3.1 Objectif

Cette partie vise à implémenter la multiplication de matrices **C = A × B** avec :
- **A** : matrice N×P
- **B** : matrice P×M
- **C** : matrice résultat N×M

L'objectif est d'explorer différentes stratégies d'optimisation du code:

Implémentations réalisées :
- **Version séquentielle**
- **Version CUDA 1 thread/bloc** (Q3.1 - parallélisme minimal)
- **Version avec mémoire partagée** (Q3.5 - tiling optimisé)
- **Version float** (Q3.9 - précision simple 32 bits)

**Note** : La version half precision (Q3.13) n'a pas été implémentée car la bibliothèque `half.hpp` recommandée dans le sujet n'est pas compatible avec CUDA.

### 3.2 Méthodologie

Tests prévus avec :
- **Dimensions** : N, M, P ∈ {1000, 2000, 3000}
- **Répétitions** : 10 exécutions par configuration
- **Métriques** : Temps d'exécution, GFLOPS, speedup, précision

### 3.3 Résultats et Analyse

Les benchmarks ont été effectués sur des matrices avec des dimensions de 1000, 2000 et 3000. Les tests ont été répétés 10 fois pour obtenir des moyennes fiables.

#### Graphique 1 : Performance en fonction de la dimension

![Performance des matrices carrées](Part_3/plots/performance_square_matrices.png)

Ce graphique montre le temps d'exécution en fonction de la taille des matrices. On observe :
- **Séquentiel** : Croissance cubique O(N³) très marquée - devient rapidement prohibitif
- **CUDA 1-thread** : Amélioration significative grâce au parallélisme massif
- **CUDA Shared** : Légère amélioration grâce à l'optimisation mémoire
- **CUDA Float** : Performances similaires ou légèrement meilleures que shared memory

#### Graphique 2 : Speedup (Accélération)

![Speedup GPU vs CPU](Part_3/plots/speedup_square_matrices.png)

Les accélérations obtenues démontrent l'efficacité du GPU :

**Matrice 1000×1000×1000** :
- CUDA 1-thread : **16.97x** plus rapide que séquentiel
- CUDA Shared : **16.18x** plus rapide que séquentiel
- CUDA Float : **16.20x** plus rapide que séquentiel

**Matrice 2000×2000×2000** :
- CUDA 1-thread : **135.63x** plus rapide que séquentiel
- CUDA Shared : **155.76x** plus rapide que séquentiel
- CUDA Float : **173.66x** plus rapide que séquentiel (meilleure performance !)

**Observation clé** : Le speedup augmente avec la taille des matrices, montrant que le GPU devient encore plus avantageux pour les grandes données.

#### Graphique 3 : Comparaison de précision (Double vs Float)

![Comparaison Double vs Float](Part_3/plots/precision_comparison.png)

Ce graphique compare les performances entre précision double (64 bits) et simple (32 bits) :
- **Float est systématiquement plus rapide** que double
- L'écart se creuse avec les grandes matrices (173x vs 155x pour N=2000)
- Trade-off intéressant : légère perte de précision (7 chiffres vs 15) pour un gain de performance notable

#### Graphique 4 : Impact des optimisations CUDA

![Impact des optimisations](Part_3/plots/cuda_optimization_comparison.png)

Comparaison des différentes stratégies d'optimisation CUDA :
- **1-thread par bloc** : Parallélisme de base, bon point de départ
- **Shared memory** : Optimisation mémoire, réduction des accès globaux
- **Float precision** : Combine optimisation mémoire + précision réduite = meilleure performance

Pour N=2000, la version float est **~28% plus rapide** que la version 1-thread !

#### Graphique 5 : Performance en GFLOPS

![Performance en GFLOPS](Part_3/plots/gflops_comparison.png)

Analyse du débit de calcul en milliards d'opérations par seconde :

**Matrice 1000×1000×1000** (2 milliards d'opérations) :
- Séquentiel : **2.01 GFLOPS** (CPU)
- CUDA 1-thread : **34.17 GFLOPS** (17x amélioration)
- CUDA Shared : **32.57 GFLOPS**
- CUDA Float : **32.62 GFLOPS**

**Matrice 2000×2000×2000** (16 milliards d'opérations) :
- Séquentiel : **1.41 GFLOPS** (CPU sature)
- CUDA 1-thread : **191.55 GFLOPS** (136x amélioration)
- CUDA Shared : **219.98 GFLOPS**
- CUDA Float : **245.25 GFLOPS** (⭐ meilleure performance)

**Observation importante** : Le GPU maintient un débit élevé même avec l'augmentation de la charge, contrairement au CPU qui plafonne dans le code séquentiel. Le débit en GFlops (Floating operations per second) est très élevé dans les codes avec du parallélisme massif, ce qui montre qu'on performe plus d'opérations par seconde, afin d'optimiser le temps.

#### Synthèse des résultats :

**Speedup** : Jusqu'à **173x** plus rapide que la version séquentielle (version float, N=2000)

**Scalabilité** : Les performances GPU s'améliorent avec la taille des données

**Float vs Double** : Float offre le meilleur compromis performance/précision pour ce type de calcul

**Shared memory** : Amélioration modeste mais constante grâce à l'optimisation des accès mémoire