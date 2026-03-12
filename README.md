# 🚘 Détection de Véhicules — YOLO vs SVM + HOG : Benchmark Comparatif

Quand on prototype un pipeline de Computer Vision, la question n'est pas
"quel modèle est le meilleur en absolu" — c'est "quel modèle est le meilleur
*pour ce cas d'usage, ces contraintes de latence, et ce budget de calcul*".

Ce projet implémente deux approches de détection de véhicules et les compare
rigoureusement : deep learning (YOLOv3) vs apprentissage classique (SVM + HOG features).
L'objectif n'est pas de déclarer un gagnant, mais de comprendre les trade-offs.

---

## Deux pipelines, deux philosophies

```
  Approche 1 : Deep Learning                Approche 2 : ML Classique
  (yolo_pipeline.py)                        (svm_pipeline.py)
  ─────────────────────────────             ────────────────────────────────

  Image/Frame                               Image/Frame
      │                                         │
      ▼                                         │  Fenêtre glissante
  ┌──────────────┐                          ┌───▼──────────────────┐
  │    YOLOv3    │                          │  HOG Feature Extract  │
  │              │                          │                       │
  │  Poids       │                          │  - orientations: 9    │
  │  pré-        │                          │  - pixels_per_cell:   │
  │  entraînés   │                          │    (8, 8)             │
  │  (COCO)      │                          │  - cells_per_block:   │
  │              │                          │    (2, 2)             │
  └──────┬───────┘                          └───────────┬───────────┘
         │                                              │
         │  Bounding boxes +                            │  Feature vector
         │  classes + scores                            ▼
         │                                  ┌───────────────────────┐
         ▼                                  │    SVM Classifier     │
  ┌──────────────┐                          │                       │
  │  NMS Filter  │                          │  + Color Histograms   │
  │              │                          │  + Spatial Binning    │
  │  seuil: 0.6  │                          │                       │
  └──────────────┘                          └───────────────────────┘
```

---

## Extraction des features HOG (approche classique)

```python
from skimage.feature import hog
from skimage import color
import numpy as np

def extract_hog_features(img, params=None):
    """
    Extrait les features HOG + histogramme couleur pour le classificateur SVM.
    L'optimisation des paramètres HOG a un impact majeur sur les performances.
    """
    if params is None:
        params = {
            'orientations':    9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'color_space':     'YCrCb'  # meilleur que RGB pour les véhicules
        }

    # Convertir dans l'espace colorimétrique optimal
    if params['color_space'] == 'YCrCb':
        img = color.rgb2ycbcr(img)
    elif params['color_space'] == 'HSV':
        img = color.rgb2hsv(img)

    # Features HOG sur chaque canal
    hog_features = []
    for channel in range(img.shape[2]):
        features, _ = hog(
            img[:, :, channel],
            orientations    = params['orientations'],
            pixels_per_cell = params['pixels_per_cell'],
            cells_per_block = params['cells_per_block'],
            visualize       = True,
            feature_vector  = True
        )
        hog_features.append(features)

    # Histogramme couleur (32 bins par canal)
    color_hist = np.concatenate([
        np.histogram(img[:, :, c], bins=32, range=(0, 256))[0]
        for c in range(img.shape[2])
    ])

    return np.concatenate(hog_features + [color_hist])
```

---

## Résultats du benchmark

```
  ┌─────────────────────────┬─────────────┬─────────────┐
  │ Métrique                │   YOLOv3    │  SVM + HOG  │
  ├─────────────────────────┼─────────────┼─────────────┤
  │ Précision               │    94.2%    │    88.7%    │
  │ Rappel                  │    91.8%    │    83.4%    │
  │ F1-score                │    0.930    │    0.859    │
  ├─────────────────────────┼─────────────┼─────────────┤
  │ Temps par frame (CPU)   │   5-10s     │   0.3-0.8s  │
  │ Temps par frame (GPU)   │   ~30ms     │   N/A       │
  ├─────────────────────────┼─────────────┼─────────────┤
  │ Mémoire requise         │   ~800MB    │   ~50MB     │
  │ GPU requis              │  Fortement  │    Non      │
  │                         │  recommandé │             │
  ├─────────────────────────┼─────────────┼─────────────┤
  │ Cas difficiles (nuit,   │   Robuste   │  Dégradation│
  │ occlusion, angle)       │             │  significative│
  └─────────────────────────┴─────────────┴─────────────┘

  Conclusion : YOLOv3 domine sur la précision et la robustesse.
  SVM + HOG reste pertinent pour les environnements sans GPU
  ou les systèmes embarqués avec contraintes de mémoire.
```

---

## Optimisation HOG — impact des hyperparamètres

```python
import itertools
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

# Grid search sur les paramètres HOG
orientations_list    = [6, 9, 12]
pixels_per_cell_list = [(8, 8), (12, 12), (16, 16)]

best_score, best_params = 0, {}

for orients, ppc in itertools.product(orientations_list, pixels_per_cell_list):
    features = [extract_hog_features(img,
                    {'orientations': orients, 'pixels_per_cell': ppc,
                     'cells_per_block': (2, 2), 'color_space': 'YCrCb'})
                for img in images]

    X = np.array(features)
    score = cross_val_score(LinearSVC(), X, labels, cv=5).mean()

    if score > best_score:
        best_score, best_params = score, {'orientations': orients, 'ppc': ppc}
    print(f"orient={orients}, ppc={ppc} → CV score: {score:.4f}")

# Résultat optimal : orientations=9, pixels_per_cell=(8,8) → 0.9187
```

---

## Ce que j'ai appris

L'optimisation des features HOG m'a montré quelque chose de contre-intuitif :
plus de `orientations` ne donne pas toujours de meilleurs résultats. Au-delà de 9,
le vecteur de features devient trop grand, le SVM overfit légèrement et les temps
d'inférence augmentent. C'est un exemple classique du trade-off complexité/performance.

Ce benchmark m'a aussi appris à raisonner en termes de **contraintes de déploiement**
avant de choisir un modèle. Sur un système embarqué dans un véhicule (CPU limité,
pas de GPU), SVM + HOG est la bonne réponse. Sur un serveur cloud avec GPU pour
de la vidéosurveillance, YOLO s'impose. La réponse juste dépend du contexte.

---

*Projet réalisé dans le cadre de ma formation ingénieur — ENSET Mohammedia*
*Par **Abderrahmane Elouafi** · [LinkedIn](https://www.linkedin.com/in/abderrahmane-elouafi-43226736b/) · [Portfolio](https://my-first-porfolio-six.vercel.app/)*
