# Prévision du Mix Énergétique Français

## Introduction

Le but fondamental de cette initiative est de concevoir un modèle d'intelligence artificielle ayant la capacité de anticiper la composition du mix énergétique en France pour la journée suivante.

### Source des Données ( [Download](https://www.rte-france.com/eco2mix/telecharger-les-indicateurs) )

Nous avons choisi d'utiliser les données du mix énergétique fournies par RTE pour l'entraînement de nos modèles. Ces données, accessibles au public, sont d'une fiabilité élevée. </br>
Les données téléchargées couvrent la période du 01/01/2019 au 31/05/2022 et portent sur la production en mégawatts des énergies primaires: Fioul, Charbon, Gaz, Nucléaire, Éolien, Solaire, Hydraulique et Bioénergies. Les données sont actualisées toutes les 30 minutes, ce qui permet de développer un modèle d'une grande précision. </br>

![CO2_mix](images/energy_mix.png "Energie mix moyen entre 2019 et 2022")

### Approche

L'objectif consiste à créer un modèle de machine learning apte à anticiper le mix énergétique de chaque source d'énergie toutes les 30 minutes en se basant sur les données des jours précédents. Afin de faciliter la comparaison entre mes divers modèles, j'ai opté pour l'utilisation d'une fonction de perte commune. J'ai choisi d'utiliser la fonction de perte qui correspond à la moyenne de la somme des erreurs quadratiques pour chaque intervalle de demi-heure et chaque source d'énergie principale sur l'ensemble des jours d'un batch. <br/>
J'ai également appliqué une max-normalisation à chaque colonne de mes données, étant donné la disparité dans la production électrique de chaque source d'énergie principale. Sans cette normalisation, le modèle aurait eu tendance à privilégier la prédiction de certaines sources d'énergie primaire très productive, alors qu'avec cette approche, chaque source d'énergie a un poids équivalent

## Premier model

### Approche

Mon approche pour construire ce modèle de machine learning repose sur l'intuition que les données quotidiennes présentent des motifs répétitifs. Pour capturer ces schémas temporels, j'ai opté pour une première couche ConvNet 1D.

### DATA

Les données d'entraînement sont segmentées par batch de `nb_days_by_batch` jours. Ces batchs couvrent la plage temporelle du 1er janvier 2019 au 30
mai 2021. Quand à elles, les données de test englobent la période allant du 31 mai 2021 au 31 mai 2022.

L'innput du modèle est constituée du mix énergétique des `nb_days_cnn` jours précédant le jour à prévoir.

### Description du model

### Hyperparamétre

- `nb_epoch`: Nombre d'epochs pour l'entraînement.
- `lr`: Taux d'apprentissage de l'optimiseur.
- `hidden_layer`: Nombre de "channels" produits par la convolution.
- `nb_days_cnn`: Longueur du noyau (en nombre de jours) pour la convolution.
- `nb_days_by_batch`: Nombre de jours par lot (batch).
- `batch_normalization`: Booléen, si défini sur "true", une normalisation est ajoutée entre chaque enregistrement pris à la même heure.

### Analyse des réultats

![Description](images/description.png "Energie mix moyen entre 2019 et 2022")

#### Analyse n°1

La normalisation des inputs n'est pas essentielle, et elle entraîne une perte d'information sur les jours précédents qui présentent des similitudes avec le jour à prédire. J'ai tracé les courbes de perte d'entraînement et de test en fonction des époques (`nb_epoch=300, lr=0.0001, hidden_layer=48, nb_days_cnn=6, nb_days_by_batch=46`)

<div style="display: flex; justify-content: space-between;">
    <figure>
        <img src="images/loss_avec_normalisation.png" alt="Image 1" ">
        <figcaption>Loss function of train (bleu) et test (orange) with normalization</figcaption>
    </figure>
    <figure>
        <img src="images/loss_whitout_normalisation.png" alt="Image 2" >
        <figcaption>Loss function of train (bleu) et test (orange) without normalization</figcaption>
    </figure>
</div>

#### Analyse n°2

L'utilisation d'un modèle avec une convolution (CNN) peut ne pas être appropriée pour certains types de valeurs. En affinant les hyperparamètres, les modèles ne parviennent pas à atteindre une valeur inférieure à 3 ou 4 sur la base de test avant d'overfit.

<p align="center">
    <img src="images/overfiting.png" alt="Image 1"   width="50%">
    <p align="center">Loss function of train (bleu) et test (orange) overfiting</p>
</p>

J'ai choisi d'afficher les prédictions du modèle et les valeurs réelles de production pour un lot de 13 jours sur l'ensemble de test. On remarque que la précision des prévisions dépend fortement du type d'énergie primaire. Pour certains types; tels que l'énergie solaire, hydraulique et Nucléaire; les prévisions sont assez précises en raison de la nature cyclique de leur production quotidienne. Dans le cas du solaire, la production dépend uniquement de l'ensoleillement, et le taux d'ensoleillement est similaire aux jours précédents, contribuant ainsi à des prévisions plus fiables.

<p align="center">
    <img src="images/prev_Solaire.png" alt="Image 1"   width="50%">
    <p align="center">Prévision production solaire ( en rouge) et valeur réelle à J+1 (en vert)</p>
</p>

Cependant, certains types de production ne suivent pas du tout un cycle quotidient. RTE augmente la production uniquement lorsque la demande est ponctuellement importante. Le modèle peine à anticiper cette augmentation et s'adapte qu'au jour suivant. C'est surtout le cas pour la production de gaz, fioul et charbon, des énergies utilisé en dernier recours.

<p align="center">
    <img src="images/prev_gaz.png" alt="Image 1"   width="50%">
    <p align="center">Prévision production dez (en rouge) et valeur réelle à J+1 (en vert)</p>
</p>

Pour ces types d'énergie primaire, l'utilisation d'un modèle CNN pour les prévisions n'est pas optimale. Un modèle RNN pourrait offrir de meilleures performances.

## Deuxième model
