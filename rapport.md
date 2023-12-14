# Prévision du Mix Énergétique Français

## Introduction

Le but fondamental de cette initiative est de concevoir un modèle d'intelligence artificielle ayant la capacité de anticiper la composition du mix énergétique en France pour la journée suivante.

### Source des Données ( [Download](https://www.rte-france.com/eco2mix/telecharger-les-indicateurs) )

Nous avons choisi d'utiliser les données du mix énergétique fournies par RTE pour l'entraînement de nos modèles. Ces données, accessibles au public, sont d'une fiabilité élevée. </br>
Les données téléchargées couvrent la période du 01/01/2019 au 31/05/2022 et portent sur la production en mégawatts des énergies primaires: Fioul, Charbon, Gaz, Nucléaire, Éolien, Solaire, Hydraulique et Bioénergies. Les données sont actualisées toutes les 30 minutes, ce qui permet de développer un modèle d'une grande précision. </br>

![CO2_mix](images/energy_mix.png "Energie mix moyen entre 2019 et 2022")

### Approche

L'objectif consiste à créer un modèle de machine learning apte à anticiper le mix énergétique de chaque source d'énergie toutes les 30 minutes en se basant sur les données des jours précédents. Afin de faciliter la comparaison entre mes divers modèles, j'ai opté pour l'utilisation d'une fonction de perte commune. J'ai opté pour la fonction de perte la moyenne de l'erreur quadratique pour chaque jour d'un batch. <br/>
