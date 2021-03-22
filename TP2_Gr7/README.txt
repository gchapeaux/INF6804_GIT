Pour exécuter le code, exécutez le fichier main.py

Les données nécessaires à YOLO doivent être téléchargées à l'adresse
https://s3-us-west-2.amazonaws.com/static.pyimagesearch.com/opencv-yolo/yolo-object-detection.zip
et extraites à la racine du projet.

Les données requises pour l'analyse doivent être extraites dans le dossier ./data à la racine du projet.
| PETS2006 : http://jacarini.dinf.usherbrooke.ca/static/dataset/baseline/PETS2006.zip
| canoe : http://jacarini.dinf.usherbrooke.ca/static/dataset/dynamicBackground/canoe.zip
| busStation : http://jacarini.dinf.usherbrooke.ca/static/dataset/shadow/busStation.zip
| sofa : http://jacarini.dinf.usherbrooke.ca/static/dataset/intermittentObjectMotion/sofa.zip


Structure du dossier data :

data
| busStation
| | groundtruth
| | input
| canoe
| | groundtruth
| | input
| PETS2006
| | groundtruth
| | input
| sofa
| | groundtruth
| | input