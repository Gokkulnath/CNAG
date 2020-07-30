mkdir ILSVRC
sudo apt install aria2 zip -y
aria2c -x 4 https://archive.org/download/Imagenet_NAG/train.zip -o train.zip
aria2c -x 4 https://archive.org/download/Imagenet_NAG/valid.zip -o valid.zip
wget https://archive.org/download/Imagenet_NAG/LOC_val_solution.csv -O ILSVRC/LOC_val_solution.csv
wget https://archive.org/download/Imagenet_NAG/LOC_synset_mapping.txt -O ILSVRC/LOC_synset_mapping.txt
unzip -qq train.zip -d ILSVRC/ 
unzip -qq valid.zip -d ILSVRC/
