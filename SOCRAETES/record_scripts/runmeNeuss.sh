#!/bin/bash 


SOCRAETES="SOCRAETES_"
dateVar=$(date +"%y%m%d_%H%M%S")
extentionVar=".hdf5"

filenameVar=$SOCRAETES$dateVar$extentionVar

cd ic_solar_energy_prediction/src/PROGNOES/SOCRAETES/
python3 record.py --port /dev/ttyACM0 --mode commit-to-file --file $filenameVar --duration 32 --environment indoor --lux 50 --weather sunny --country Germany --city Neuss --street Itterstr --postcode 41469

python3 record.py --port /dev/ttyACM0 --duration 32 --environment indoor --lux 50 --weather sunny --country Germany --city Neuss --street Itterstr --postcode 41469
