#!/bin/bash 


SOCRAETES="SOCRAETES_SM141K07L_"
SolarCell="SM141K07L"
dateVar=$(date +"%y%m%d_%H%M%S")
extentionVar=".hdf5"

filenameVar=$SOCRAETES$dateVar$extentionVar

cd ic_solar_energy_prediction/src/PROGNOES/SOCRAETES/
python3 record.py --port /dev/ttyACM0 --mode commit-to-file --file $filenameVar --solar_cell $SolarCell --duration 32 --environment indoor --temperature 5.2 --lux 50 --weather sunny --country Germany --city Essen --street Kuglerstr --postcode 45144

python3 record.py --port /dev/ttyACM0 --duration 10 --environment indoor --lux 50 --weather sunny --country Germany --city Essen --street Kuglerstr --postcode 45144
