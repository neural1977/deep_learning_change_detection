python trainvgg16.py -d Eurosat -m euvgg.model -l lb.pickle
python trainres50.py -d Eurosat -m eures.model -l lb.pickle
python trainincv3.py -d Eurosat -m euinc.model -l lb.pickle
python trainincresv2.py -d Eurosat -m euincresv2.model -l lb.pickle