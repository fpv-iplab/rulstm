mkdir models/ek55

# RGB Branch
python main.py train data/ek55 models/ek55 --modality rgb --task early_recognition --sequence_completion --epochs 200
python main.py train data/ek55 models/ek55 --modality rgb --task early_recognition --epochs 200

# Optical Flow Branch
python main.py train data/ek55 models/ek55 --modality flow --task early_recognition --sequence_completion --epochs 200
python main.py train data/ek55 models/ek55 --modality flow --task early_recognition --epochs 200

# Object Branch
python main.py train data/ek55 models/ek55 --modality obj --task early_recognition --sequence_completion --epochs 200 --feat_in 352
python main.py train data/ek55 models/ek55 --modality obj --task early_recognition --epochs 200 --feat_in 352

