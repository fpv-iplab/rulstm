mkdir -p models/ek55

# RGB branch
python main.py train data/ek55 models/ek55 --modality rgb --sequence_completion
python main.py train data/ek55 models/ek55 --modality rgb

# Optical Flow branch
python main.py train data/ek55 models/ek55 --modality flow --sequence_completion
python main.py train data/ek55 models/ek55 --modality flow

# Object branch
python main.py train data/ek55 models/ek55 --modality obj --feat_in 352 --sequence_completion
python main.py train data/ek55 models/ek55 --modality obj --feat_in 352

# Complete architecture with MATT
python main.py train data/ek55 models/ek55 --modality fusion --feat_in 352
