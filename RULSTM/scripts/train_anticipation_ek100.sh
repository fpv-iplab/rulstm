mkdir -p models/ek100

# RGB branch
python main.py train data/ek100 models/ek100 --modality rgb --sequence_completion --mt5r --num_class 3806
python main.py train data/ek100 models/ek100 --modality rgb --mt5r --num_class 3806

# Optical Flow branch
python main.py train data/ek100 models/ek100 --modality flow --sequence_completion --mt5r --num_class 3806
python main.py train data/ek100 models/ek100 --modality flow --mt5r --num_class 3806

# Object branch
python main.py train data/ek100 models/ek100 --modality obj --feat_in 352 --sequence_completion --mt5r --num_class 3806
python main.py train data/ek100 models/ek100 --modality obj --feat_in 352 --mt5r --num_class 3806

# Complete architecture with MATT
python main.py train data/ek100 models/ek100 --modality fusion --feat_in 352 --mt5r --num_class 3806
