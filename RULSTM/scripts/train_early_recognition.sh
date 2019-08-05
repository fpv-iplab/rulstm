# RGB Branch
python main.py train data models --modality rgb --task early_recognition --sequence_completion --visdom --epochs 200
python main.py train data models --modality rgb --task early_recognition --visdom --epochs 200

# Optical Flow Branch
python main.py train data models --modality flow --task early_recognition --sequence_completion --visdom --epochs 200
python main.py train data models --modality flow --task early_recognition --visdom --epochs 200

# Object Branch
python main.py train data models --modality obj --task early_recognition --sequence_completion --visdom --epochs 200 --feat_in 352
python main.py train data models --modality obj --task early_recognition --visdom --epochs 200 --feat_in 352

