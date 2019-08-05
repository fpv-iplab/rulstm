# RGB branch
python main.py train data models --modality rgb --sequence_completion --visdom
python main.py train data models --modality rgb --visdom

# Optical Flow branch
python main.py train data models --modality flow --sequence_completion --visdom
python main.py train data models --modality flow --visdom

# Object branch
python main.py train data models --modality obj --feat_in 352 --sequence_completion --visdom
python main.py train data models --modality obj --feat_in 352 --visdom

# Complete architecture with MATT
python main.py train data models --modality fusion --feat_in 352 --visdom
