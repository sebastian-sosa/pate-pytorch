# PATE

## Description
The following is a PyTorch implementation of the technique [Private Aggregation of Teacher Ensembles](https://arxiv.org/abs/1610.05755) applied to the classification of malicious web request logs.

## Project structure
- `/tasks` contains bash scripts that invoke scripts  
- `/classifier/*.py` are the scripts that conduct the training steps of PATE, using the abstractions found on the subfolders  
- `/classifier/weights/` contains the weights.  
- If you want to train a 100 or 250-model teacher ensemble, you should train them with - `train_teachers.py` and save them in the corresponding folder. On `train_teachers.sh`, setting `--gpus` to > 1 may help you speed up the training process with distributed training, this should work on ORT cluster.  
- Once the teacher ensemble is trained, `python3 classifier/aggregate_labels.py` helps obtaining target labels for the student model. The output of this script is a numpy file `student_labels.npy` with the labels.
- Once you have `student_labels.npy`, `classifier/train_student.py` helps training a student model with a given dataset and student labels.
- `classifier/evaluate_student.py` computes the model metrics and privacy cost of the student.
