#!/bin/bash
python classifier/evaluate_student.py '{"dataset": "WebrequestDataset", "dataset_args": {"num_workers": 4, "max_student_train_queries":700}, "model": "Student", "aggregator": "LaplaceAggregator", "aggregator_args": {"gamma": 0.05}, "network": "fcn1", "network_teacher": "fcn1", "exp_name": "2020-11-30-01:14-messy-earthworm", "network_args": {}, "train_args": {"teachers_exp_name": "2020-07-21-23:19-cautious-roadrunner", "batch_size": 64, "epochs": 25, "learning_rate": 0.003}}' --use_wandb
