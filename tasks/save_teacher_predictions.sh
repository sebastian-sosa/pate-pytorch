#!/bin/bash
python classifier/save_teacher_predictions.py '{"dataset": "WebrequestDataset", "dataset_args": {"num_workers": 4}, "network_teacher": "fcn1", "network_args": {}, "train_args": {"teachers_exp_name": "2020-07-21-23:19-cautious-roadrunner", "batch_size": 128}}'