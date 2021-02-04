#python train_qRC_MC.py --EBEE EB --config config_qRC_training_5M.yaml --n_evts 1000000 --backend ray --clusterid 192.33.123.23:8786 -s 1
python train_qRC_MC.py --EBEE EB --config config_qRC_training_5M.yaml --n_evts 1000000 --backend ray --clusterid 192.33.123.23:6379 -s 2

#python train_qRC_MC.py --EBEE EE --config config_qRC_training_5M.yaml --n_evts -1 --backend dask --clusterid 192.33.123.23:8786 -s 1
#python train_qRC_MC.py --EBEE EE --config config_qRC_training_5M.yaml --n_evts -1 --backend dask --clusterid 192.33.123.23:8786 -s 2
