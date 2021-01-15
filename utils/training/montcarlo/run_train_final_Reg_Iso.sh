#python train_final_Reg_Iso.py --EBEE EB --config config_qRC_training_ChI_5M.yaml --n_evts 4700000 --backend dask --clusterid 192.33.123.23:8786 --n_jobs 70
#python train_final_Reg_Iso.py --EBEE EB --config config_qRC_training_PhI_5M.yaml --n_evts 4700000 --backend dask --clusterid 192.33.123.23:8786 --n_jobs 70

#python train_final_Reg_Iso.py --EBEE EE --config config_qRC_training_ChI_5M.yaml --n_evts 4700000 --backend dask --clusterid 192.33.123.23:8786 --n_jobs 70
#python train_final_Reg_Iso.py --EBEE EE --config config_qRC_training_PhI_5M.yaml --n_evts 4700000 --backend dask --clusterid 192.33.123.23:8786 --n_jobs 70

python train_final_Reg_Iso.py --EBEE EB --config config_qRC_training_ChI_5M.yaml --n_evts 4700000
python train_final_Reg_Iso.py --EBEE EB --config config_qRC_training_PhI_5M.yaml --n_evts 4700000

python train_final_Reg_Iso.py --EBEE EE --config config_qRC_training_ChI_5M.yaml --n_evts 4700000
python train_final_Reg_Iso.py --EBEE EE --config config_qRC_training_PhI_5M.yaml --n_evts 4700000
