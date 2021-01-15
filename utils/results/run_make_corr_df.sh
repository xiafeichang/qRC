#python make_corr_df.py --config config_make_corr_df.yaml --EBEE EB -N 4700000 --backend Dask --clusterid 192.33.123.23:8799 --n_jobs 5
#python make_corr_df.py --config config_make_corr_df.yaml --EBEE EE -N 4700000

python make_corr_df.py --config config_make_corr_df.yaml --EBEE EB -N 4700000 --backend Dask --clusterid 192.33.123.23:8786 --n_jobs 15 --final --mvas
#python make_corr_df.py --config config_make_corr_df.yaml --EBEE EE -N 4700000 --backend Dask --clusterid 192.33.123.23:8786 --n_jobs 10 --final --mvas

#python make_corr_df.py --config config_make_corr_df.yaml --EBEE EB -N 4700000 --final --mvas
#python make_corr_df.py --config config_make_corr_df.yaml --EBEE EE -N 4700000 --final --mvas
