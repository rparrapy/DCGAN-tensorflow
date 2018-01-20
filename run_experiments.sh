#!/bin/bash
# Run a batch of experiments for GAN imbalance thesis
# $1 experiment name, i.e. expression to test
# $2 number of runs
# $3 run offset (in case some of previous interruption, start from this number)
# $4 imbalance proportion
# $5 cache directory
# $6 data directory
# $7 expression to use as label
# $8 gpu memory fraction

results_folder=results_$1_$7_$4
mkdir -p $results_folder
touch $results_folder/progress.txt
counter=$3

while [ $counter -lt $2 ]
do
	echo "Attempting run $counter out of $2" >> $results_folder/progress.txt
	python -u /home/rparra/workspace/DCGAN-tensorflow/main.py --dataset celeba --input_height=218 --output_height=109 --input_width=178 --output_width=89 --c_dim=3 --is_train --is_crop --epoch=15 --learning_rate=0.0001 --imbalance_proportion=$4 --cache_dir=$5 --data_dir=$6 --label_attr=$7 --gpu_memory_fraction=$8
	result=$?
	if [ $result -eq 0 ]; then
		((counter++))
  		echo "Completed $counter out of $1 runs" >> $results_folder/progress.txt
		run_folder=$results_folder/run.$counter
		mkdir $run_folder
		mv checkpoint $run_folder
		mv logs $run_folder
		mv samples $run_folder
		mv result_summary_celeba.csv $run_folder
		mv $5/celeba_ytrain.dat $run_folder	
		mv $5/celeba_ytest.dat $run_folder	
		mv $5/celeba_imgs_train $run_folder	
		mv $5/celeba_imgs_test $run_folder	
	else	
  		echo "Execution failed, restarting run" >> $results_folder/progress.txt
		rm -rf checkpoint logs samples $5
		if [ $counter -gt 10 ]; then
			echo "Too many failed attempts." >> $results_folder/progress.txt
			break
		fi
	fi
done
mv nohup.out $results_folder

