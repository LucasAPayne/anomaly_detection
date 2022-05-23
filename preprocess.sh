#!/bin/bash
# TODO: Retrieve AIT dataset from the Internet

script_path="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )" # Magical incantation to get the absolute path to this script
slogert_path="${script_path}/vendor/slogert"
out_path="${script_path}/data/AIT"

read -p "Would you like to run the AIT demo [y/n]? " -n 1 -r $DEMO
echo

# Check if response from demo query matches "n" or "N" and nothing else
if [[ $DEMO =~ ^[nN]$ ]]
then
    # Currently, the demo is the only option. Exit
    [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1
fi

# Clean up any leftover files from a previous aborted run
rm -rf ${slogert_path}/output/*
rm -rf ${slogert_path}/input/*
rm -rf ${out_path}/*
mkdir -p $out_path/{raw,train,test,valid}

read -p "Enter the path to the directory where the dataset is stored: " -r dataset_path

# Copy the dataset to the anomaly_detection directory
echo "Copying dataset to anomaly_detection directory..."
cp -r $dataset_path anomaly_detection
echo "Dataset moved"

# Extract 1-day dataset from AIT (dataset is placed in output dir)
cd anomaly_detection
python extract_dataset.py

# Move training dataset to SLOGERT, run SLOGERT commands, then move the raw dataset (raw dir in out_path)
cp -r output/train ${slogert_path}/input
cd $slogert_path
python slogert.py gen-kg -a -o ${out_path}/train.ttl
cp -r input/train ${out_path}/raw
rm -rf input/*

# Move testing dataset to SLOGERT, run SLOGERT commands, apply labels
cd ${script_path}/anomaly_detection
cp -r output/test ${slogert_path}/input
cd $slogert_path
python slogert.py gen-kg -a -o ${out_path}/test.ttl
python ${script_path}/anomaly_detection/label.py -i ${out_path}/test.ttl

# Move validation dataset to SLOGERT, run SLOGERT commands, apply labels
cd ${script_path}/anomaly_detection
cp -r output/valid ${slogert_path}/input
rm -rf output
rm -rf ${dataset_path##*/} # Remove copied dataset (this gets the final subdir of the path given to the dataset)
cd $slogert_path
python slogert.py gen-kg -a -o ${out_path}/valid.ttl
python ${script_path}/anomaly_detection/label.py -i ${out_path}/valid.ttl

# Generate IDs for entities and relations, reconstruct KG using them, move the raw dataset
python slogert.py post-process -i ${out_path}/train.ttl -g
python slogert.py post-process -i ${out_path}/test.ttl -g -l
python slogert.py post-process -i ${out_path}/valid.ttl -g -l
cp -r input/test ${out_path}/raw
cp -r input/valid ${out_path}/raw
rm -rf input/*

# Move training and testing data to their own folders, which is how cyberML expects them
cd $out_path
mv train.ttl train.txt entity_ids.txt relation_ids.txt train/
mv test.ttl test.txt test_labels.txt test/
mv valid.ttl valid.txt valid_labels.txt valid/
