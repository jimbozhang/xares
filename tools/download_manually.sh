#!/bin/bash -e

# Normally, you DO NOT need to use this script to download data, as it could be automatically downloaded by `xares.run`.
# However, if automatic downloading keep fails, you can run this script to manually download the data.
# You may select the ones you want to download by commenting out the others.

if ! command -v wget &> /dev/null; then
    echo "wget could not be found, please install it to proceed."
    exit 1
fi

if ! command -v unzip &> /dev/null; then
    echo "unzip could not be found, please install it to proceed."
    exit 1
fi

mkdir env

# asvspoof2015
mkdir env/asvspoof2015
wget https://zenodo.org/api/records/14718430/files-archive -O env/asvspoof2015/14718430.zip
unzip env/asvspoof2015/14718430.zip -d env/asvspoof2015/
touch env/asvspoof2015/.audio_tar_ready

# cremad
mkdir env/cremad
wget https://zenodo.org/api/records/14646870/files-archive -O env/cremad/14646870.zip
unzip env/cremad/14646870.zip -d env/cremad/
touch env/cremad/.audio_tar_ready

# esc50
wget https://zenodo.org/api/records/14614287/files-archive -O env/esc50/14614287.zip
unzip env/esc50/14614287.zip -d env/esc50/
touch env/esc50/.audio_tar_ready

# fluentspeechcommands
wget https://zenodo.org/api/records/14722453/files-archive -O env/fluentspeechcommands/14722453.zip
unzip env/fluentspeechcommands/14722453.zip -d env/fluentspeechcommands/
touch env/fluentspeechcommands/.audio_tar_ready

# freemusicarchive
wget https://zenodo.org/api/records/14725056/files-archive -O env/freemusicarchive/14725056.zip
unzip env/freemusicarchive/14725056.zip -d env/freemusicarchive/
touch env/freemusicarchive/.audio_tar_ready

# fsdkaggle2018
wget https://zenodo.org/api/records/14725117/files-archive -O env/fsdkaggle2018/14725117.zip
unzip env/fsdkaggle2018/14725117.zip -d env/fsdkaggle2018/
touch env/fsdkaggle2018/.audio_tar_ready

# gtzan_genre
wget https://zenodo.org/api/records/14722472/files-archive -O env/gtzan_genre/14722472.zip
unzip env/gtzan_genre/14722472.zip -d env/gtzan_genre/
touch env/gtzan_genre/.audio_tar_ready

# libricount
wget https://zenodo.org/api/records/14722478/files-archive -O env/libricount/14722478.zip
unzip env/libricount/14722478.zip -d env/libricount/
touch env/libricount/.audio_tar_ready

# librispeechmalefemale
wget https://zenodo.org/api/records/14716252/files-archive -O env/librispeechmalefemale/14716252.zip
unzip env/librispeechmalefemale/14716252.zip -d env/librispeechmalefemale/
touch env/librispeechmalefemale/.audio_tar_ready

# nsynthinstument
wget https://zenodo.org/api/records/14725174/files-archive -O env/nsynthinstument/14725174.zip
unzip env/nsynthinstument/14725174.zip -d env/nsynthinstument/
touch env/nsynthinstument/.audio_tar_ready

# ravdess
wget https://zenodo.org/api/records/14722524/files-archive -O env/ravdess/14722524.zip
unzip env/ravdess/14722524.zip -d env/ravdess/
touch env/ravdess/.audio_tar_ready

# speechcommandsv1
wget https://zenodo.org/api/records/14722647/files-archive -O env/speechcommandsv1/14722647.zip
unzip env/speechcommandsv1/14722647.zip -d env/speechcommandsv1/
touch env/speechcommandsv1/.audio_tar_ready

# speechocean762
wget https://zenodo.org/api/records/14725291/files-archive -O env/speechocean762/14725291.zip
unzip env/speechocean762/14725291.zip -d env/speechocean762/
touch env/speechocean762/.audio_tar_ready

# urbansound8k
wget https://zenodo.org/api/records/14722683/files-archive -O env/urbansound8k/14722683.zip
unzip env/urbansound8k/14722683.zip -d env/urbansound8k/
touch env/urbansound8k/.audio_tar_ready

# vocalsound
wget https://zenodo.org/api/records/14722710/files-archive -O env/vocalsound/14722710.zip
unzip env/vocalsound/14722710.zip -d env/vocalsound/
touch env/vocalsound/.audio_tar_ready

# voxceleb1
wget https://zenodo.org/api/records/14725363/files-archive -O env/voxceleb1/14725363.zip
unzip env/voxceleb1/14725363.zip -d env/voxceleb1/
touch env/voxceleb1/.audio_tar_ready

# voxlingua33
wget https://zenodo.org/api/records/14723799/files-archive -O env/voxlingua33/14723799.zip
unzip env/voxlingua33/14723799.zip -d env/voxlingua33/
touch env/voxlingua33/.audio_tar_ready
