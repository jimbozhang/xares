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

mkdir -p env

# asvspoof2015
if [ ! -f env/asvspoof2015/.audio_tar_ready ]; then
    mkdir -p env/asvspoof2015
    wget https://zenodo.org/api/records/14718430/files-archive -O env/asvspoof2015/14718430.zip
    unzip env/asvspoof2015/14718430.zip -d env/asvspoof2015/
    touch env/asvspoof2015/.audio_tar_ready
fi

# clotho
if [ ! -f env/clotho/.audio_tar_ready ]; then
    mkdir -p env/clotho
    wget https://zenodo.org/api/records/14856454/files-archive -O env/clotho/14856454.zip
    unzip env/clotho/14856454.zip -d env/clotho/
    touch env/clotho/.audio_tar_ready
fi

# cremad
if [ ! -f env/cremad/.audio_tar_ready ]; then
    mkdir -p env/cremad
    wget https://zenodo.org/api/records/14646870/files-archive -O env/cremad/14646870.zip
    unzip env/cremad/14646870.zip -d env/cremad/
    touch env/cremad/.audio_tar_ready
fi

# desed
if [ ! -f env/desed/.audio_tar_ready ]; then
    mkdir -p env/desed
    wget https://zenodo.org/api/records/14808180/files-archive -O env/desed/14808180.zip
    unzip env/desed/14808180.zip -d env/desed/
    touch env/desed/.audio_tar_ready
fi

# esc50
if [ ! -f env/esc50/.audio_tar_ready ]; then
    mkdir -p env/esc50
    wget https://zenodo.org/api/records/14614287/files-archive -O env/esc50/14614287.zip
    unzip env/esc50/14614287.zip -d env/esc50/
    touch env/esc50/.audio_tar_ready
fi

# fluentspeechcommands
if [ ! -f env/fluentspeechcommands/.audio_tar_ready ]; then
    mkdir -p env/fluentspeechcommands
    wget https://zenodo.org/api/records/14722453/files-archive -O env/fluentspeechcommands/14722453.zip
    unzip env/fluentspeechcommands/14722453.zip -d env/fluentspeechcommands/
    touch env/fluentspeechcommands/.audio_tar_ready
fi

# freemusicarchive
if [ ! -f env/freemusicarchive/.audio_tar_ready ]; then
    mkdir -p env/freemusicarchive
    wget https://zenodo.org/api/records/14725056/files-archive -O env/freemusicarchive/14725056.zip
    unzip env/freemusicarchive/14725056.zip -d env/freemusicarchive/
    touch env/freemusicarchive/.audio_tar_ready
fi

# fsd50k
if [ ! -f env/fsd50k/.audio_tar_ready ]; then
    mkdir -p env/fsd50k
    wget https://zenodo.org/api/records/14868441/files-archive -O env/fsk50k/14868441.zip
    unzip env/fsd50k/14868441.zip -d env/fsd50k/
    touch env/fsd50k/.audio_tar_ready
fi

# fsdkaggle2018
if [ ! -f env/fsdkaggle2018/.audio_tar_ready ]; then
    mkdir -p env/fsdkaggle2018
    wget https://zenodo.org/api/records/14725117/files-archive -O env/fsdkaggle2018/14725117.zip
    unzip env/fsdkaggle2018/14725117.zip -d env/fsdkaggle2018/
    touch env/fsdkaggle2018/.audio_tar_ready
fi

# gtzan_genre
if [ ! -f env/gtzan_genre/.audio_tar_ready ]; then
    mkdir -p env/gtzan_genre
    wget https://zenodo.org/api/records/14722472/files-archive -O env/gtzan_genre/14722472.zip
    unzip env/gtzan_genre/14722472.zip -d env/gtzan_genre/
    touch env/gtzan_genre/.audio_tar_ready
fi

# libricount
if [ ! -f env/libricount/.audio_tar_ready ]; then
    mkdir -p env/libricount
    wget https://zenodo.org/api/records/14722478/files-archive -O env/libricount/14722478.zip
    unzip env/libricount/14722478.zip -d env/libricount/
    touch env/libricount/.audio_tar_ready
fi

# librispeechmalefemale
if [ ! -f env/librispeechmalefemale/.audio_tar_ready ]; then
    mkdir -p env/librispeechmalefemale
    wget https://zenodo.org/api/records/14716252/files-archive -O env/librispeechmalefemale/14716252.zip
    unzip env/librispeechmalefemale/14716252.zip -d env/librispeechmalefemale/
    touch env/librispeechmalefemale/.audio_tar_ready
fi

# maestro
if [ ! -f env/maestro/.audio_tar_ready ]; then
    mkdir -p env/maestro
    wget https://zenodo.org/api/records/14858022/files-archive -O env/maestro/14858022.zip
    unzip env/maestro/14858022.zip -d env/maestro/
    touch env/maestro/.audio_tar_ready
fi

# nsynthinstument
if [ ! -f env/nsynthinstument/.audio_tar_ready ]; then
    mkdir -p env/nsynthinstument
    wget https://zenodo.org/api/records/14725174/files-archive -O env/nsynthinstument/14725174.zip
    unzip env/nsynthinstument/14725174.zip -d env/nsynthinstument/
    touch env/nsynthinstument/.audio_tar_ready
fi

# ravdess
if [ ! -f env/ravdess/.audio_tar_ready ]; then
    mkdir -p env/ravdess
    wget https://zenodo.org/api/records/14722524/files-archive -O env/ravdess/14722524.zip
    unzip env/ravdess/14722524.zip -d env/ravdess/
    touch env/ravdess/.audio_tar_ready
fi

# speechcommandsv1
if [ ! -f env/speechcommandsv1/.audio_tar_ready ]; then
    mkdir -p env/speechcommandsv1
    wget https://zenodo.org/api/records/14722647/files-archive -O env/speechcommandsv1/14722647.zip
    unzip env/speechcommandsv1/14722647.zip -d env/speechcommandsv1/
    touch env/speechcommandsv1/.audio_tar_ready
fi

# speechocean762
if [ ! -f env/speechocean762/.audio_tar_ready ]; then
    mkdir -p env/speechocean762
    wget https://zenodo.org/api/records/14725291/files-archive -O env/speechocean762/14725291.zip
    unzip env/speechocean762/14725291.zip -d env/speechocean762/
    touch env/speechocean762/.audio_tar_ready
fi

# urbansound8k
if [ ! -f env/urbansound8k/.audio_tar_ready ]; then
    mkdir -p env/urbansound8k
    wget https://zenodo.org/api/records/14722683/files-archive -O env/urbansound8k/14722683.zip
    unzip env/urbansound8k/14722683.zip -d env/urbansound8k/
    touch env/urbansound8k/.audio_tar_ready
fi

# vocalsound
if [ ! -f env/vocalsound/.audio_tar_ready ]; then
    mkdir -p env/vocalsound
    wget https://zenodo.org/api/records/14722710/files-archive -O env/vocalsound/14722710.zip
    unzip env/vocalsound/14722710.zip -d env/vocalsound/
    touch env/vocalsound/.audio_tar_ready
fi

# voxceleb1
if [ ! -f env/voxceleb1/.audio_tar_ready ]; then
    mkdir -p env/voxceleb1
    wget https://zenodo.org/api/records/14811963/files-archive -O env/voxceleb1/14811963.zip
    unzip env/voxceleb1/14811963.zip -d env/voxceleb1/
    touch env/voxceleb1/.audio_tar_ready
fi

# voxlingua33
if [ ! -f env/voxlingua33/.audio_tar_ready ]; then
    mkdir -p env/voxlingua33
    wget https://zenodo.org/api/records/14812245/files-archive -O env/voxlingua33/14812245.zip
    unzip env/voxlingua33/14812245.zip -d env/voxlingua33/
    touch env/voxlingua33/.audio_tar_ready
fi

# vocalimiations
if [ ! -f env/vocalimitations/.audio_tar_ready ]; then
    mkdir -p env/vocalimitations
    wget https://zenodo.org/api/records/14862060/files-archive -O env/vocalimitations/14862060.zip
    unzip env/vocalimitations/14862060.zip -d env/vocalimitations/
    touch env/vocalimitations/.audio_tar_ready
fi