mkdir VoxCeleb1
cd VoxCeleb1

wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad
cat vox1_dev* > vox1_dev_wav.zip
unzip vox1_dev_wav.zip

wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip
unzip vox1_test_wav.zip

# download the official SUPERB train-dev-test split
wget https://raw.githubusercontent.com/s3prl/s3prl/master/s3prl/downstream/voxceleb1/veri_test_class.txt