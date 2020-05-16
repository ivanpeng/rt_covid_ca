#!/bin/bash
# A simple script that will run the full gaussian model, and push to the front-end repo
# Triggered by a cron every day at 9 am PST
# Activate conda env, run gaussian, cp model to front-end folder, git push
# exit on error so we don't proceed
set -e

cd /root/rt_covid_ca_backend
rm -f data/linelist.csv
wget -O data/linelist.csv https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/cases.csv
source /root/anaconda3/etc/profile.d/conda.sh
conda activate rt_covid
cd src
PYTHONPATH=/root/rt_covid_ca_backend python rt_gaussian.py

d=$(date +"%Y-%m-%d")
filename="rt-ca-${d}.json"
# Copy, git push
#echo "cp /root/rt_covid_ca_backend/export/data_by_day/$filename /root/rt-canada/src/caseData/latest.json"
cp /root/rt_covid_ca_backend/export/data_by_day/$filename /root/rt-canada/src/caseData/latest.json
cd /root/rt-canada
git checkout master
git pull
git status
git add .
git commit -m "Update data $d"
git push origin master
echo "Complete"
