專案分為 classification task 和 regression task，由於先前儲存空間不足，只取原資料約 1% 訓練
原資料集位於 /home/bdm0065/snap/0065/physionet.org/files/mimic-iv-ecg/1.0/files/  

To run :
pip install -r requirements.txt

python3 train.py
python3 eval.py

python3 train_regression.py
python3 eval_regression.py --mode multi / any / mild / severe  (for different baselines)