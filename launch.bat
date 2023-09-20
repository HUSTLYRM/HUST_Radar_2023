CALL D:\Environment\Anaconda\Scripts\activate.bat D:\Environment\Anaconda
CALL conda activate radar_merge 
:: 切换py脚本所在盘符。当bat文件所在位置与py脚本所在位置一样时，则不需要切换盘符。
D: 
cd D:\WorkSpace\Projects\PythonProjects\Radar2023_merge
:: python main.py > log.txt
python main.py 
