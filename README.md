中鋼人工智慧挑戰賽-字元辨識
===

# 比賽資訊
* [Tbrain比賽網站/CSC比賽資料](https://tbrain.trendmicro.com.tw/Competitions/Details/17)
* [TB2 技術文件](./tech.md)


# 比賽結果
* Team : **TB2**
* Public LB: No.2
    * Score: 44 (Acc 約 98.53%)
* Private LB: No.8 
    * Score: 128 (Acc 約 98.71%)

# 復現
- 環境
    - 先行安裝符合運行條件的 [`PaddleOCR`](https://github.com/PaddlePaddle/PaddleOCR) 相關之套件
    - 將`PaddleOCR`源碼放入此資料夾前一層
    - 將比賽檔案放入`data`資料夾
    - 安裝必要套件，執行 `pip install -U requirements.txt`
- 假資料生成
    - 偵測任務
        - 執行 `./fake_generate/det_fake.ipynb`
        - 假資料路徑 `./fake_generate/det_fake_dataset`
    - 識別任務
        - 執行 `./fake_generate/rec_fake.ipynb`
        - 假資料路徑 `./fake_generate/rec_fake_dataset`
- 模型訓練
    - 將訓練資料標籤化後，修改 `./config/` 下的相關檔案位置
    - 在`PaddleOCR`路徑下執行 `python3 tools/train.py -c ../CSC-OCR-competition/config/det1027_config.yml`
    - 在`PaddleOCR`路徑下執行 `python3 tools/train.py -c ../CSC-OCR-competition/config/det1104_config.yml`
    - 在`PaddleOCR`路徑下執行 `python3 tools/train.py -c ../CSC-OCR-competition/config/rec1109_config.yml`
    - **注意**
        - config修改需依照不同設備設定不同的batch size
        - 訓練不需要使用外部的預訓練模型
        - 訓練時長依照GPU不同有所差異 
            - 參考值: 在 RTX 8000上 訓練偵測模型需要48hrs, 識別模型需要16hrs
- 模型預測
    - 執行 `./e2e_pred.ipynb`
