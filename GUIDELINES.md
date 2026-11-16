# 專案使用指南

這份指南說明怎麼安裝、跑示範、接入模型和交付結果。資料欄位、模型 API 與驗收條件請看 [Interface Contract](INTERFACE_CONTRACT.md)。

## 1. 跑一次示範

以下命令從本 Git repo 根目錄執行，也就是有 `run.py` 的 `project/`：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/make_demo_data.py
python run.py --config configs/demo.json --check-interface
python run.py --config configs/demo.json
```

若已有虛擬環境，直接啟用即可。Demo 使用四張合成示範圖和亮度分數，只供檢查路徑、讀圖及結果輸出。驗收模式會印出檢查結果。正式執行模式會在 `results/demo_performance/` 產生 `predictions.csv`、`summary.json` 與 `run_info.json`。同一輸出位置再次執行會覆寫結果。

## 2. 第一階段怎麼開工

| 成員 | 可先從這裡開始 |
| --- | --- |
| HSIANG | 在 `models/univfd.py` 接入 UnivFD，記錄權重、官方前處理及 ViT 候選層。 |
| WEI | 在 `models/syntheticsight.py` 接入 SyntheticSight，與 YUZI 核對 140k 的訓練及驗證重疊。 |
| YUZI | 依 `data/manifest_template.csv` 整理資料。先交一份包含真、假圖的小清單，供三模型驗收。 |
| DA | 在 `models/cnndetection.py` 接入 CNNDetection，維護共用入口並負責三模型整合驗收。 |

模型接入時，照 [Interface Contract](INTERFACE_CONTRACT.md#2-模型-api) 完成 adapter，然後在 `models/registry.py` 登記 factory。正式模型使用各自需要的 PyTorch 與依賴，照模型官方設定準備環境，並記錄實際測試的版本與裝置。權重可以放在本機 `weights/`，大型資料另行管理。

複製 `configs/formal.example.json` 為本機設定檔，改成實際模型、權重、小清單、切分、裝置和輸出資料夾。每個模型各用一份設定和不同的輸出資料夾。驗收時使用相同的小清單及切分。

```bash
cp configs/formal.example.json configs/cnndetection.local.json
# 編輯本機設定檔後執行。
python run.py --config configs/cnndetection.local.json --check-interface
python run.py --config configs/cnndetection.local.json
```

先用第一條 Python 命令確認一般分數及梯度，再用第二條確認逐圖 CSV 可寫出。範本所指的正式 adapter、權重及清單要先由負責人接入。把模型、權重版本、裝置和驗收結果交給 DA。

## 3. Notebook 和第二階段工作區

使用 notebook 時，從同一個虛擬環境執行：

```bash
python -m pip install -r requirements-notebook.txt
python -m notebook notebooks
```

先開 `notebooks/00_example.ipynb`。可在各實驗的 notebook 資料夾探索，將能重跑的程式整理到 `experiments/`。

| 工作區 | 如何交接 |
| --- | --- |
| `experiments/performance/` | DA 從 YUZI 的同一份正式清單跑三模型，交出各模型的 `predictions.csv`。YUZI 負責效能指標、信賴區間及 FP/FN 清單。 |
| `experiments/occlusion/` | HSIANG 維護共用 ROI／Mask 工具並交給 DA，也負責原圖與修改圖的分數比較。 |
| `experiments/features/` | DA 使用共用 ROI 逐圖計算特徵並完成 TCAV。YUZI 用特徵表與模型分數做組間比較和相關分析。 |
| `experiments/xai/` | WEI 使用 DA 的分數和 YUZI 提供的誤判清單，挑案例並產出熱圖。 |

正式跑三模型時，先確認每份設定都指向同一個 test 清單。分享結果時保留程式產出的 CSV、執行紀錄和必要設定。三個正式模型接入前可用 demo 驗證流程，XAI、遮擋及特徵／TCAV 的正式實驗入口會在實作前明確提示。
