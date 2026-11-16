# 專案實作準則

這個 `project/` 是大家共用的實驗骨架。**目前只有 `demo` 假模型和效能評估可以跑通；它們只用來檢查流程，不能當研究結果。** CNNDetection、UnivFD、SyntheticSight 的權重與推論程式還沒有接入。

## 先跑一次示範

在專案最外層執行：

```bash
python3 -m pip install -r project/requirements.txt
python3 project/examples/make_demo_data.py
python3 project/run.py --config project/configs/demo.json
```

結果會在 `project/results/demo_performance/`：

- `predictions.csv`：每張圖的分數與預測，方便找錯誤案例。
- `summary.json`：整體與各來源的簡單指標。某來源沒有真圖或假圖時，相關指標會是 `null`。

重新執行同一設定會覆寫該輸出資料夾裡的這兩個檔案。正式實驗請替每組模型與實驗指定不同的 `output_dir`。

## 各人改哪裡

| 範圍 | 負責內容 |
| --- | --- |
| `data/` | 共用樣本清單、讀圖；所有人使用同一批測試影像與 `sample_id`。 |
| `models/` | 各模型的載入、專用前處理、分數方向與門檻。接模型的人新增檔案並登記到 `registry.py`。 |
| `experiments/performance/` | A：所有模型的效能與跨來源比較；已有最小範例。 |
| `experiments/xai/` | B：所有模型的歸因方法與視覺化；需要記錄各模型實際用的方法。 |
| `experiments/occlusion/` | C：所有模型的遮擋與局部擾動；同樣的擾動設定要套用到同一批影像。 |
| `experiments/features/` | D：臉部特徵統計、概念測試與控制組。 |
| `configs/` | 一次執行的模型、資料、實驗、輸出位置設定。 |
| `results/` | 程式輸出；請勿手改數值。 |
| `notebooks/` | 四個實驗各自的探索空間；範本示範如何使用共用資料與模型。 |

**分工以實驗為主軸。** 例如 C 完成遮擋實驗後，由 C 用共用模型介面跑每個模型並比較結果。模型接入是前期共同工作，可以分人接不同模型；不代表往後每人只分析自己的模型。

## 如何使用 notebook

從專案最外層執行以下指令，或在 VS Code 安裝 Jupyter 擴充套件並選擇同一個 Python 環境：

```bash
python3 -m pip install -r requirements-notebook.txt
python3 project/examples/make_demo_data.py
python3 -m notebook project/notebooks
```

開啟 `notebooks/00_example.ipynb`，依序執行每個儲存格。它會載入示範清單、顯示一張假資料，並呼叫共用的 `demo` 模型。這些資料和分數只供測試；建立示範資料時會覆寫先前產生的同名示範檔案。

每個人可以把範例複製到自己的資料夾：A 用 `notebooks/performance/`、B 用 `notebooks/xai/`、C 用 `notebooks/occlusion/`、D 用 `notebooks/features/`。在 notebook 裡先探索影像、測試想法及畫圖；確定後，把正式分析邏輯整理進對應的 `experiments/`，並在 `run.py` 的共同入口執行，讓組員能重跑。呼叫共用的 `data/`、`models/`，以維持相同樣本與模型前處理。

分享 notebook 前，建議清除輸出並從第一格重新執行一次，確認沒有依賴先前隱藏的執行狀態；記下使用的模型、樣本清單、實驗參數與套件版本。大型資料、權重及產生的結果請另行管理。

## 共用約定

1. **資料清單**：CSV 必須有 `sample_id,path,label,source,split`。`label=0` 是真圖，`label=1` 是生成圖；`source` 填資料或生成器來源。影像的相對路徑從 CSV 所在資料夾起算。可複製 `data/manifest_template.csv` 開始填。
2. **模型輸入**：`score_fake(image)` 收到的是 `H×W×3` 的 RGB `uint8` 陣列。各模型需要的尺寸、正規化、裝置及批次處理，放在自己的模型檔案裡。
3. **模型輸出**：回傳 `0～1` 的分數，**越高越偏向生成圖**。這個分數不一定是校準過的機率。若原模型輸出方向相反，adapter 要先轉換。`threshold` 寫在模型 adapter；若需自行選門檻，只能用獨立驗證集，不能看測試集調整。
4. **公平比較**：先固定測試清單、切分和來源標籤，再讓所有模型跑同一批影像。不要把模型訓練過的影像或近乎重複的影像放進獨立測試集。
5. **結果可追溯**：每筆結果保留 `sample_id`、模型、來源、原始分數及實驗設定。遮擋與 XAI 額外記錄方法、參數和必要的對照條件。
6. **缺功能要明說**：目前 B、C、D 的 `run.py` 會明確回報「尚待實作」。接手者要在自己的目錄完成它，不要輸出看似正式的空白結果。

## 接入正式模型的最短步驟

1. 在 `models/` 新增一個類別，繼承 `FaceDetector`，設定 `name`、`threshold`，實作 `score_fake`。
2. 在 `models/registry.py` 登記名稱與建立模型的函式。權重路徑可以從專用設定檔或環境變數讀取，不要把權重放進版本控制。
3. 用少量已知樣本確認：RGB 順序正確、真／假分數方向正確、輸出在 `0～1`，並確認門檻來源。
4. 複製 `configs/demo.json`，改成正式模型名稱、正式 manifest 與獨立輸出資料夾，再執行 `run.py`。

先接通一個正式模型與 A 的流程，再由 B、C、D 沿著同一份清單和模型介面擴充；其餘模型接入後，不必重寫各實驗的資料讀取與執行入口。

