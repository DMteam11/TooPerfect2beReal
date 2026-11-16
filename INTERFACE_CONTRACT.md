# Interface Contract｜第一階段共用規格

這份文件定義資料、模型與結果如何接在一起。安裝、執行與交付步驟請看 [GUIDELINES.md](GUIDELINES.md)。

第一階段由 HSIANG 接 UnivFD、WEI 接 SyntheticSight、DA 接 CNNDetection。YUZI 提供共用測試清單，DA 負責模型介面與整合驗收。各負責人在小樣本通過驗收後，再照分工計畫進入正式實驗。

## 1. 資料清單與讀圖

正式清單使用 UTF-8 CSV，表頭為 `sample_id,path,label,source,split,generator`。可從 [manifest_template.csv](data/manifest_template.csv) 開始。

| 欄位 | 規則 |
| --- | --- |
| `sample_id` | 同一份清單中唯一的字串。固定後供三模型的預測、特徵表、遮擋及 XAI 對照。 |
| `path` | 影像路徑。相對路徑以 CSV 所在資料夾為準。 |
| `label` | 整數 `0` 為真人，`1` 為生成圖。 |
| `source` | 資料集或子來源，例如 `140k`、`dff_wiki`。 |
| `split` | 資料切分。小樣本可用 `pilot`，門檻驗證用 `val`，正式測試用 `test`。 |
| `generator` | 生成方法，例如 `stylegan` 或 `sd_text2img`。假圖必填，真圖留空。 |

`load_manifest(path, split=None, *, require_generator=False)` 回傳 `list[Sample]`。正式 CLI 會開啟 `require_generator=True`。舊的五欄 demo 清單仍可直接讀取。正式比較使用同一份測試清單和同一個切分。YUZI 與 WEI 先處理已知訓練／驗證重疊，並記錄無法核實的樣本。

`load_rgb_image(path)` 統一處理 EXIF 方向，回傳 `numpy.ndarray`，排列為 `H × W × 3`，型別為 `uint8`，通道順序為 RGB，值域為 0–255。模型負責人把官方尺寸、裁切、插值和正規化放在自己的 adapter，讓效能與 XAI 共用同一套前處理。

## 2. 模型 API

正式模型繼承 [ExplainableFaceDetector](models/base.py)。`demo` 只實作基本分數介面。

| 成員 | 登記名稱 | 建議開發檔案 |
| --- | --- | --- |
| HSIANG | `univfd` | `models/univfd.py` |
| WEI | `syntheticsight` | `models/syntheticsight.py` |
| DA | `cnndetection` | `models/cnndetection.py` |

| API | 輸入與輸出 |
| --- | --- |
| `preprocess_tensor(image) -> Tensor` | 輸入一張共用 RGB 影像。輸出模型所在裝置上的有限浮點張量，形狀為 `(1, 3, H, W)`。 |
| `forward_logits(tensor) -> Tensor` | 輸入同裝置的 `(N, 3, H, W)` 模型張量。輸出同裝置、有限且可微分的 `(N,)` fake logits，每張圖一個，越大越偏向生成圖。 |
| `score_fake(image) -> float` | 回傳 `0..1` 的生成分數，越高越偏向生成圖。基底類別把 fake logit 經 sigmoid 轉成分數。分數可用於排序與門檻分類，校準需另行驗證。 |
| `predict_label(image) -> int` | 呼叫 `score_fake` 並套用門檻。`score >= threshold` 回傳 `1`，其餘回傳 `0`。已有分數時用 `label_from_score(score)`。 |
| `get_target_layers() -> Mapping[str, nn.Module]` | 回傳正在使用的模型上的候選層，名稱固定且可註冊 forward hook。至少一層可取得張量 activation 和對 fake logit 的梯度。 |

模型載入後使用 `eval()`。若官方輸出兩個 logits，以 `fake_logit - real_logit` 統一方向。若二元頭以真人為正類，將其 logit 反號。可微分的 forward 保留 PyTorch 的計算圖。用 `torch.no_grad()` 的地方是一般分數推論，IG、Grad-CAM、TCAV 直接使用可微分的路徑。輸入梯度從前處理完成的張量開始。UnivFD 的 adapter 要提供 ViT 層名及 Token 形狀，ViT Grad-CAM 選層與 Token 重排屬於 WEI 在第二階段的實驗。

每個 adapter 要提供 `name`、`threshold`、`threshold_source`、`weights_id` 和 `preprocessing`。門檻位於 0–1，使用官方門檻或獨立驗證集決定，並在正式測試前固定。`threshold_source` 記錄門檻來源，`weights_id` 記錄權重版本或雜湊，`preprocessing` 記錄官方尺寸、裁切及正規化等設定。

[registry.py](models/registry.py) 中的 factory 接收關鍵字參數 `weights_path` 和 `device`，回傳 adapter。正式模型需要可讀取的權重檔，`model.name` 必須與登記名稱一致。裝置由呼叫端選擇，例如 `cpu`、`cuda:0` 或 `mps`。

## 3. 執行設定與結果

| 欄位 | 規則 |
| --- | --- |
| `experiment` | `performance`、`xai`、`occlusion` 或 `features`。目前可執行的實驗是 `performance` 的逐圖推論與基本摘要。 |
| `model` | Registry 名稱。每次執行一個模型。 |
| `manifest`, `output_dir` | 輸入清單與輸出資料夾。相對路徑從專案根目錄起算。 |
| `split` | 指定要跑的切分，省略或為 `null` 時讀全表。正式評估應明確指定 `test`。 |
| `weights`, `device` | 權重路徑相對於專案根目錄，正式模型必填。`device` 省略時使用 `cpu`。 |

`predictions.csv` 由 `performance` 產生。欄位是 `sample_id,path,label,source,split,generator,model,score_fake,threshold,predicted_label`，每個模型每張圖一列。跨模型彙整時用 `(model, sample_id)` 辨識一筆預測。DA 交出三個模型的逐圖分數後，YUZI 可按 `sample_id` 接上臉部特徵表，WEI 可以同一編號對照誤判案例。

`summary.json` 目前提供整體及來源別的樣本數、Accuracy、真人正確率和假圖召回率。缺少必要類別的指標用 `null`。正式指標及信賴區間由 YUZI 根據逐圖分數計算，分組 AUC 需要同時含真圖與假圖。`run_info.json` 記錄清單、切分、模型、裝置和門檻，正式模型另記權重、前處理、門檻來源與 PyTorch 版本。

第二階段的特徵表以 `sample_id` 對接，一張圖只算一次，並用 `roi_failed` 保留臉部定位失敗。HSIANG 的 ROI／Mask 工具交給 DA 後，DA 負責特徵計算，YUZI 使用表格做統計。特徵欄位與統計程式由第二階段定義。

## 4. 第一階段驗收

1. YUZI 提供包含真、假圖的同一份小清單。模型負責人逐一核對官方權重、前處理、真假分數方向與門檻來源。
2. DA 用共用入口的 `--check-interface` 驗證真、假圖都能讀取和評分，分數與 fake logit 一致，單張與批次介面可用，且能取得輸入及至少一個中間層的梯度。再用正常執行模式確認三個模型能各自寫出相同欄位的逐圖結果。
3. 各模型負責人記錄項目、權重版本、裝置、套件版本、候選層和測試結果，交 DA 整合。驗收用的幾張圖只用來確認介面，效能比較使用定版的獨立測試集。

目前 workspace 已提供共用介面、demo 與驗收入口。三個正式 adapter、權重及正式資料仍由分工負責人接入。
