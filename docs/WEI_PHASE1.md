# WEI 第一階段交接：SyntheticSight

## 範圍與目前狀態

依 2026-11-17 Team Plan 第 3 頁，第一階段交付 adapter、小樣本驗收與 140k 官方切分資訊。本次不實作第二階段 XAI。

- 已完成：官方 final ResNet-50 checkpoint 接入、registry、設定範例、可微分輸出、可 hook 中間層、權重下載驗證工具、重疊檢查工具。
- 已驗證：官方推論數值一致、CPU 分數與梯度驗收、逐圖 CSV 與執行紀錄輸出。
- 尚待資料：YUZI 的真／假人臉共用小清單、140k train/validation 影像與對照清單。現有驗收使用人工像素測試圖，不能代表真實人臉效能，也不代表 DA 已完成團隊共同驗收。
- 140k 重疊狀態：**未完成逐張核對**；已查明官方訓練／驗證範圍並備妥工具。無參考資料時，工具明確輸出 `unverified_no_reference`。

## 官方來源與模型設定

上游：[kristinealli/SyntheticSight](https://github.com/kristinealli/SyntheticSight)，固定版本 `8d4c81c9f5cac65d66f2ddebd02f2111f0f1c3c7`。

- [模型結構](https://github.com/kristinealli/SyntheticSight/blob/8d4c81c9f5cac65d66f2ddebd02f2111f0f1c3c7/src/synthetic_sight/model.py)：ResNet-50，分類頭 `2048 → 256 → BatchNorm → ReLU → Dropout(0.3) → 1`。載入完整權重，不另外下載 ImageNet backbone。
- [官方推論](https://github.com/kristinealli/SyntheticSight/blob/8d4c81c9f5cac65d66f2ddebd02f2111f0f1c3c7/src/synthetic_sight/inference.py)：RGB、PIL bilinear resize 至 224×224、除以 255、ImageNet mean/std；無裁切或隨機增強。EXIF 已由共用讀圖函式處理。
- [模型中繼資料](https://github.com/kristinealli/SyntheticSight/blob/8d4c81c9f5cac65d66f2ddebd02f2111f0f1c3c7/models/model_metadata.json)：Real=0、Fake=1；單一 binary fake logit，分數為 sigmoid(logit)，分數越大越偏假。官方 validation Fake F1 選定 epoch 13、門檻 0.51，直接沿用，未用本專案測試資料調參。
- 權重：`models/best_resnet50.pth`，220,413,587 bytes，SHA256 `d9a7fd6a692c942b550f9848500dc3ffb10d5809cb0d0091990648bf369ad21c`。adapter 強制核對此雜湊，拒絕 legacy 或未查核替代權重。
- 候選層：`layer3`、`layer4`（ResNet spatial activation，N×C×H×W）；同一已載入模型上的真實 module，供後續方法選層。模型為 `eval()`；只有一般分數推論關閉梯度。

## 安裝與執行

以下從專案根目錄執行，Windows 啟用 venv 使用 `.venv\Scripts\Activate.ps1`。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-syntheticsight.txt
python -m tools.download_syntheticsight
```

權重下載工具使用固定上游版本與 SHA256；不覆寫已存在但雜湊不同的檔案。權重由本機 `weights/` 管理，不加入 Git。

先跑可重現的人工介面測試圖：

```powershell
python examples/make_demo_data.py
python run.py --config configs/syntheticsight.interface.json --check-interface
python run.py --config configs/syntheticsight.interface.json
python -m unittest discover -s tests -v
```

`syntheticsight.interface.json` 指向 demo 像素圖，其真／假標籤只供分支與梯度測試，輸出摘要的 accuracy **沒有研究意義**。

拿到 YUZI 清單後，複製 `configs/syntheticsight.example.json` 為 `configs/syntheticsight.local.json`，設定 `manifest`、`split`、`device` 與獨立 `output_dir`：

```powershell
python run.py --config configs/syntheticsight.local.json --check-interface
python run.py --config configs/syntheticsight.local.json
```

交給 DA：設定、清單版本、介面驗收輸出、`predictions.csv`、`summary.json`、`run_info.json`。正式比較前固定門檻與清單。CUDA/MPS 尚未在本機驗證；需依實際裝置安裝對應 PyTorch 版本並重跑驗收。

## 140k 官方切分與交給 YUZI 的規則

依[官方 data protocol](https://github.com/kristinealli/SyntheticSight/blob/8d4c81c9f5cac65d66f2ddebd02f2111f0f1c3c7/docs/data.md)，來源為 Kaggle `xhlulu/140k-real-and-fake-faces`：

| 官方 split | Real (FFHQ) | Fake (StyleGAN) | 用途 |
| --- | ---: | ---: | --- |
| train | 50,000 | 50,000 | 已用於模型訓練，排除本專案獨立評估 |
| validation | 10,000 | 10,000 | 已用於模型與門檻選擇，排除本專案獨立評估 |
| test | 10,000 | 10,000 | 官方保留測試；可作候選來源，仍須記錄來源與重疊檢查 |

seed=13 是官方訓練設定，不能單靠 seed 重建逐張影像身分。上游文件提及 `resnet50_sample_manifest.csv`，但本次檢視的公開版本未提供此檔及原始資料。官方稱路徑無交集，也明確指出不能據此排除近重複或相同人臉身分。勿把檔案改名、重新切 split 或路徑不同當成獨立資料的證據。

YUZI 應保留官方原始 split／相對路徑對照，把 train 與 validation 整理成現有共用六欄 manifest（`split` 使用 train、val 或 validation；假圖 generator=stylegan）。使用實際影像核對：

```powershell
python -m tools.audit_140k_overlap --manifest data/pilot_manifest.csv --reference data/140k_train_manifest.csv --reference data/140k_val_manifest.csv --output-dir results/140k_overlap
```

工具以檔案 SHA256 與 EXIF 校正後 RGB 像素 SHA256 比對，保留每筆候選的 sample_id、狀態與相符參考紀錄：

- `known_overlap`：已知重疊；`exclude_known_overlap=True`，由 YUZI 排除後重新定版清單。工具不自動修改來源清單。
- `no_exact_match_in_supplied_references`：僅在提供的參考範圍未找到完全相符；不代表獨立性已核實。
- `unverified_no_reference`：沒有提供參考影像，保留未核實狀態。

輸出 `overlap_audit.csv` 和 `overlap_summary.json`。參考清單混入 test/pilot 會拒絕執行；缺圖也會明確失敗，避免不完整資料被當成零重疊。雜湊不會偵測有損重壓縮、近重複、同身分不同照片；這些仍須另外查核或列為限制。

## 實際驗證紀錄

執行日期：2026-11-27（Asia/Taipei）。專案基準 `5b457c56a531f7679da3de91e46cadd0e8a608b0`，分支 `wei/phase1-syntheticsight`。

本機 Python 3.13.14、PyTorch 2.15.0+cpu、TorchVision 0.29.2+cpu，Windows CPU；其餘版本見 `validation/syntheticsight_environment.json`。權重內記載原訓練環境 Python 3.12.13、PyTorch 2.11.0+cu128、TorchVision 0.26.0+cu128；與本機版本分開記錄。

- 官方數值比對：固定 seed=13，四種尺寸的 RGB 像素測試圖；前處理逐值一致，fake logit 與分數誤差皆為 0。完整結果見 `validation/syntheticsight_upstream_parity.json`。
- 共用介面：通過單張／批次分數、輸入梯度、`layer3` 與 `layer4` activation 梯度檢查。
- 正常 CLI：四張人工測試圖成功輸出標準三個檔案；不做真實人臉正確率結論。
- 重疊工具：測試更名與 PNG→BMP 無損轉存仍可找出相同像素，未相符及無參考時保留不同狀態；拒絕把 test 當成 train/val 證據。
- Registry 採延後 import，demo 路徑維持不載入 torch。

重跑官方推論比對需先取得固定上游版本（僅讀取來源）：

```powershell
git clone https://github.com/kristinealli/SyntheticSight.git ../SyntheticSight-upstream
git -C ../SyntheticSight-upstream checkout 8d4c81c9f5cac65d66f2ddebd02f2111f0f1c3c7
python -m tools.verify_syntheticsight_upstream --upstream-root ../SyntheticSight-upstream --output results/syntheticsight_parity.json
```

## 尚未完成的共同驗收

1. YUZI 提供有真、假人臉的同一份共用小清單，以正式 adapter 重跑介面驗收及逐圖輸出。
2. 取得官方 train/validation 影像或完整身分對照，產出實際重疊清單；未核實部分保留標記。
3. DA 在團隊目標裝置／環境確認整合結果。尚未執行三模型比較、600 張 pilot、GPU 效能或第二階段 XAI。
