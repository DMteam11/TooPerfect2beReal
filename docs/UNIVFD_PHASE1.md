# HSIANG 第一階段驗收記錄：UnivFD

**分支**：`hsiang/univfd`（基於 `main` `5b457c5`）
**日期**：2026-11-27
**依據**：`INTERFACE_CONTRACT.md` 第 2、4 節、`GUIDELINES.md`；分工表 HSIANG 第 2 頁（第一階段）

## 1. 權重與前處理

| 項目 | 內容 |
|---|---|
| 分類頭 | `fc_weights.pth`，取自官方 UnivFD repo commit `030495aea3300a8b54c0ec37ec7fe1dd7e63c619` 的 `pretrained_weights/`；sha256 `477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847` |
| Backbone | CLIP ViT-L/14；sha256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`（與官方下載網址的雜湊一致） |
| `weights_id` | `univfd-head-477100745713-clipL14-b8cca3fd41ae` |
| 前處理（`preprocessing`） | 官方 `validate.py`：CenterCrop(224)，不 resize；ToTensor；CLIP mean/std 正規化；fp32 |
| 分數方向 | 單一 fake logit，`sigmoid > 0.5` 判為生成圖，與官方一致 |
| 門檻（`threshold`） | 0.5，來源為官方 `validate.py`，未以測試資料調整 |

**註記**：分工表第 2 頁寫「保留官方 Resize／Normalize」。官方 UnivFD 的 `validate.py` 實際做的是 CenterCrop、不 resize，本實作依此採用。若團隊之後決定改用 CLIP 原版的 Resize + CenterCrop，需要同步修改 adapter 與本記錄。

## 2. 套件與裝置

- Python 3.12.3（`.venv`）
- torch 2.11.0+cu128、torchvision 0.26.0+cu128、numpy 2.5.4、Pillow 12.3.0、scikit-learn 1.9.1、scipy 1.18.1、ftfy 6.3.1、regex 2026.11.25、tqdm 4.70.1、CLIP（openai/CLIP `d05afc4`）
- 完整版本見 `requirements-univfd.txt`
- 測試裝置：CPU；CUDA：NVIDIA GeForce RTX 5060 Laptop GPU（8151 MiB），驅動 610.88
- **未測：MPS**（本機沒有 Apple 裝置）

## 3. 候選層與 ViT Token 形狀（交付 WEI）

`get_target_layers()` 回傳 25 層：`visual.transformer.resblocks.0` 至 `.23`，以及 `visual.ln_post`。

- 每個 resblock 的 activation 與 gradient 形狀為 `(257, N, 1024)`，sequence-first；index 0 為 CLS，其餘 256 個為 16×16 patch。
- `visual.ln_post` 形狀為 `(N, 1024)`，只有 pooled CLS。
- Token 重排（已檢查 round-trip）：

```python
tokens = act.permute(1, 0, 2)                      # (N, 257, 1024)，index 0 為 CLS
patches = tokens[:, 1:, :].reshape(N, 16, 16, 1024)  # patch 網格，順序與 conv1 展開一致
```

**梯度測例**（2 張圖，CPU；CUDA 結果相同）：

```python
grad = torch.autograd.grad(logits.sum(), act)[0]
patch_nonzero = (grad.permute(1, 0, 2)[:, 1:, :].abs().sum(-1) > 0).float().mean(-1)
```

- `resblocks.0` 至 `.22`：patch token 梯度非零，`patch_nonzero = 1.0`
- `resblocks.23`：`patch_nonzero = 0.0`。因為 `ln_post` 只取 CLS，最後一層的 patch token 對 logit 沒有梯度；做 patch 層級分析時應避開此層。

## 4. 測試結果

### 4.1 真假分數方向與官方一致（分工表：確認真／假分數方向）

與官方 UnivFD（同一組權重，`validate.py` 前處理）比對 8 張圖：4 張 256×256 人工圖，4 張人工雜訊圖（224、300×400、512、1024）。

| 裝置 | logit 最大差（單張） | logit 最大差（batch） | score 最大差 | 判斷一致 |
|---|---|---|---|---|
| CPU | 0.0 | 0.0 | 0.0 | 8/8 |
| CUDA | 0.0 | 0.0 | 6e-8 | 8/8 |

### 4.2 `--check-interface`（spec 第 4 節第 2 項）

- CPU、CUDA 皆通過（exit 0）。
- 分數與 sigmoid(logit) 一致；可微分；輸入梯度可取得；所有 resblock 與 `ln_post` 皆可取得 activation 與梯度。
- 設定檔：`configs/univfd.interface.json`（CPU）。

### 4.3 正常執行模式（spec 第 4 節第 2 項）

- CPU、CUDA 皆產生 `predictions.csv`、`summary.json`、`run_info.json`，欄位符合 spec。
- 輸入為 4 張人工圖，accuracy 1.0，**這不代表模型效能**。

### 4.4 耗時與記憶體（分工表：記錄 CPU／MPS／CUDA 的耗時及記憶體）

forward 為 `no_grad`，forward+backward 為對輸入求梯度；3 次重複取中位數，第一次為 warmup。

| 裝置 | batch | forward 總秒 | forward 每張秒 | forward+backward 總秒 | forward+backward 每張秒 | 峰值 VRAM（MiB，forward / forward+backward） |
|---|---|---|---|---|---|---|
| CPU | 1 | 0.735 | 0.735 | 1.433 | 1.433 | — |
| CPU | 2 | 1.415 | 0.707 | 2.867 | 1.433 | — |
| CPU | 4 | 2.638 | 0.660 | 5.648 | 1.412 | — |
| CPU | 8 | 5.292 | 0.662 | 12.122 | 1.515 | — |
| CUDA | 1 | 0.046 | 0.046 | 0.095 | 0.095 | 1663 / 2037 |
| CUDA | 2 | 0.113 | 0.056 | 0.174 | 0.087 | 1685 / 2358 |
| CUDA | 4 | 0.168 | 0.042 | 0.350 | 0.087 | 1718 / 3064 |
| CUDA | 8 | 0.326 | 0.041 | 0.699 | 0.087 | 1784 / 4475 |

CUDA 上 batch 8 的 forward+backward 峰值為 4475 MiB，8 GB 顯卡可以承受，沒有 OOM。

### 4.5 未包含的項目

- MPS（見第 2 節）
- 真實人臉的效能與準確率：屬於第二階段與 YUZI 的工作
- YUZI 真假人臉小清單的驗收：見第 6 節

## 5. 交接

- **DA**（spec 第 4 節第 3 項；分工表 UnivFD 共用驗收）：本記錄、權重版本與雜湊、前處理設定、可微分輸出與層資訊、`configs/univfd.interface.json`、`--check-interface` 與正常執行的輸出。
- **WEI**（分工表 UnivFD 的 ViT Grad-CAM）：第 3 節的層名、形狀、Token 重排方式與梯度測例；特別注意 `resblocks.23` 沒有 patch 梯度。

## 6. 待辦（YUZI 清單到位後）

- 用 YUZI 的真假人臉小清單重跑 `--check-interface` 與正常執行模式（spec 第 4 節第 1、2 項），結果附在本記錄後面。

## 7. 工時（分工表：W1-W2 記錄投入與剩餘預估工時）

- 今日投入：約 3 小時（估計；前段約 2 小時，後段約 1 小時）
- Phase 1 剩餘：約 1 小時（推送後等 YUZI 清單，重跑 3.3、3.4，更新本記錄）
- Phase 2 預估：約 5 小時以上（依分工表 3/5 對 5/5 的比例粗估，尚未細分）
