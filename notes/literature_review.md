# 文獻回顧：LLM 信心校準 × 新聞預測股價 × Look-ahead bias

整理日期：2026-09-11。每篇論文的期刊、會議和年份，都從原始頁面（arXiv 摘要頁、會議論文集、作者任職學校的頁面）確認過。2026 年的幾篇（Hsiao、Look-Ahead-Bench、FinCAD、KalshiBench）目前仍是 arXiv preprint，沒有經過同儕審查。

---

## 一、LLM 的信心值能不能信（校準研究）

**已有共識：LLM 普遍過度自信，而且跟 RLHF 訓練有關。**
- [Tian et al., *EMNLP 2023*](https://aclanthology.org/2023.emnlp-main.330/)：ChatGPT、GPT-4、Claude 這類經過 RLHF 的模型，**模型自己寫的信心數字比 token 條件機率更準**，ECE 相對降低約 50%（TriviaQA、SciQ、TruthfulQA）。
- [Xiong et al., *ICLR 2024*](https://iclr.cc/virtual/2024/poster/18135)：模型寫的信心數字普遍過度自信，可能是在模仿人類的說話習慣。多次取樣看答案一致程度能緩解。白箱方法比較好，但差距不大。
- [Lyu et al., *AAAI 2025*](https://ojs.aaai.org/index.php/AAAI/article/view/34120)：用多次取樣的一致性當信心值，校準效果勝過既有的事後校正方法。另外發現**指令微調會讓校準變難**。
- [Leng et al., 2024（arXiv）](https://arxiv.org/abs/2410.09724)：RLHF 的獎勵模型本身偏好高信心回答，不管答案對不對，等於把模型訓練成過度自信。
- 綜述：[NAACL 2024](https://aclanthology.org/2024.naacl-long.366/)、[KDD 2025](https://dl.acm.org/doi/10.1145/3711896.3736569)。
- LLM 校準方法：[Thermometer, *ICML 2024*](https://proceedings.mlr.press/v235/shen24c.html)，跨任務學一個輔助模型來做 temperature scaling。

**2026 年的新進展：**
- [Sanz-Guerrero et al., *ACL 2026 Findings*](https://arxiv.org/abs/2606.03437)：發現「所有權偏誤」。模型對自己產生的答案，信心比對使用者給的同一個答案高最多 26%。把答案改成以使用者身分呈現，校準最多可改善 26%。
- [Zhao et al., *COLM 2026*](https://arxiv.org/abs/2604.01457)：從電路層級找到製造過度自信的 MLP 和注意力頭（集中在中後層），直接介入這些元件就能大幅改善校準。
- [Hsiao, 2026 年 9 月 preprint](https://arxiv.org/abs/2609.10996)：主張在 LLM 當評審的任務上（SummEval、AggreFact、HelpSteer2，最多 18 個模型），**2025 年以後的專有模型**用它自己寫的信心數字比用 logprob 可靠，傳統上「偏好 logprob」的建議不再成立。

**跟本專案的關係：** 我們在 gpt-4o-mini 上發現 logprob 幾乎都是 1（中位數約 1 − 2×10⁻⁹）、校準比模型自己寫的數字差得多。這跟 Tian 2023 的結論一致。有一點要注意：Hsiao 說的「logprob 比較好」是針對 2025 年前的模型、而且是評審任務；我們用的是 2025 年前的模型、預測方向任務、再加上嚴格 JSON schema 的受限解碼，所以兩者不算互相矛盾。

---

## 二、LLM 能不能從新聞預測股價

- [Lopez-Lira & Tang, *Journal of Financial Economics* vol. 184, 104335 (2026)](https://arxiv.org/abs/2304.07619)：這個領域的代表作，已正式刊出（期刊資訊經 [University of Florida 作者頁](https://scholars.ufl.edu/ytang1/publications) 確認）。有三個細節最重要：
  - 他們報告約 90% 命中率，但那是對**無法交易的初始反應**，也就是新聞出來當下的價格變動。
  - 後續的延續走勢（drift）也能預測，但**主要集中在小型股和負面新聞**。
  - 採用 LLM 的人越多，策略報酬就越低，代表市場變得更有效率。
- [Chen, Kelly & Xiu, *Expected Returns and Large Language Models*（SSRN，2023 GSU-RFS FinTech 最佳論文）](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4416687)：用 LLM 的文字嵌入向量做橫斷面報酬預測，涵蓋 16 個市場、13 種語言，扣除交易成本後仍有顯著的 Sharpe ratio，作者表示結果不是 look-ahead bias 造成的。

**跟本專案的關係：** 這篇大概是解讀我們結果最重要的一篇。我們刻意要求進場價必須晚於新聞發布，所以量到的正是文獻裡比較難預測的延續走勢，而不是初始反應。AAPL 又是資訊效率最高的超大型股，不是小型股。所以準確率跟擲硬幣差不多，**跟文獻是一致的，不是 harness 出了問題**。

---

## 三、Look-ahead bias 與記憶污染：這兩年最熱的議題

研究重心已經從「LLM 能不能預測」轉到「回測結果到底可不可信」。
- [Glasserman & Lin, 2023（arXiv）](https://arxiv.org/abs/2309.17322)：在模型訓練期間內，匿名化公司名反而表現更好，他們稱這是「分心效應」，比 look-ahead bias 的影響還大，大型公司尤其明顯。
- [Sarkar & Vafa, *ICML 2025*](https://icml.cc/virtual/2025/51018)：提出直接檢驗 look-ahead bias 的方法，在財報電話會議和選舉預測兩個應用中都找到證據。
- [Lopez-Lira, Tang & Zhu, 2025（arXiv / SSRN）](https://arxiv.org/abs/2504.14765)：GPT-4o 記得訓練期間內的經濟數據。**遮蔽公司名無效**，因為模型能從很少的上下文重建出實體和日期。在只有黑箱存取的情況下，分不出是真的預測能力還是記憶。
- [Didisheim, Fraschini & Somoza, *Economics Letters* 256 (2025)](https://ideas.repec.org/a/eee/ecolet/v256y2025ics0165176525004392.html)：記憶效應在低頻資料和大盤指數上最嚴重，在高頻、個別標的上很輕微；小模型的偏誤明顯較低。
- 解法：
  - [ChronoBERT / ChronoGPT（He, Lv, Manela & Wu, 2025，列入 AEA 2026 年會議程）](https://arxiv.org/abs/2502.21206)：只用每個時間點之前的資料訓練模型，結果顯示 look-ahead bias 影響有限。
  - [偵測指標 Lookahead Propensity（Gao, Jiang & Yan, 2025/12，2026/06 修訂）](https://arxiv.org/abs/2512.23847)：衡量模型「知道結果」的機率，發現它在訓練截止日之後幾乎歸零。
  - [Look-Ahead-Bench（Benhenda, 2026/01）](https://arxiv.org/abs/2601.13770)：專門衡量這個偏誤的評測基準，比較一般 LLM 和 point-in-time LLM。
  - [FinCAD（Li et al., 2026/05）](https://arxiv.org/abs/2605.24564)：在推論時壓制模型的記憶，在被記住的日期上把回測報酬壓低了最多 67%，對 2025 年樣本外的影響很小。

**跟本專案的關係：** 我們選用訓練截止日之後的資料、而不是遮蔽公司名，正是目前文獻認為最可靠的做法。「遮蔽無效」這個發現直接支持了我們的選擇。

---

## 四、預測校準的評估方法

- [ForecastBench, *ICLR 2025*](https://arxiv.org/abs/2409.19839)：整體 Brier 分數，超級預測者 0.076，GPT-4o 0.130（數字取自 ICLR 版本，[排行榜](https://www.forecastbench.org/leaderboards/)之後持續更新）。
- [Halawi et al., *NeurIPS 2024*](https://neurips.cc/virtual/2024/poster/95949)：加上檢索的 LM 系統，在訓練截止日之後的題目上接近人類預測者群體的水準。
- [KalshiBench（Nel, 2025/12 preprint）](https://arxiv.org/abs/2512.16030)：用 300 題預測市場題目測前沿模型，**全部過度自信**（ECE 從 0.120 到 0.395）。推理強化的模型校準反而更差，只有一個模型的 Brier skill 是正的。
- **小樣本的校準方法，對我們最實用：**
  - [CORP 可靠度圖（Dimitriadis, Gneiting & Jordan, *PNAS 2021*）](https://www.pnas.org/doi/abs/10.1073/pnas.2016191118)：用保序迴歸（PAV 演算法）自動決定分組，不需要人為設定分箱邊界，結果可重現，還能把分數拆解成「校準、分辨力、不確定性」三部分。
  - [Roelofs et al., *AISTATS 2022*](https://proceedings.mlr.press/v151/roelofs22a.html)：等寬分箱的 ECE 偏差大，等樣本數分箱（ECE_SWEEP）或去偏估計比較好。
  - [Smooth ECE（Błasiok & Nakkiran, *ICLR 2024*）](https://arxiv.org/abs/2309.12236)：用核平滑取代分箱，有現成套件 `pip install relplot`。

---

## 對本專案最值得做的下一步（依價值排序）

1. **加入 CORP 可靠度圖和分數拆解。** 我們目前用等寬分箱，在 n=77 時正是 Roelofs 指出偏差最大的情況。CORP 能把「不準」拆成「校準差」和「分辨不出對錯」兩部分，正好對應我們觀察到的兩個問題。
2. **把 Lopez-Lira & Tang 的脈絡寫進 README。** 說明我們量的是延續走勢、標的是超大型股，所以接近擲硬幣的結果符合文獻預期。
3. **第三種信心值：多次取樣一致性**（Lyu 2025、Xiong 2024）。這是目前多篇研究中校準最好的方法，而且可以直接沿用現有的成對比較框架。
