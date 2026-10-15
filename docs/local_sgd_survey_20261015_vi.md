# Survey và thí nghiệm tiếp theo cho HeteroViT-MPI

Ngày khảo sát: **15/10/2026**. Phạm vi: communication compression, Local SGD và heterogeneous distributed training. Đây là khảo sát có chọn lọc các công trình liên quan, không phải systematic review hoặc xác nhận một phương pháp đứng đầu mọi benchmark. Các bài arXiv/report mới được phân biệt với bài đã công bố tại hội nghị. Chưa chạy training mới trong lượt survey này.

**1. Quyết định đề xuất**

**Cập nhật theo phạm vi người dùng: hướng chính là lượng tử hóa communication.** Lộ trình ưu tiên được thay bằng **delta INT8 theo block → error feedback → adaptive bit-width**, giữ H, optimizer và training recipe cố định khi so sánh. Xem [kế hoạch chuyên về lượng tử hóa](quantization_next_steps_20261015_vi.md). Outer momentum, đổi H và overlap chỉ là đối chứng/phân tích bổ sung.

Run FP32 H40 80 epochs bên dưới chỉ là đối chứng horizon tùy chọn, không phải bước nghiên cứu lượng tử hóa chính. Bước chính tiếp theo cần triển khai codec INT8 và đối chứng có/không error feedback; hiện repo chưa có hai mode này.

**2. Dữ liệu hiện tại nói gì?**

Nguồn: [evidence JSON](local_sgd_survey_20261015_evidence.json), tính từ `train.csv`, `step_timing_breakdown.csv` rank 0 và `summary.json` của các sweep:

- [Seed 42, sweep sau sửa evaluation](../results/fp16_sweep_uydYAoMC/summary.csv).
- [Seed 43, sweep đủ 5 methods](../results/fp16_sweep_seed43_RIHAAkW2/summary.csv).
- [Seed 44, sweep đủ 5 methods](../results/fp16_sweep_seed44_swTaOOce/summary.csv).

| Method | Test accuracy TB ± sample SD, % | Train TB, giây |
|---|---:|---:|
| FP32 | 66,95 ± 2,17 | 2647,97 |
| Whole FP16 AllGather | 66,30 ± 1,29 | 2491,20 |
| Whole FP16 AllReduce | 65,95 ± 0,66 | 2556,61 |
| Delta FP16 AllGather | 65,94 ± 0,32 | 2482,35 |
| Delta FP16 AllReduce | 66,26 ± 0,50 | 2676,42 |

Ba seed chưa chứng minh superiority, equivalence hoặc non-inferiority. Sample SD thấp hơn chưa đủ để kết luận phương pháp ổn định hơn. Seed 42 có FP32 nổi bật, cần giữ toàn bộ paired results thay vì chỉ báo trung bình.

LR của các run được đọc giữ `0.001` từ epoch đầu đến cuối. Ví dụ Delta AllGather seed 43: validation accuracy ở epoch 20/30/40 là **61,08 / 63,00 / 66,40%**; train accuracy từ epoch 30 đến 40 tăng **61,351 → 64,649%**. Dữ liệu không ủng hộ kết luận mọi run đã plateau ở 66%. Việc train thêm hoặc thay LR có cải thiện bao nhiêu vẫn chưa biết.

Trace rank 0, trung bình ba seed:

| Method | Local compute + optimizer, giây | Gather trước sync, giây | Toàn bộ model sync, giây | Tổng các step được đo, giây |
|---|---:|---:|---:|---:|
| FP32 | 1591,07 | 540,72 | 291,24 | 2434,42 |
| Whole FP16 AllGather | 1591,37 | 314,11 | 362,13 | 2278,89 |
| Whole FP16 AllReduce | 1539,78 | 615,30 | 179,54 | 2345,92 |
| Delta FP16 AllGather | 1543,86 | 312,56 | 403,28 | 2270,98 |
| Delta FP16 AllReduce | 1559,48 | 646,50 | 245,48 | 2462,93 |

Trong Local SGD, cột `barrier_ms` thực tế là elapsed time của sample-count `gather`; `allreduce_ms` là toàn bộ model sync kể cả khi collective là AllGather. Compute đã gồm optimizer. Tổng step không gồm đầy đủ validation/checkpoint/epoch overhead. Đây là quan sát trên rank 0; không cộng maxima của các ranks rồi gọi là critical path. Elapsed collective có thể chứa chờ rank khác, không đồng nghĩa thời gian truyền bytes thuần.

Suy luận cần kiểm tra: rank readiness/load balance và chi phí host-side là mục tiêu đáng khảo sát. Chưa có bằng chứng xác định CPU worker nào là thủ phạm hoặc chứng minh loại bỏ CPU sẽ nhanh hơn. Batch rank 0 chiếm 266/300 samples không phải phép đo phần trăm đóng góp vào convergence.

**3. Các hướng nghiên cứu liên quan và mức độ áp dụng**

| Công trình | Cơ chế và bằng chứng trong bài | Ý nghĩa với repo |
|---|---|---|
| [FedPAQ, AISTATS 2020](https://proceedings.mlr.press/v108/reisizadeh20a.html) | Periodic local updates, quantize chênh lệch local model với global model. | Tiền lệ trực tiếp cho delta; ý tưởng delta tự nó không mới. Quantizer và assumptions khác cast FP16/AdamW hiện tại. |
| [Qsparse-local-SGD, NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/hash/d202ed5bcfa858c15a9f383c3e386ab2-Abstract.html) | Local computation + sparsification + quantization + error compensation. | Baseline nền tảng nếu thêm sparse delta và residual. Cần tính cả index/metadata và chi phí encode/merge. |
| [SlowMo, ICLR 2020](https://www.pdl.cmu.edu/PDL-FTP/associated/SlowMo.pdf) | Momentum ở vòng ngoài sau một nhóm local updates; có thí nghiệm image classification và translation. | Ưu tiên cao cho ViT: thay quy tắc cập nhật global, có cơ hội cải thiện optimization mà không cần tăng số lần sync. Cần tuning và quy định rõ inner optimizer buffers. |
| [SCAFFOLD, ICML 2020](https://proceedings.mlr.press/v119/karimireddy20a.html) | Control variates để giảm client drift do non-IID data. | Chỉ ưu tiên nếu xác nhận statistical heterogeneity/drift; hardware chậm không đồng nghĩa dữ liệu non-IID. Có thêm state và traffic. |
| [FedCOM/FedCOMGATE, AISTATS 2021](https://proceedings.mlr.press/v130/haddadpour21a.html) | Compressed local updates; global step size và gradient tracking cho heterogeneous data. | Đối chứng cho correction/outer update; không được mặc định assumptions của bài đúng với AdamW và unequal batches. |
| [PowerSGD, NeurIPS 2019](https://arxiv.org/abs/1905.13727) | Low-rank gradient compression với aggregation bằng AllReduce. | Hợp lý cho tensor matrix lớn; áp dụng cho outer delta là adaptation cần kiểm chứng, không phải nguyên bản PowerSGD. |
| [DiLoCo, công bố ban đầu 2023](https://deepmind.google/research/publications/57039/) | Inner AdamW, outer Nesterov momentum, nhiều local steps; thí nghiệm language models. | Cấu trúc optimizer phù hợp để học hỏi vì repo đã dùng AdamW. Chưa có bảo đảm chuyển kết quả C4/LLM sang ViT/CIFAR-10. |
| [DeMo, arXiv 2024](https://arxiv.org/abs/2411.19870) | Decoupled local momentum, structured compression và trao đổi thành phần chọn lọc. | Là một optimizer/data-parallel algorithm, không phải thay một dòng cast trong communicator. Mức thay đổi cao. |
| [Streaming DiLoCo, arXiv 2025](https://arxiv.org/abs/2501.18512) | Fragment synchronization, overlap compute/communication và quantization. | Hướng giảm blocking/peak bandwidth. Cần quản lý version, snapshot và delayed merge; gọi nonblocking MPI đơn thuần chưa đủ. |
| [SparseLoCo, arXiv 2025; v3 23/07/2026](https://arxiv.org/abs/2508.15706v3) | Top-k + 2-bit quantization; báo cáo giữ 1–3% coordinates và loss thấp hơn dense DiLoCo trên các LLM đã thử. Tên v3: *Overcoming the Communication-Performance Tradeoff in LLM Pretraining*. | Tham chiếu compression hiện đại đáng thử sau khi có outer-optimizer baseline. Không suy diễn là giữ 1–3% bytes hoặc chắc chắn cải thiện ViT. |
| [Heterogeneous Low-Bandwidth Pre-Training of LLMs, arXiv 01/2026](https://arxiv.org/abs/2601.02360) | Kết hợp SparseLoCo với pipeline/activation compression, selective compression cho các participant khác nhau. | Chứng minh hướng co-design heterogeneity + compression đang được nghiên cứu. Topology model-parallel khác repo hiện tại. |
| [Decoupled DiLoCo, technical report 04/2026](https://arxiv.org/abs/2604.21428v1) | Learners bất đồng bộ, parameter fragments, quorum và token-weighted merge; hướng tới stragglers/failures. | Liên quan nút thắt chờ rank, nhưng cần đổi kiến trúc synchronous collective hiện tại. Không dùng số speedup datacenter của bài để dự đoán cluster này. |
| [LOSCAR-SGD, arXiv 05/2026](https://arxiv.org/abs/2605.20866v1) | Sparse model averaging + overlap + delay-corrected merge, cho workers có thể làm số local steps khác nhau; có phân tích smooth non-convex. | Gần câu hỏi heterogeneous Local SGD nhất trong nhóm mới khảo sát. Là preprint; chưa được tái hiện trên cluster này. |

Không có cơ sở gọi một dòng trong bảng là “SOTA cho HeteroViT-MPI”. Các benchmark, optimizer, model scale, network và tiêu chí khác nhau. Nhóm phương pháp hiện đại đáng đối chiếu là **outer optimization + compression có xử lý sai số + overlap/asynchrony**, với từng cơ chế được ablate riêng.

**4. Code hiện tại giải quyết được phần nào?**

`fp16_delta_communicator.py` hiện thực baseline anchor FP32, delta dense FP16 và reconstruction. Theo follow-up FP16, đã thêm `fp16_delta_ef_communicator.py` cho persistent residual FP32 và có/không EF trên seed43; xem [kế hoạch hiện tại](quantization_next_steps_20261015_vi.md). Chưa có outer momentum, sparse payload hoặc communication/computation overlap. Quantization-error norm và SHA256 anchor checking chạy mỗi sync, kể cả khi reference diagnostic tắt. Overhead này cần đo riêng trước khi diễn giải chênh lệch transport; không mặc định đây là nút thắt lớn nhất.

Repo đã có adaptive-H controllers V1/V2/V3 và scalar consensus metrics. Nên đánh giá chúng trước khi tạo một controller mới. H thích nghi có thể điều chỉnh tradeoff số sync/drift, nhưng không tự loại bỏ straggler khi mọi rank vẫn phải dự cùng collective.

`optimizer_state_sync: preserve_local` giữ moments local giữa các lần average weights. Đây là một lựa chọn thuật toán cần mô tả và đối chứng nếu thêm outer optimizer; không phải bằng chứng code sai và cũng không phải lý do để tự động AllReduce toàn bộ optimizer state.

**5. Các experiment bổ sung để phân biệt nguyên nhân**

Bảng này lưu các đối chứng rộng từ survey ban đầu. Thứ tự nghiên cứu lượng tử hóa hiện tại nằm trong [kế hoạch chuyên biệt](quantization_next_steps_20261015_vi.md); không cần hoàn tất toàn bộ bảng trước khi thử codec mới.

| Thứ tự | Run/analysis | Câu hỏi và quyết định |
|---|---|---|
| 0, đã phân tích log | Learning curves + timing từng rank; audit guards/diagnostics | Phân biệt horizon, rank waiting, sync overhead. Cần thêm per-rank readiness/round duration trước khi attribution cho worker cụ thể. |
| 1A, chạy được hiện tại | FP32 H40 seed43, 80 epochs, LR fixed hiện tại | So epoch 40 với 80 trong cùng run và val-target theo wall time. Nếu vẫn tăng rõ thì 40e chưa đủ để đánh giá trần accuracy. |
| 1B, cấu hình/launcher riêng | FP32 H8 so H40, cùng 40e/seed43/settings | Kiểm tra local-update drift có đáng kể. H8 tốn nhiều sync; nếu quality không tăng thì chưa ưu tiên giảm H. Với 150 steps/epoch và forced sync cuối epoch, H8 có 19 rounds/epoch, không phải 150/8 rounds nguyên. |
| 1C, cần triển khai LR schedule | FP32 H40 80e fixed LR so warmup 5e + cosine decay 80e | Isolate schedule; không đổi đồng thời data augmentation, model, H và precision. Horizon giống nhau. Schedule là lựa chọn thử nghiệm, không hứa trước tăng accuracy. |
| 2, baseline thuật toán ưu tiên | Thêm SlowMo hoặc DiLoCo-style outer momentum vào FP32; giữ horizon/schedule/H bằng control | Nếu validation tốt hơn control với cùng updates hoặc tới target nhanh hơn, mới ghép compression. Tune outer LR/momentum trên validation, ngân sách tuning bằng nhau cho các đối chứng. |
| 3, compression | Trên outer baseline đã chọn: FP32 → FP16 delta → blockwise INT8 + error feedback; sau đó mới sparse | Nếu FP16 chưa gây khoảng cách đáng kể, EF riêng FP16 có thể ít ích. INT8/sparse nhằm giảm bytes mạnh hơn; phải tính CPU/metadata/merge cost. |
| 4, hệ thống | GPU-only vs 5 nodes; adaptive batches/H hiện có; overlap có delayed merge | Nếu chờ rank lớn và compression không làm giảm wall time đủ, ưu tiên workload/participation hoặc overlap. Match samples, global batch và quality khi so GPU-only. |

Mỗi nhánh chỉ thay một yếu tố ở bước diagnostic. Seed43 là pilot được chọn để so với run đã có, không dùng pilot làm kết luận cuối. Sau khi chọn cấu hình bằng validation, chạy paired seeds 42/43/44 và thêm ít nhất hai seed mới cho finalists nếu ngân sách cho phép. Không chọn seed thuận lợi hoặc tune bằng test accuracy.

Chạy 80 epochs từ đầu có khoảng 12.000 local optimizer steps/rank theo 150 steps/epoch, gấp đôi budget 40e. Cải thiện cuối run vì thêm compute không phải bằng chứng thuật toán mới nhanh hơn. `--resume` hiện có chỉ phục hồi weights/epoch theo entrypoint; không mặc định phục hồi toàn bộ AdamW moments/RNG/dataloader để nối một experiment tương đương.

**6. Lệnh đối chứng horizon bằng script hiện có**

Từ workspace root:

```bash
# Xác nhận plan; không training/network.
bash HeteroViT-MPI/scripts/sweep_fp16_comparison.sh \
  --dry-run --methods fp32 --seed 43 --epochs 80

# Pilot horizon: chỉ một method, snapshot source riêng và collect reports 5 ranks.
bash HeteroViT-MPI/scripts/sweep_fp16_comparison.sh \
  --methods fp32 --seed 43 --epochs 80
```

Chỉ sau khi cần đối chứng precision ở cùng horizon mới chạy tiếp:

```bash
bash HeteroViT-MPI/scripts/sweep_fp16_comparison.sh \
  --methods fp16_delta_allgather --seed 43 --epochs 80
```

Script sweep hiện chưa nhận `--local-sgd-h`, outer-optimizer hoặc LR-schedule options. Các hàng 1B/1C/2/3 cần chuẩn bị launcher/config/code tương ứng, chưa thể gọi các lệnh trên và giả định đã bật chúng. Script base config validation vẫn kiểm tra YAML gốc 40e; `--epochs 80` là CLI override được truyền vào training.

**7. Thế nào là giải quyết được?**

Metric chính đề xuất: **wall time tới validation accuracy mục tiêu**, kèm chất lượng cuối cùng ở cùng sample/update budget. Có thể dùng 65% làm mốc chung ban đầu và 67% làm mốc khó hơn, khóa mốc trước run. Run chưa tới target được báo “chưa đạt trong budget”, không bỏ khỏi mean time. Ghi riêng thời gian training và end-to-end job, bytes thực tế kể cả metadata, encode/decode, gather wait, sync và peak memory.

Một mục tiêu kỹ thuật để chốt trước experiment: giảm ít nhất **10% wall time tới cùng target** trong khi mất không quá **0,5 điểm phần trăm** quality so control tại budget đã định. Đây là tiêu chí đề xuất, không phải kết quả đã đạt hoặc ngưỡng SOTA. Dùng paired seed differences và uncertainty; margin 0,5 không biến một point estimate thành chứng minh non-inferiority.

Nếu outer momentum không cải thiện control sau tuning giới hạn, dừng nhánh đó thay vì ghép thêm nhiều cơ chế. Nếu compression giảm bytes nhưng không giảm time-to-target, chuyển công sức sang chờ rank/overlap. Nếu GPU-only đã tốt hơn về time-to-quality, bài toán nghiên cứu cần giải thích điều kiện nào CPU participation mới hữu ích.

Hướng có thể phát triển thành câu hỏi nghiên cứu là **đồng thiết kế workload, H và compression theo cost/drift trên CPU/GPU không đồng nhất**. Mỗi thành phần đã có tiền lệ; ghép chúng chưa tự tạo novelty. Cần ablation, đối chứng fixed settings mạnh và so các phương pháp gần như LOSCAR/DiLoCo-style outer optimization theo phạm vi có thể tái hiện.

Có các cơ chế hợp lý để thử giải quyết communication, drift và stragglers. Chưa có bằng chứng bảo đảm chúng đồng thời tăng accuracy và tốc độ trên ViT 2,69M/CIFAR-10 của cluster này. Bước tiếp theo nên tạo bằng chứng phân biệt các nguyên nhân trước khi đầu tư một implementation lớn.
