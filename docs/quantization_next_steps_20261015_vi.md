# Hướng tiếp theo: lượng tử hóa delta trong Local SGD

Ngày 15/10/2026. Phạm vi đã xác nhận: nghiên cứu lượng tử hóa communication trên HeteroViT-MPI. Giữ training FP32, AdamW, H=40, sample weighting và recipe hiện tại để đánh giá tác dụng của quantizer. Outer momentum/overlap không phải trọng tâm. Đã thêm nhánh FP16 Delta AllGather + error feedback và hoàn tất full comparison 40 epochs trên seed43. Kết quả lần này chưa ủng hộ EF cải thiện accuracy.

**Ưu tiên hiện tại: FP16, chỉ seed43, đối chứng 2 config có/không EF.** Nhánh mới giữ phần sai số cast còn lại cho lần sync sau. Đây là giả thuyết cần thực nghiệm; không bảo đảm tăng accuracy. Payload vẫn 16-bit cùng kích thước, thêm một residual FP32 khoảng 10,77 MB/rank và chi phí tính toán. Các đề xuất INT8/INT4/adaptive ở phần sau là hướng tương lai.

Diagnostic Whole/Delta trên cùng inputs đã hoàn tất. Bộ chạy `smoke` và `full` hiện chọn đúng hai methods: `fp16_delta_allgather` và `fp16_delta_allgather_ef`. Mỗi method là một training trajectory riêng, thực sự áp dụng kết quả đồng bộ của mình.

**Diagnostic seed43 đã hoàn thành**

Nguồn: [diagnostic summary](../results/fp16_sweep_seed43_E2dtk5hD/diagnostic_summary.json). Run hoàn tất 360 optimizer steps/rank, 10 sync rounds trên 5 ranks. Cả 5 CSV có 10 records hữu hạn; Delta có error L2 thấp hơn Whole trong toàn bộ records. Global error metrics giữa các ranks nhất quán trong tolerance sai số tính norm FP64.

| Chỉ số trên rank0 | Whole FP16 AllGather | Delta FP16 AllGather |
|---|---:|---:|
| Mean error L2 / FP32 global-update norm | 0,487490% | 0,020450% |
| Max error L2 / FP32 global-update norm | 0,857447% | 0,020686% |
| Payload bytes/rank/sync | 5.387.156 | 5.387.156 |

Tỷ số hai mean relative errors khoảng 23,84. Đây là bằng chứng Delta giữ thông tin cập nhật tốt hơn Whole trong pilot đầu training này, không phải accuracy tăng 24 lần hoặc sai số giảm tương ứng ở mọi epoch. Trajectory thực vẫn FP32 nên không dùng validation accuracy của run để đánh giá chất lượng training bằng FP16.

Đã triển khai [FP16DeltaEFCommunicator](../src/training/fp16_delta_ef_communicator.py) như một nhánh riêng để giữ baseline cũ. Trong mỗi round: `u = delta + residual`, `q = FP16(u)`, `residual_next = u - FP32(q)`. Residual chỉ commit sau khi sync/finite/anchor-consensus checks thành công. Global averaging và reconstruction giữ FP32. Nhánh EF chỉ hỗ trợ AllGather với relative sample weights cố định; forced sync cuối epoch có cùng ratios vẫn được hỗ trợ.

EF nhằm kiểm tra sai số tích lũy; mức sai số Delta hiện đã nhỏ nên chưa biết có cải thiện accuracy thực tế không. `relative_l2_quantization_error` của nhánh EF đo lỗi cast trên `delta + residual`; `delta_l2_norm` vẫn đo raw delta. Nếu bật reference checking, chênh lệch với raw FP32 delta có thể chứa correction từ rounds trước, không chỉ lỗi cast hiện tại.

**Smoke có/không EF đã PASS trên 5 node thật:** mỗi method 120 optimizer steps/rank, 3 syncs, model/optimizer FP32 hữu hạn, weights bằng anchor và hash đồng nhất. Cả 5 residual snapshots EF đã được collect và khớp model hash cuối. Xem [báo cáo validation](../validation/fp16_ef_seed43_20261015/README.md). Full training đã hoàn tất với kết quả dưới đây.

**Kết quả full training seed43, H40, 40 epochs**

Nguồn: [summary CSV](../results/fp16_sweep_seed43_2tyCNQ2H/summary.csv) và [bản phân tích đã lưu](../validation/fp16_ef_seed43_20261015/full_comparison.json). Cả hai methods hoàn tất, 160 syncs/rank, đủ 40 epochs. Accuracy trong summary khớp final-test logs. EF được bật trong toàn bộ 160 telemetry records/rank; các residual snapshots cuối đều FP32 hữu hạn, cùng shape với model, anchor hash đồng nhất giữa 5 ranks.

| Chỉ số | Delta FP16 | Delta FP16 + EF |
|---|---:|---:|
| Test accuracy từ best-validation checkpoint | 65,86% | 65,34% |
| Best validation accuracy | 66,40% (epoch40) | 66,14% (epoch39) |
| Training time | 2606,04 s | 2521,61 s |
| Payload bytes/rank/sync | 5.387.156 | 5.387.156 |

EF thấp hơn 0,52 điểm phần trăm test accuracy trong seed này. Training time giảm 84,43 s (3,24%), nhưng rank0 gather-wait giảm từ 369,20 xuống 255,62 s trong khi compute tăng từ 1614,80 lên 1639,50 s, full model-sync tăng từ 402,97 lên 407,38 s. Vì vậy chưa quy thời gian giảm cho EF/transport. Đây là một paired seed; chưa chứng minh EF luôn kém, nhưng chưa có cơ sở chọn EF làm bản tốt hơn baseline.

Local Eq của EF chuẩn hóa theo corrected input `delta + residual`, nên không dùng việc Eq tương đương baseline để kết luận cumulative error đã được bù hay chưa. Không dùng smoke hoặc diagnostic FP32-trajectory trước đó để suy ra EF tăng accuracy. Giữ Delta FP16 không EF làm control cho công việc tiếp theo; không tự động mở rộng sang nhiều seeds chỉ để tìm kết quả thuận lợi.

Mỗi rank lưu `fp16_ef_state_rankN.npz` sau training, trước khi test có thể nạp best-validation checkpoint. File chứa anchor cuối training, residual FP32 và sample ratios. Collector chép cả 5 snapshots về result root. Đây là communication state cho last weights, không phải complete checkpoint của optimizer/RNG/dataloader. EF training từ chối `--resume` weights-only; chạy mới từ đầu để tránh vô tình reset memory.

**Config và lệnh chạy FP16 đã chuẩn bị**

- [Run plan](../configs/mpi/fp16_followup.yaml): diagnostic seed43 đã chạy; smoke 120 steps cho mỗi method; full seed43, 40 epochs cho mỗi method. Đây là plan cho launcher, không truyền trực tiếp vào `train_mpi.py`.
- [Baseline config](../configs/mpi/local_sgd_h40_fp16_delta_allgather_40e.yaml) và [EF config](../configs/mpi/local_sgd_h40_fp16_delta_allgather_ef_40e.yaml) giữ các settings giống nhau, khác tên experiment và `comm_error_feedback: true`. Base YAML seed42 được launcher override thành seed43, lưu lại trong config/log của run.
- [Diagnostic training config](../configs/mpi/fp16_same_inputs_seed43_smoke.yaml): FP32 Local SGD, H40, verify weights mỗi sync, cùng architecture/batches/AdamW/LR với baseline. Phải dùng helper chẩn đoán qua launcher để có shadow comparison.
- [Launcher](../scripts/run_fp16_followup.sh) mặc định chỉ chạy diagnostic, dừng khi pha đó hoàn tất.

Từ workspace root:

```bash
# Chỉ in toàn bộ plan, không SSH hoặc training.
bash HeteroViT-MPI/scripts/run_fp16_followup.sh --phase all --dry-run

# Kiểm tra end state cho cả hai methods, mỗi method 120 steps trên seed43.
bash HeteroViT-MPI/scripts/run_fp16_followup.sh --phase smoke

# Sau smoke PASS: chạy 2 config tuần tự, mỗi config 40 epochs, seed43.
nohup bash HeteroViT-MPI/scripts/run_fp16_followup.sh --phase full \
  > HeteroViT-MPI/fp16_ef_seed43_40e.log 2>&1 &

tail -f HeteroViT-MPI/fp16_ef_seed43_40e.log
```

Có thể sửa `diagnostic.max_steps` thành 120 và `diagnostic.epochs` thành 1 trong run plan để smoke ngắn hơn. Với mặc định 360 steps và 150 steps/epoch, có 10 lần sync vì vẫn sync ở cuối epoch. Đây là phép đo đầu training, chưa đại diện sai số ở cuối 40 epochs. Không đưa ra kết luận về lợi ích cả quá trình chỉ từ pilot này.

Mỗi job có source snapshot/output root riêng như sweep cũ. Diagnostic root ghi `run_kind.txt: same_input_fp16_diagnostic`, `diagnostic_summary.json` và CSV `ranks/fp32/rankN/fp16_same_inputs_rankN.csv`. Không tạo summary performance thông thường cho diagnostic. Cột error đo kết quả global của từng FP16 method so với FP32 control trên cùng local weights; relative error chuẩn hóa theo độ lớn FP32 global update. Đọc cùng absolute error khi update gần zero.

Anchor của diagnostic shadow ở round sau luôn là global model FP32 thực sự đã áp dụng. Model/optimizer không áp dụng kết quả shadow Whole/Delta. Shadow timing có thêm collectives và diagnostics, không dùng để công bố speedup. Full phase có/không EF chạy training thật, không chạy shadow/helper hoặc bật reference verification; root ghi `training_comparison` và có `summary.csv/json` như trước. Seeds và epochs được CLI override và ghi vào config/log của từng run.

Nếu chọn `--phase all` không có `--dry-run`, launcher tự chạy diagnostic, smoke rồi full; lệnh đó không có bước chờ người dùng đánh giá kết quả. Muốn xem kết quả trước, chạy các pha riêng như trên. Dừng khi một job lỗi; launcher không tự resume.

**Mục tiêu thực nghiệm**

Giảm số bits truyền và thời gian đạt cùng chất lượng, với sai số lượng tử hóa được kiểm soát. Accuracy không nhất thiết phải cao hơn FP32: giữ chất lượng gần FP32 với chi phí thấp hơn đã là cải tiến hợp lệ. Kết quả delta FP16 hiện tại chưa chứng minh lợi ích thêm so Whole FP16 cùng collective.

**Survey sát phạm vi**

| Công trình | Cơ chế liên quan | Vai trò đối chứng |
|---|---|---|
| [FedPAQ, AISTATS 2020](https://proceedings.mlr.press/v108/reisizadeh20a.html) | Quantize local model updates sau nhiều local steps. | Baseline ý tưởng delta; quantizer khác cast FP16. |
| [DAdaQuant, ICML 2022](https://proceedings.mlr.press/v162/honig22a.html) | Adaptive quantization theo round và client. | Bắt buộc khảo sát nếu đề xuất adaptive precision theo node/thời gian. |
| [AdaGQ, arXiv 2022](https://arxiv.org/abs/2212.08272) | Quantization resolution thay đổi theo round và heterogeneity của clients, tối ưu wall time. | Tiền lệ gần cho heterogeneous quantization; tốc độ compute và link cần đo riêng. |
| [FedFQ, arXiv 2024](https://arxiv.org/abs/2408.08977) | Fine-grained parameter-level bit allocation. | Tiền lệ cho phân bổ bits chi tiết; layer/block allocation đơn thuần chưa đủ khẳng định novelty. |
| [Qsparse-local-SGD, NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/hash/d202ed5bcfa858c15a9f383c3e386ab2-Abstract.html) | Kết hợp local updates, compression và error compensation. | Nguồn nền tảng cho error feedback; sparsity là thành phần riêng để ablate. |

Đây là các đối chứng liên quan, không phải khẳng định một paper là SOTA tuyệt đối. Kết quả FL trong bài không tự chuyển thành bảo đảm cho MPI/AdamW của repo.

**Bước 1: codec có thể đo được**

Triển khai dense blockwise symmetric INT8 cho delta với scales FP32. Giải mã trên từng rank và sample-weighted accumulate bằng FP32. Bắt đầu cùng transport AllGather với delta FP16 để giảm confounding. Không dùng trực tiếp `MPI.SUM` trên integers mang scales khác nhau theo rank.

Mỗi block dùng scale riêng; cần xử lý zero block, outliers/clipping, finite inputs và byte count gồm scales/padding. Block size 256 hoặc 1024 là hai lựa chọn pilot, chưa phải cấu hình đã được chọn bằng dữ liệu. Chốt rounding policy và dùng cùng policy khi đối chứng EF.

INT8 dùng khoảng một nửa bytes của FP16 cho phần tử payload; INT4 packed dùng khoảng một phần tư. Tỷ lệ thực phải cộng scales, padding và metadata. Cast sang `int8` trực tiếp không phải quantizer có scaling đúng.

**Bước 2: error feedback và phép thử tách biệt**

Với delta chưa nhân sample weight, residual FP32 ở mỗi rank:

```text
u_i = delta_i + residual_i
q_i = encode(u_i)
decoded_i = decode(q_i)
residual_i = u_i - decoded_i
global_next = anchor + sum_i p_i * decoded_i
```

Đối chứng cùng INT8 có/không residual. Công thức lưu lỗi encode/decode local, không mặc định bù mọi sai số reduction/reconstruction hoặc client drift. Ban đầu giữ p_i cố định; khi sampling/batch weights thay đổi cần phân tích lại quy tắc residual. Residual phải được checkpoint để resume đúng.

| Method | Tình trạng | Mục đích |
|---|---|---|
| FP32 Local SGD hiện có | Có | Reference hệ thống hiện tại; collective khác AG nên chưa isolate riêng quantizer. |
| Identity FP32 delta trên cùng AllGather | Cần thêm | Control cùng representation/collective khi cần phân tích sai số và chi phí codec. |
| Delta FP16 AllGather | Có | Baseline 16-bit hiện tại. |
| Delta INT8 AllGather | Cần thêm | Lợi ích/chi phí của 8-bit fixed precision. |
| Delta INT8 AllGather + EF | Cần thêm | Tác dụng riêng của bù sai số. |
| Delta INT4 packed + EF | Bước sau | Stress test tradeoff; triển khai sau khi INT8 hoạt động. |

Trước full run: numerical replay trên cùng delta tensors của tất cả ranks, sau đó real-cluster smoke khoảng 120–360 steps. Chưa có bộ local-delta snapshots được xác nhận trong khảo sát; cần thu có giới hạn nếu làm replay. Replay một tensor độc lập không chứng minh tác dụng residual qua nhiều rounds, phải replay sequence để đo EF.

Sau smoke, chạy pilot INT8 có/không EF cùng seed43, 40 epochs, cùng settings với baseline hiện có. Dùng cùng source/telemetry và rerun control nếu thay diagnostics. Finalists chạy paired seeds 42/43/44, thêm seeds mới nếu cần uncertainty đáng tin hơn. Chọn cấu hình bằng validation, không dùng test để tune.

**Bước 3: câu hỏi adaptive quantization**

Nếu fixed INT8 có lợi, khảo sát chọn 4/8/16 bits theo block, rank hoặc round dưới budget sai số và communication. Không mặc định mọi CPU rank nên dùng ít bits: batch weights, độ lớn delta, compute cost và network cost ảnh hưởng khác nhau. Mixed payload cần counts/offsets và có thể cần Allgatherv; phải tính overhead của quyết định và metadata.

Đo lỗi averaged update `||sum_i p_i*(decoded_i-delta_i)||`, đồng thời normalized error theo `||sum_i p_i*delta_i|| + epsilon`. Khi averaged delta gần zero, báo cả absolute error để tránh diễn giải sai relative ratio. Không chỉ dùng local Eq hoặc norm toàn weights làm tiêu chí.

Câu hỏi có thể theo đuổi: chọn precision để giảm thời gian sync trong khi giới hạn sai số cập nhật có trọng số trên cluster CPU/GPU không đồng nhất. Cần so fixed INT8/INT4 mạnh, policy thích nghi đơn giản và các tiền lệ DAdaQuant/AdaGQ/FedFQ. Adaptive bits và EF đều đã có; chưa thể gọi tổ hợp này là thuật toán mới trước khi xác định khác biệt và bằng chứng.

**Tiêu chí dừng/tiếp tục**

Ghi payload thực, encode/decode time, total sync, peak memory, averaged-update error, validation quality và time-to-target. Nếu INT8 giảm bytes nhưng CPU overhead khiến tổng thời gian không giảm, chưa đạt mục tiêu tốc độ. Nếu INT8 + EF không khác INT8 đáng kể, không thêm EF chỉ để tăng độ phức tạp. Nếu cả fixed và adaptive giữ chất lượng, so tại cùng byte budget hoặc cùng time-to-target; không đòi accuracy phải vượt FP32.
