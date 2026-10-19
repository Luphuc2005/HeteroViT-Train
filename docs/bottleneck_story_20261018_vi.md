# Điểm nghẽn và câu chuyện thực nghiệm hiện tại — 18/10/2026

## Kết luận từ log

**Baseline AllReduce mỗi bước bị chi phối bởi đồng bộ trên mạng băng thông thấp. Local SGD đã giảm mạnh chi phí này. Ở baseline FP32 Local SGD H40 hiện tại, compute chiếm phần lớn thời gian, tiếp theo là chờ trước sync; CPU ranks 2/3 có local work dài nhất trong các cửa sổ sau khởi động. Vì vậy, giảm payload tiếp không đồng nghĩa giảm tương ứng tổng thời gian.**

FP16 AllGather có kết quả giảm thời gian chạy đủ 40 epochs. Delta AllGather có mean training time thấp nhất trong năm methods đã chạy, nhưng chỉ nhỉnh hơn Whole AllGather rất ít; chưa chứng minh lợi ích accuracy của delta hoặc lợi ích bổ sung của EF. Chưa có cơ sở nhận một biến thể FP16 là phương pháp tốt hơn một cách tổng quát về time-to-accuracy.

## 1. Đã kiểm tra những gì?

- Ba sweep gốc: seed42 `fp16_sweep_uydYAoMC`, seed43 `fp16_sweep_seed43_RIHAAkW2`, seed44 `fp16_sweep_seed44_swTaOOce`: 15 full runs.
- Paired seed43 `fp16_sweep_seed43_2tyCNQ2H`: Delta AllGather và Delta AllGather + EF, 2 full runs riêng.
- Tổng cộng **17 runs × 5 ranks = 85 step traces**, mỗi trace đủ 6.000 optimizer steps; mỗi rank có 160 syncs. Đã kiểm tra thứ tự epoch/step, vị trí sync 40/80/120/150, batch từng rank, timer hữu hạn và tổng các thành phần step.
- Đối chiếu completion time và final-test accuracy trong `.log` với `summary.csv`. Đối chiếu maxima communication/sync trong summary với tổng CSV gốc của từng rank. Không thấy các dấu hiệu `Traceback`, `[SYNC FAILED]`, `Non-finite` trong 17 launcher logs được kiểm tra; đây không phải khẳng định mọi warning TensorFlow đều vắng mặt.
- Kiểm tra timeline đủ 1.800.000 processed training samples/run và sample counts từng cửa sổ/từng rank. Learning rate trong 40 dòng epoch của mỗi run giữ 0,001.
- Đọc source snapshots ghi trong `source_stage.txt` để kiểm tra phạm vi timer; hash các file liên quan được lưu trong evidence. Không mặc định source hiện tại giống source chạy thí nghiệm.
- Đọc thêm trace rank0 của baseline AllReduce 20e và [network audit](network_heterogeneity_report.md). Baseline cũ dùng để giải thích chi phí hệ thống; không trộn vào thống kê ba seeds FP16.

CSV Whole FP16 cũ có **320 dòng data cho 160 syncs**: một dòng communicator và một dòng trainer có thêm `total_sync_ms`, trong khi header thiếu cột cuối. Audit dùng parser có sẵn [communication_rows](../tools/summarize_fp16_sweep.py) và chỉ lấy 160 dòng trainer hoàn chỉnh. Không cộng cả hai dòng.

Tái chạy audit từ thư mục `HeteroViT-MPI`:

```bash
python tools/audit_bottlenecks.py
```

Kết quả: [evidence JSON đầy đủ](bottleneck_audit_20261018.json), tạo bởi [script audit](../tools/audit_bottlenecks.py). Lượt kiểm tra này không chạy training mới và không sửa log gốc.

## 2. Baseline ban đầu: đồng bộ mỗi bước là chi phí lớn nhất

Nguồn: [step trace AllReduce 20e](../results/allreduce_baseline_300_20e_20261006_153923/step_timing_breakdown.csv), lọc rank0, đủ 3.000 steps.

| Thành phần rank0 | Tổng (s) |
|---|---:|
| Compute | 669,62 |
| Optimizer | 119,18 |
| Barrier trước AllReduce | 217,95 |
| MPI AllReduce | 4.820,39 |
| Tổng step | 5.831,44 |

Elapsed MPI AllReduce chiếm **82,66% tổng step**, trung bình khoảng **1,607 giây/lần**. Timer MPI có thể gồm chờ, không phải phép đo wire time thuần. Tuy nhiên, [network profile](network_heterogeneity_report.md) đo bandwidth payload lớn khoảng **11,2 MB/s**, gần giới hạn Ethernet **100 Mbps**, và microbenchmark collective khoảng 1,6 giây với payload FP32 khoảng 10,77 MB/rank. Hai nguồn bằng chứng nhất quán với chi phí truyền thông lớn khi đồng bộ mỗi bước.

Baseline này **đã dùng allocation lệch [266, 6, 6, 6, 16]**, không phải equal batches. Không dùng câu chuyện GPU phải chờ CPU do chia đều batch để mô tả chính run này. File [rebalance_log.csv](../results/allreduce_baseline_300_20e_20261006_153923/rebalance_log.csv) ghi allocation và quyết định scheduler.

Cơ chế đã xử lý trực tiếp chi phí đồng bộ lặp lại là **Local SGD / periodic model averaging**. Với H40, 150 local steps/epoch chỉ có 4 syncs/epoch, gồm cửa sổ cuối 30 steps. Trong các run cũ 20e, total loop là **99,01 phút** cho AllReduce mỗi bước và **23,35 phút** cho H40: khoảng **4,24 lần** về thời gian hoàn thành 20 epochs. Đây là phép so fixed-budget; không tự chuyển thành speedup tới cùng accuracy. Nguồn: [throughput comparison audit](throughput_comparison_audit_20261017.json).

Mạng có bandwidth thấp nhưng các cặp node gần đồng nhất về bandwidth payload lớn. Chưa có bằng chứng sắp lại thứ tự ring sẽ mang lại lợi ích đáng kể; thuật toán collective thực tế cũng không được coi là ring cố định chỉ từ tên trong comment code.

## 3. Baseline hiện tại H40: compute lớn nhất, chờ rank là phần đáng xử lý

Các số dưới là **trung bình tổng rank0 qua ba seeds**, đơn vị giây. Compute đã gồm optimizer; gather là elapsed sample-count gather trước model sync; model sync là toàn bộ quy trình đồng bộ weights, bất kể AllGather hay AllReduce.

| Method | Compute + optimizer | Gather trước sync | Toàn bộ model sync | Tổng step | Total training loop |
|---|---:|---:|---:|---:|---:|
| FP32 | 1.591,07 | 540,72 | 291,24 | 2.434,42 | 2.647,97 |
| Whole FP16 AG | 1.591,37 | 314,11 | 362,13 | 2.278,89 | 2.491,20 |
| Whole FP16 AR | 1.539,78 | 615,30 | 179,54 | 2.345,92 | 2.556,61 |
| Delta FP16 AG | 1.543,86 | 312,56 | 403,28 | 2.270,98 | 2.482,35 |
| Delta FP16 AR | 1.559,48 | 646,50 | 245,48 | 2.462,93 | 2.676,42 |

Đối với **FP32 H40**, chia theo toàn bộ training-loop timer:

- Compute + optimizer: **60,09%**.
- Gather trước sync: **20,42%**.
- Model sync: **11,00%**.
- Data fetch trong step: khoảng **0,43%**.
- Phần ngoài tổng step: khoảng **8,06%**, gồm validation và các overhead còn lại; `other_ms` trong step rất nhỏ.

Validation trung bình **177,67 giây/run**. `training_time_seconds - sum(epoch_time)` trung bình **24,45 giây**; `sum(train_time) - sum(step_ms)/1000` khoảng **11,42 giây** trên rank0. Những khoảng này không được tự gọi toàn bộ là checkpoint hoặc I/O: timer chưa tách từng nguyên nhân.

Nếu lấy mẫu số **tổng step**, ba tỷ lệ compute/gather/model sync tương ứng là **65,36% / 22,21% / 11,96%**. Hai bộ phần trăm khác nhau vì khác mẫu số. Không dùng chúng thay thế nhau.

Ở **Delta FP16 AG**, compute/gather/model sync lần lượt chiếm khoảng **62,19% / 12,59% / 16,25% training loop**. Đây là bằng chứng compression không làm biến mất chi phí local training hoặc chờ rank. Tỷ trọng lớn nhất là compute, nhưng tỷ trọng lớn không chứng minh toàn bộ compute có thể tối ưu hoặc đang bão hòa GPU: chưa có trace kernel/device utilization để kết luận điều đó.

### 3.1. Đối chiếu từng rank: không còn chỉ là suy đoán CPU chậm

FP32 H40, trung bình ba seeds:

| Rank / host | Batch | Compute + optimizer (s) | Gather trước sync (s) | Model sync trong step (s) | Payload MPI call (s) |
|---|---:|---:|---:|---:|---:|
| 0 / lab01, 2 GPU | 266 | 1.591,07 | 540,72 | 291,24 | 254,05 |
| 1 / lab02, CPU | 6 | 1.510,43 | 0,01 | 543,29 | 536,16 |
| 2 / lab03, CPU | 6 | 1.786,36 | 0,02 | 648,65 | 638,21 |
| 3 / lab04, CPU | 6 | 1.790,04 | 0,02 | 644,91 | 634,37 |
| 4 / lab05, CPU | 16 | 1.385,90 | 0,01 | 1.040,25 | 1.033,64 |

Trong 480 cửa sổ của ba runs FP32, rank3 có `data + compute` lớn nhất **385 lần**, rank2 **92 lần**, rank0 **3 lần**. Ba lần rank0 đều thuộc epoch1. Bỏ epoch1, **468/468 cửa sổ có local work lớn nhất ở rank2 hoặc rank3**. Delta AG và Delta AR cũng có rank2/3 lớn nhất trong toàn bộ 468 cửa sổ sau epoch1 của mỗi nhóm.

**Kết luận được phép:** allocation hiện tại vẫn có độ lệch thời gian local work; lab03/lab04 là hai ranks có local work dài nhất trong phạm vi đã đo, là đối tượng cần kiểm tra khi cân bằng tải.

**Giới hạn:** đây là thời lượng local work, chưa phải thứ tự thực tế tới sync. Mỗi rank có thể rời sync trước đó ở thời điểm khác nhau, và ghi telemetry có phần nằm ngoài timer step. Không có timestamp readiness được căn chỉnh để phân bổ chính xác 540,72 giây gather-wait cho từng worker. Không kết luận tắt hai CPU đó sẽ nhanh hơn hoặc toàn bộ wait sẽ biến mất.

### 3.2. Vì sao CPU rank có gather gần 0 mà vẫn đang chờ?

`comm.gather(..., root=0)` thu sample counts. Worker có thể gửi dữ liệu nhỏ và đi tiếp sớm; root phải nhận dữ liệu từ mọi rank. Phần chờ có thể xuất hiện trong collective/scalar check tiếp theo của worker.

Ví dụ rank4 có compute thấp hơn rank2/3, nhưng elapsed payload MPI cao nhất. **Thời gian MPI lớn nhất không có nghĩa rank4 là CPU tính toán chậm nhất hoặc đường mạng của rank4 chậm nhất.**

Do đó không lấy maximum payload collective khoảng 1.034 giây rồi cộng với 540,72 giây gather của rank0 hoặc với maximum sync. Các số đó có thể ghi cùng một khoảng chờ từ các ranks khác nhau. Cũng không dùng cột `is_straggler` trong epoch timeline làm bằng chứng arrival order: source đặt nhãn theo tổng compute epoch, không đo rank nào tới từng sync cuối cùng.

## 4. FP16 giảm bytes, nhưng total loop còn phụ thuộc waiting và đường code sync

Kết quả **chạy đủ 40 epochs**, đối chiếu trực tiếp với slide. Mean time tính từ số giây chưa làm tròn; phần trăm dưới đây là tỷ lệ hai mean thời gian.

| Method | Test accuracy TB ± sample SD (%) | Train TB (phút) | Thay đổi thời gian so FP32 | Payload/rank/sync |
|---|---:|---:|---:|---:|
| FP32 | 66,95 ± 2,17 | 44,13 | Baseline | 10,77 MB |
| Whole FP16 AG | 66,30 ± 1,29 | 41,52 | Giảm 5,92% | 5,39 MB |
| Whole FP16 AR | 65,95 ± 0,66 | 42,61 | Giảm 3,45% | 5,39 MB |
| Delta FP16 AG | 65,94 ± 0,32 | 41,37 | Giảm 6,25% | 5,39 MB |
| Delta FP16 AR | 66,26 ± 0,50 | 44,61 | Tăng 1,07% | 5,39 MB |

MB ở đây dùng bytes/1.000.000. Đây là buffer payload một rank, không phải toàn bộ traffic collective, không gồm mọi scalar checks/metadata. AllGather và AllReduce có cách trao đổi khác nhau; cùng bytes/rank không bảo đảm cùng network traffic.

Hai quan sát giải thích vì sao “MPI nhanh hơn” chưa đủ:

1. **Whole AR vs Whole AG:** model sync rank0 giảm **362,13 → 179,54 giây**, nhưng gather tăng **314,11 → 615,30 giây**. Total loop AR vẫn lâu hơn **65,41 giây**. Không thể chọn transport bằng một cột MPI rồi suy ra tổng thời gian.
2. **Delta AR vs Delta AG:** payload MPI rank0 giảm **305,71 → 136,37 giây**, model sync giảm **403,28 → 245,48 giây**, nhưng gather tăng **312,56 → 646,50 giây**. Total loop AR lâu hơn **194,08 giây**. Đây là mô tả timing của các runs riêng; chưa chứng minh đổi collective tự gây ra toàn bộ chênh lệch gather.

Whole AG có payload MPI rank0 khoảng **305,99 giây**, Delta AG khoảng **305,71 giây**, gần như bằng nhau. Tuy nhiên, full model sync của Delta AG cao hơn khoảng **41,15 giây**. Tổng loop của Delta AG chỉ thấp hơn Whole AG khoảng **8,85 giây**, tức **0,36%**; trong các runs này compute của rank0 cũng thấp hơn khoảng **47,51 giây**. Chênh lệch nhỏ này chưa chứng minh delta làm transport nhanh hơn. Theo metric throughput train của tài liệu trước, chênh lệch hai mean rates khoảng 0,40%; đó là công thức khác tỷ lệ mean time.

Source snapshot cho thấy model sync gồm lấy/flatten weights, payload collective, scalar collectives, đo divergence và set weights. Delta còn có subtract/reconstruct, norm FP64 để đo lỗi và SHA256 anchor checks mỗi sync. Chúng có chi phí nhưng **chưa có timer riêng đủ để quy 41,15 giây thêm toàn bộ cho hash/norm**. Không coi residual của phép trừ `model_sync - payload_MPI - encode - decode` là CPU overhead thuần vì còn có chờ trong scalar collectives.

## 5. Delta và EF: kết quả số học chưa chuyển thành lợi ích chất lượng

Trong [same-input diagnostic seed43](../results/fp16_sweep_seed43_E2dtk5hD/diagnostic_summary.json), mean global error chuẩn hóa theo FP32 global update giảm **0,48749% → 0,02045%**, tỷ số khoảng **23,84 lần**, khi chuyển Whole FP16 AG sang Delta FP16 AG. Đây là diagnostic 360 steps/10 rounds với trajectory thực FP32, không phải bằng chứng accuracy tăng tương ứng hoặc đại diện mọi epoch.

Trong full runs, Delta AG so Whole AG có test chênh **+0,69 / −1,69 / −0,09 điểm phần trăm** trên seeds42/43/44; mean Delta AG thấp hơn Whole AG khoảng **0,36 điểm phần trăm**. Delta AR có mean test cao hơn Whole AR khoảng **0,30 điểm phần trăm**, nhưng total loop cũng lâu hơn khoảng **119,82 giây**. Chưa có một biến thể delta thắng đồng thời thời gian và accuracy một cách nhất quán.

Paired seed43 có/không EF, dùng control chạy lại cùng đợt:

| Chỉ số | Delta AG | Delta AG + EF |
|---|---:|---:|
| Test accuracy (%) | 65,86 | 65,34 |
| Train (s) | 2.606,04 | 2.521,61 |
| Rank0 compute + optimizer (s) | 1.614,80 | 1.639,50 |
| Rank0 gather (s) | 369,20 | 255,62 |
| Rank0 model sync (s) | 402,97 | 407,38 |
| Payload/rank/sync (bytes) | 5.387.156 | 5.387.156 |

Thời gian giảm **84,43 giây** đi cùng gather giảm **113,58 giây**, trong khi compute tăng **24,70 giây**, model sync tăng **4,41 giây**. EF không giảm payload. Vì vậy không quy speedup quan sát được cho transport hoặc kết luận EF đã chữa nguyên nhân làm chậm. Test thấp hơn **0,52 điểm phần trăm** trong paired seed này; chưa có bằng chứng chọn EF làm bản tốt hơn baseline.

Ba seeds chưa đủ để khẳng định thứ hạng thống kê hoặc equivalence. Sample SD nhỏ hơn chỉ là mô tả ba kết quả, không chứng minh độ ổn định tổng quát. Cũng chưa có đối chứng xác định client drift/non-IID, sai số lượng tử hóa hay optimizer state local là nguyên nhân chính của khoảng cách accuracy. LR đang cố định và validation còn tăng cuối run ở nhiều cấu hình; chưa xác định trần chất lượng ở 40e.

## 6. Time-to-accuracy phải được báo riêng

Các bảng ở mục 4 và slide là thời gian hoàn thành 40 epochs. Kiểm tra phụ sau đây dùng **mốc validation 65% chọn hồi cứu**, lấy tổng `train.csv.epoch_time` đến epoch đầu tiên vượt/đạt mốc. Đây là proxy, không phải wall time chính xác từ launcher hoặc training-loop start: còn thiếu một số overhead, và độ phân giải là một epoch. Không dùng bảng này để bác số thời gian 40 epochs.

| Method | Seed42 (phút) | Seed43 (phút) | Seed44 (phút) |
|---|---:|---:|---:|
| FP32 | 34,26 | 37,34 | 37,79 |
| Whole FP16 AG | 34,64 | 32,01 | 34,59 |
| Whole FP16 AR | Chưa đạt trong 40e | 34,42 | 38,93 |
| Delta FP16 AG | 36,06 | 33,06 | 36,14 |
| Delta FP16 AR | 34,11 | 33,30 | 36,03 |

Ở mốc 66%, Whole AG và Delta AG chưa đạt trên seed44 dù hoàn thành training sớm hơn FP32. Delta AR đã đạt 65% sớm hơn FP32 trong cả ba seeds theo proxy trên, nhưng đây vẫn là kiểm tra hồi cứu ba seeds, không phải kết luận tổng quát hoặc bảo đảm ở mốc khác. Phải cố định target và cách đo trước experiment tiếp theo; runs chưa đạt phải được báo giữ nguyên, không loại khỏi phép so.

## 7. Câu chuyện có thể trình bày hiện tại

> Trên cụm 5 máy CPU/GPU dị thể nối Ethernet 100 Mbps, đồng bộ mỗi bước gây chi phí lớn. Local SGD giảm số lần trao đổi model và giảm mạnh thời gian chạy cùng số epochs. Sau bước này, compute và độ lệch thời gian local work giữa các ranks trở nên quan trọng; rank0 còn chờ đáng kể trước sync. FP16 giảm một nửa buffer payload và các biến thể AllGather giảm khoảng 6% thời gian hoàn thành 40 epochs so với FP32 H40 trong ba seeds. Truyền delta giảm sai số cập nhật trong diagnostic, nhưng chưa cải thiện accuracy nhất quán so với Whole FP16 và chỉ có chênh lệch nhỏ về tổng thời gian. EF trên một paired seed cũng chưa cho lợi ích chất lượng. Vì thế, kết quả hiện tại là một phân tích tradeoff hệ thống và chất lượng; chưa phải bằng chứng một thuật toán delta/EF vượt baseline về time-to-accuracy.

Đây là câu chuyện phù hợp các log đã có. Không gọi giảm payload 50% là giảm communication wall time 50%; không gán mọi khoảng chờ cho CPU ranks; không nhận thứ hạng mean của ba seeds là superiority thống kê.

## 8. Phép đo tiếp theo để phân biệt nguyên nhân

**Ưu tiên là đo đường găng từng round trước khi nhận một cơ chế mới tốt hơn.** Ghi thời điểm round bắt đầu, local work kết thúc, vào/ra sample gather và model sync của cả 5 ranks, với mốc thời gian có thể đối chiếu giữa máy. Tách thêm get/set weights, encode/decode, norm/hash/finite checks, scalar collectives và ghi telemetry. Ghi elapsed từ training-loop start ngay sau validation để có time-to-target trực tiếp.

Từ đó mới phân biệt:

- Chênh lệch local work ở lab03/lab04 có thực sự làm chậm round, hay phần đáng kể nằm trong host-side gaps/collective checks?
- Model sync tăng bao nhiêu vì codec/checks, bao nhiêu vì chờ? Rank có MPI time dài nhất có đang đến sớm hơn các ranks khác không?
- Nếu thay allocation để giảm readiness spread, wall time tới cùng target có giảm sau khi tính thay đổi training trajectory và sample weighting không?

Đối chứng chẩn đoán nên giữ H40, total samples, global batch 300, optimizer/LR và evaluation như control. So một allocation điều chỉnh theo timing với allocation hiện tại [266,6,6,6,16], giữ communication mode cố định; không đổi đồng thời quantizer và H. Allocation mới phải được chọn bằng pilot/validation, không chọn bằng test. Chỉ claim cải tiến khi cả wait/round time và time-to-target thực đo cùng cải thiện trong paired runs.

Nếu tiếp tục nghiên cứu quantization, cần đối chứng cùng collective và cùng guards để tách tác dụng quantizer khỏi transport/diagnostics; FP32 AllReduce so FP16 AllGather hiện tại chưa isolate riêng yếu tố precision. Các phép đo này là bước xác minh nguyên nhân, không phải kết quả đã đạt hay cam kết một allocation/codec mới chắc chắn nhanh hơn.
