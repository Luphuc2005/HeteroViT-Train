# Kết quả FP16 theo seed — 15/10/2026

Phân tích log từng rank và câu chuyện thực nghiệm cập nhật: [Điểm nghẽn — 18/10/2026](bottleneck_story_20261018_vi.md), kèm [evidence audit 17 runs](bottleneck_audit_20261018.json).

Tất cả full runs bên dưới dùng 40 epochs, H40, global batch 300, sample-weighted averaging, AdamW, LR 0.001 và optimizer state local. Accuracy là test accuracy sau khi nạp checkpoint tốt nhất theo validation. Mỗi run có 160 syncs/rank.

**Test accuracy (%)**

| Method | Seed 42 | Seed 43 | Seed 44 | TB ± sample SD trên 3 seeds |
|---|---:|---:|---:|---:|
| FP32 | 69,41 | 65,30 | 66,14 | 66,95 ± 2,17 |
| Whole FP16 AllGather | 64,97 | 67,55 | 66,38 | 66,30 ± 1,29 |
| Whole FP16 AllReduce | 65,20 | 66,21 | 66,45 | 65,95 ± 0,66 |
| Delta FP16 AllGather | 65,66 | 65,86 | 66,29 | 65,94 ± 0,32 |
| Delta FP16 AllReduce | 65,84 | 66,12 | 66,81 | 66,26 ± 0,50 |
| Delta FP16 AllGather + EF | — | 65,34 | — | Chỉ có seed43 |

**Thời gian train của các sweep gốc (phút)**

| Method | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|
| FP32 | 43,67 | 44,55 | 44,18 |
| Whole FP16 AllGather | 41,60 | 42,16 | 40,80 |
| Whole FP16 AllReduce | 43,03 | 43,69 | 41,11 |
| Delta FP16 AllGather | 41,07 | 40,29 | 42,76 |
| Delta FP16 AllReduce | 44,89 | 43,54 | 45,39 |

**Đối chứng EF riêng trên seed43**

| Method | Test accuracy (%) | Train (giây) | Train (phút) |
|---|---:|---:|---:|
| Delta FP16 AllGather | 65,86 | 2606,04 | 43,43 |
| Delta FP16 AllGather + EF | 65,34 | 2521,61 | 42,03 |

EF giảm 0,52 điểm phần trăm test accuracy trong paired run seed43. Training time giảm 84,43 giây (3,24%), đi cùng gather-wait thấp hơn; payload không giảm và chưa quy chênh lệch thời gian cho EF. Baseline Delta AllGather đã được chạy lại trong paired experiment: accuracy vẫn 65,86%, nhưng thời gian 2606,04 giây khác sweep gốc 2417,58 giây. Dùng cùng đợt chạy để so thời gian EF.

**Nguồn và cách chọn runs**

- Seed 42: [summary](../results/fp16_sweep_uydYAoMC/summary.csv).
- Seed 43: [summary](../results/fp16_sweep_seed43_RIHAAkW2/summary.csv).
- Seed 44: [summary](../results/fp16_sweep_seed44_swTaOOce/summary.csv).
- EF paired seed43: [summary](../results/fp16_sweep_seed43_2tyCNQ2H/summary.csv), [bản phân tích đã lưu](../validation/fp16_ef_seed43_20261015/full_comparison.json).
- [Evidence 15 runs của sweep gốc](local_sgd_survey_20261015_evidence.json).

Mỗi seed chỉ dùng một sweep đủ năm methods trong bảng tổng hợp. Seed42 dùng sweep `uydYAoMC` sau cập nhật evaluation; không gộp sweep cũ `83LF5CdI`. Các lần chạy lại seed43 không được tính thành seed độc lập. Smoke 120 steps và diagnostic 360 steps không có final-test comparison và không nằm trong bảng full training.

**Nhận xét**

- FP32 có mean test accuracy cao nhất trong ba seeds đã chạy, nhưng biến động lớn. Ba seeds chưa đủ để khẳng định thứ hạng thống kê.
- Delta AllGather so Whole AllGather chênh +0,69 / -1,69 / -0,09 điểm phần trăm trên seed42/43/44; lợi ích accuracy chưa nhất quán.
- Delta AllReduce so Whole AllReduce chênh +0,64 / -0,09 / +0,36 điểm phần trăm; vẫn cần tính overhead khi đánh giá hiệu quả.
- EF chỉ có một paired seed, hiện chưa ủng hộ cải thiện accuracy. Không suy diễn rằng EF luôn kém ở mọi seed.

**Chi tiết từng seed**

Các bảng dưới dùng AG = AllGather, AR = AllReduce. Train loss/accuracy là telemetry trong local training của rank0, không phải phép evaluate lại toàn bộ train set trên global model cuối epoch. Validation đo trên model đã đồng bộ. Test dùng best-validation checkpoint; vì vậy không bắt buộc tương ứng epoch40.

Thời gian train là elapsed time của training loop, gồm validation và overhead trong loop, chưa cộng final test. Payload-collective và total-sync ở bảng cost là maximum của tổng từng rank; hai maxima có thể thuộc các ranks khác nhau và không được cộng với nhau hoặc cộng với rank0 trace. Elapsed MPI call có thể gồm chờ, không phải đo wire time thuần.

**Seed 42: chất lượng model**

Nguồn: [summary](../results/fp16_sweep_uydYAoMC/summary.json). Mọi method đủ 6000 local optimizer steps theo trace, 40 epochs và 160 syncs/rank.

| Method | Train loss E40 | Train acc E40 (%) | Val acc E40 (%) | Best val (%) / epoch | Test loss | Test acc (%) |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 0,89843 | 67,835 | 68,82 | 68,82 / E40 | 0,8661 | 69,41 |
| Whole FP16 AG | 0,96441 | 65,110 | 65,66 | 66,06 / E35 | 0,9720 | 64,97 |
| Whole FP16 AR | 0,97480 | 64,877 | 64,82 | 64,82 / E40 | 0,9609 | 65,20 |
| Delta FP16 AG | 0,96831 | 65,246 | 66,34 | 66,40 / E38 | 0,9572 | 65,66 |
| Delta FP16 AR | 0,94494 | 65,995 | 66,16 | 66,48 / E35 | 0,9471 | 65,84 |

**Seed 42: chi phí và chênh lệch so FP32**

| Method | Train (s) | Test − FP32 (điểm %) | Giảm thời gian so FP32 (%) | Max-rank payload collective (s) | Max-rank total sync (s) |
|---|---:|---:|---:|---:|---:|
| FP32 | 2619,95 | +0,00 | +0,00 | 989,69 | 996,27 |
| Whole FP16 AG | 2495,99 | -4,44 | +4,73 | 305,78 | 895,34 |
| Whole FP16 AR | 2581,69 | -4,21 | +1,46 | 132,42 | 925,50 |
| Delta FP16 AG | 2464,07 | -3,75 | +5,95 | 306,73 | 857,19 |
| Delta FP16 AR | 2693,61 | -3,57 | -2,81 | 136,93 | 1043,54 |

Giảm thời gian dương = nhanh hơn FP32; âm = chậm hơn. FP32 payload 10.774.312 bytes/rank/sync; các FP16 methods 5.387.156 bytes. Đây là buffer một rank, chưa phải tổng traffic của collective.

**Seed 42: validation accuracy theo tiến trình (%)**

| Method | E10 | E20 | E30 | E40 |
|---|---:|---:|---:|---:|
| FP32 | 54,94 | 60,52 | 64,14 | 68,82 |
| Whole FP16 AG | 54,88 | 60,30 | 63,98 | 65,66 |
| Whole FP16 AR | 55,50 | 60,16 | 62,60 | 64,82 |
| Delta FP16 AG | 54,56 | 59,72 | 64,52 | 66,34 |
| Delta FP16 AR | 56,24 | 61,18 | 65,06 | 66,16 |

**Seed 42: phân rã thời gian rank0 (giây)**

| Method | Data | Compute + optimizer | Gather trước sync | Toàn bộ model sync | Tổng step |
|---|---:|---:|---:|---:|---:|
| FP32 | 11,26 | 1497,22 | 607,97 | 290,58 | 2407,14 |
| Whole FP16 AG | 11,33 | 1502,83 | 405,82 | 361,40 | 2281,51 |
| Whole FP16 AR | 11,25 | 1509,41 | 672,34 | 178,83 | 2371,96 |
| Delta FP16 AG | 11,26 | 1493,13 | 342,40 | 403,50 | 2250,42 |
| Delta FP16 AR | 11,22 | 1496,96 | 729,96 | 244,70 | 2482,96 |

Cột gather là `barrier_ms` trong schema hiện tại; cột model sync là `allreduce_ms` kể cả khi dùng AllGather. Tổng step còn có phần `other_ms` nhỏ và không bao gồm toàn bộ validation/checkpoint/epoch overhead.

**Seed 42: sai số local quantization của delta**

| Method | Mean Eq trên 5 ranks × 160 rounds | Max Eq trên 5 ranks × 160 rounds |
|---|---:|---:|
| Delta FP16 AG | 2.0736275e-04 | 2.0883459e-04 |
| Delta FP16 AR | 2.3785918e-04 | 5.0074707e-04 |

Eq là relative L2 ratio của local encode/decode, không phải điểm accuracy hoặc sai số averaged global update. Eq AllReduce còn chịu scaling trước cast; không so trực tiếp Eq của delta với sai số lượng tử hóa whole weights.

Nguồn chi tiết từng method:

- FP32: [train.csv](../results/fp16_sweep_uydYAoMC/ranks/fp32/rank0/train.csv), [log](../results/fp16_sweep_uydYAoMC/fp32.log).
- Whole FP16 AG: [train.csv](../results/fp16_sweep_uydYAoMC/ranks/fp16_whole_allgather/rank0/train.csv), [log](../results/fp16_sweep_uydYAoMC/fp16_whole_allgather.log).
- Whole FP16 AR: [train.csv](../results/fp16_sweep_uydYAoMC/ranks/fp16_whole_allreduce/rank0/train.csv), [log](../results/fp16_sweep_uydYAoMC/fp16_whole_allreduce.log).
- Delta FP16 AG: [train.csv](../results/fp16_sweep_uydYAoMC/ranks/fp16_delta_allgather/rank0/train.csv), [log](../results/fp16_sweep_uydYAoMC/fp16_delta_allgather.log).
- Delta FP16 AR: [train.csv](../results/fp16_sweep_uydYAoMC/ranks/fp16_delta_allreduce/rank0/train.csv), [log](../results/fp16_sweep_uydYAoMC/fp16_delta_allreduce.log).

**Seed 43: chất lượng model**

Nguồn: [summary](../results/fp16_sweep_seed43_RIHAAkW2/summary.json). Mọi method đủ 6000 local optimizer steps theo trace, 40 epochs và 160 syncs/rank.

| Method | Train loss E40 | Train acc E40 (%) | Val acc E40 (%) | Best val (%) / epoch | Test loss | Test acc (%) |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 0,98524 | 64,313 | 66,30 | 66,30 / E40 | 0,9642 | 65,30 |
| Whole FP16 AG | 0,92240 | 66,709 | 68,24 | 68,24 / E40 | 0,8965 | 67,55 |
| Whole FP16 AR | 0,95091 | 65,511 | 66,88 | 67,20 / E36 | 0,9478 | 66,21 |
| Delta FP16 AG | 0,97607 | 64,649 | 66,40 | 66,40 / E40 | 0,9503 | 65,86 |
| Delta FP16 AR | 0,97005 | 64,729 | 66,68 | 66,68 / E40 | 0,9501 | 66,12 |

**Seed 43: chi phí và chênh lệch so FP32**

| Method | Train (s) | Test − FP32 (điểm %) | Giảm thời gian so FP32 (%) | Max-rank payload collective (s) | Max-rank total sync (s) |
|---|---:|---:|---:|---:|---:|
| FP32 | 2673,00 | +0,00 | +0,00 | 1066,35 | 1072,95 |
| Whole FP16 AG | 2529,34 | +2,25 | +5,37 | 306,44 | 943,21 |
| Whole FP16 AR | 2621,33 | +0,91 | +1,93 | 132,44 | 940,23 |
| Delta FP16 AG | 2417,58 | +0,56 | +9,56 | 306,53 | 823,04 |
| Delta FP16 AR | 2612,51 | +0,82 | +2,26 | 137,18 | 1006,91 |

Giảm thời gian dương = nhanh hơn FP32; âm = chậm hơn. FP32 payload 10.774.312 bytes/rank/sync; các FP16 methods 5.387.156 bytes. Đây là buffer một rank, chưa phải tổng traffic của collective.

**Seed 43: validation accuracy theo tiến trình (%)**

| Method | E10 | E20 | E30 | E40 |
|---|---:|---:|---:|---:|
| FP32 | 54,56 | 59,80 | 64,08 | 66,30 |
| Whole FP16 AG | 53,40 | 60,82 | 64,96 | 68,24 |
| Whole FP16 AR | 52,20 | 60,76 | 63,00 | 66,88 |
| Delta FP16 AG | 53,36 | 61,08 | 63,00 | 66,40 |
| Delta FP16 AR | 53,88 | 60,90 | 63,22 | 66,68 |

**Seed 43: phân rã thời gian rank0 (giây)**

| Method | Data | Compute + optimizer | Gather trước sync | Toàn bộ model sync | Tổng step |
|---|---:|---:|---:|---:|---:|
| FP32 | 11,38 | 1622,98 | 532,24 | 291,76 | 2458,47 |
| Whole FP16 AG | 10,81 | 1649,75 | 299,04 | 362,45 | 2322,17 |
| Whole FP16 AR | 10,82 | 1571,60 | 651,20 | 178,97 | 2412,69 |
| Delta FP16 AG | 11,03 | 1636,08 | 156,85 | 402,74 | 2206,82 |
| Delta FP16 AR | 11,62 | 1673,71 | 460,63 | 246,40 | 2392,49 |

Cột gather là `barrier_ms` trong schema hiện tại; cột model sync là `allreduce_ms` kể cả khi dùng AllGather. Tổng step còn có phần `other_ms` nhỏ và không bao gồm toàn bộ validation/checkpoint/epoch overhead.

**Seed 43: sai số local quantization của delta**

| Method | Mean Eq trên 5 ranks × 160 rounds | Max Eq trên 5 ranks × 160 rounds |
|---|---:|---:|
| Delta FP16 AG | 2.0735350e-04 | 2.0884878e-04 |
| Delta FP16 AR | 2.3733143e-04 | 5.3365946e-04 |

Eq là relative L2 ratio của local encode/decode, không phải điểm accuracy hoặc sai số averaged global update. Eq AllReduce còn chịu scaling trước cast; không so trực tiếp Eq của delta với sai số lượng tử hóa whole weights.

Nguồn chi tiết từng method:

- FP32: [train.csv](../results/fp16_sweep_seed43_RIHAAkW2/ranks/fp32/rank0/train.csv), [log](../results/fp16_sweep_seed43_RIHAAkW2/fp32.log).
- Whole FP16 AG: [train.csv](../results/fp16_sweep_seed43_RIHAAkW2/ranks/fp16_whole_allgather/rank0/train.csv), [log](../results/fp16_sweep_seed43_RIHAAkW2/fp16_whole_allgather.log).
- Whole FP16 AR: [train.csv](../results/fp16_sweep_seed43_RIHAAkW2/ranks/fp16_whole_allreduce/rank0/train.csv), [log](../results/fp16_sweep_seed43_RIHAAkW2/fp16_whole_allreduce.log).
- Delta FP16 AG: [train.csv](../results/fp16_sweep_seed43_RIHAAkW2/ranks/fp16_delta_allgather/rank0/train.csv), [log](../results/fp16_sweep_seed43_RIHAAkW2/fp16_delta_allgather.log).
- Delta FP16 AR: [train.csv](../results/fp16_sweep_seed43_RIHAAkW2/ranks/fp16_delta_allreduce/rank0/train.csv), [log](../results/fp16_sweep_seed43_RIHAAkW2/fp16_delta_allreduce.log).

**Seed 44: chất lượng model**

Nguồn: [summary](../results/fp16_sweep_seed44_swTaOOce/summary.json). Mọi method đủ 6000 local optimizer steps theo trace, 40 epochs và 160 syncs/rank.

| Method | Train loss E40 | Train acc E40 (%) | Val acc E40 (%) | Best val (%) / epoch | Test loss | Test acc (%) |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 0,94163 | 65,947 | 64,92 | 66,36 / E39 | 0,9337 | 66,14 |
| Whole FP16 AG | 0,95834 | 65,378 | 64,46 | 65,40 / E39 | 0,9336 | 66,38 |
| Whole FP16 AR | 0,96617 | 65,118 | 65,06 | 66,06 / E39 | 0,9392 | 66,45 |
| Delta FP16 AG | 0,95261 | 65,356 | 64,38 | 65,94 / E39 | 0,9399 | 66,29 |
| Delta FP16 AR | 0,95140 | 65,441 | 65,28 | 65,86 / E39 | 0,9321 | 66,81 |

**Seed 44: chi phí và chênh lệch so FP32**

| Method | Train (s) | Test − FP32 (điểm %) | Giảm thời gian so FP32 (%) | Max-rank payload collective (s) | Max-rank total sync (s) |
|---|---:|---:|---:|---:|---:|
| FP32 | 2650,96 | +0,00 | +0,00 | 1044,89 | 1051,43 |
| Whole FP16 AG | 2448,27 | +0,24 | +7,65 | 306,61 | 819,82 |
| Whole FP16 AR | 2466,80 | +0,31 | +6,95 | 135,05 | 811,84 |
| Delta FP16 AG | 2565,39 | +0,15 | +3,23 | 307,00 | 948,65 |
| Delta FP16 AR | 2723,15 | +0,67 | -2,72 | 137,45 | 1082,55 |

Giảm thời gian dương = nhanh hơn FP32; âm = chậm hơn. FP32 payload 10.774.312 bytes/rank/sync; các FP16 methods 5.387.156 bytes. Đây là buffer một rank, chưa phải tổng traffic của collective.

**Seed 44: validation accuracy theo tiến trình (%)**

| Method | E10 | E20 | E30 | E40 |
|---|---:|---:|---:|---:|
| FP32 | 54,18 | 59,86 | 64,54 | 64,92 |
| Whole FP16 AG | 54,96 | 60,46 | 63,34 | 64,46 |
| Whole FP16 AR | 53,60 | 59,28 | 64,08 | 65,06 |
| Delta FP16 AG | 54,08 | 59,72 | 64,46 | 64,38 |
| Delta FP16 AR | 54,76 | 60,86 | 64,52 | 65,28 |

**Seed 44: phân rã thời gian rank0 (giây)**

| Method | Data | Compute + optimizer | Gather trước sync | Toàn bộ model sync | Tổng step |
|---|---:|---:|---:|---:|---:|
| FP32 | 11,21 | 1653,02 | 481,94 | 291,37 | 2437,66 |
| Whole FP16 AG | 11,35 | 1621,52 | 237,46 | 362,53 | 2232,99 |
| Whole FP16 AR | 11,47 | 1538,33 | 522,37 | 180,82 | 2253,12 |
| Delta FP16 AG | 11,21 | 1502,36 | 438,44 | 403,59 | 2355,72 |
| Delta FP16 AR | 11,20 | 1507,77 | 748,91 | 245,33 | 2513,33 |

Cột gather là `barrier_ms` trong schema hiện tại; cột model sync là `allreduce_ms` kể cả khi dùng AllGather. Tổng step còn có phần `other_ms` nhỏ và không bao gồm toàn bộ validation/checkpoint/epoch overhead.

**Seed 44: sai số local quantization của delta**

| Method | Mean Eq trên 5 ranks × 160 rounds | Max Eq trên 5 ranks × 160 rounds |
|---|---:|---:|
| Delta FP16 AG | 2.0736887e-04 | 2.0992220e-04 |
| Delta FP16 AR | 2.3733849e-04 | 5.1139798e-04 |

Eq là relative L2 ratio của local encode/decode, không phải điểm accuracy hoặc sai số averaged global update. Eq AllReduce còn chịu scaling trước cast; không so trực tiếp Eq của delta với sai số lượng tử hóa whole weights.

Nguồn chi tiết từng method:

- FP32: [train.csv](../results/fp16_sweep_seed44_swTaOOce/ranks/fp32/rank0/train.csv), [log](../results/fp16_sweep_seed44_swTaOOce/fp32.log).
- Whole FP16 AG: [train.csv](../results/fp16_sweep_seed44_swTaOOce/ranks/fp16_whole_allgather/rank0/train.csv), [log](../results/fp16_sweep_seed44_swTaOOce/fp16_whole_allgather.log).
- Whole FP16 AR: [train.csv](../results/fp16_sweep_seed44_swTaOOce/ranks/fp16_whole_allreduce/rank0/train.csv), [log](../results/fp16_sweep_seed44_swTaOOce/fp16_whole_allreduce.log).
- Delta FP16 AG: [train.csv](../results/fp16_sweep_seed44_swTaOOce/ranks/fp16_delta_allgather/rank0/train.csv), [log](../results/fp16_sweep_seed44_swTaOOce/fp16_delta_allgather.log).
- Delta FP16 AR: [train.csv](../results/fp16_sweep_seed44_swTaOOce/ranks/fp16_delta_allreduce/rank0/train.csv), [log](../results/fp16_sweep_seed44_swTaOOce/fp16_delta_allreduce.log).

**Seed43: chi tiết paired experiment có/không EF**

Đây là đợt chạy riêng với control Delta FP16 được chạy lại cùng EF. Không ghép thời gian control của sweep cũ vào phép so này.

| Method | Train loss E40 | Train acc E40 (%) | Val acc E40 (%) | Best val (%) / epoch | Test loss | Test acc (%) | Train (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Delta FP16 AG | 0,97607 | 64,649 | 66,40 | 66,40 / E40 | 0,9503 | 65,86 | 2606,04 |
| Delta FP16 AG + EF | 0,96854 | 65,083 | 66,02 | 66,14 / E39 | 0,9693 | 65,34 | 2521,61 |

| Method | Rank0 compute + optimizer (s) | Rank0 gather wait (s) | Rank0 model sync (s) |
|---|---:|---:|---:|
| Delta FP16 AG | 1614,80 | 369,20 | 402,97 |
| Delta FP16 AG + EF | 1639,50 | 255,62 | 407,38 |

EF thêm khoảng 10,77 MB residual FP32/rank, payload FP16 không đổi. Residual cuối trên ranks0–4 có L2 khoảng 0,001273 / 0,001677 / 0,001460 / 0,001515 / 0,001492; đã kiểm tra hữu hạn và anchor hash đồng nhất. Local Eq của EF dùng corrected input `delta + residual`, không cùng định nghĩa raw-input error với baseline.

Kết luận của paired seed43: test giảm 0,52 điểm phần trăm với EF; train accuracy E40 tăng nhẹ nhưng validation/test thấp hơn. Đây là quan sát, chưa đủ để khẳng định nguyên nhân hoặc nói EF luôn gây overfit. Training time giảm đi cùng gather-wait thấp hơn; không phải bằng chứng EF làm transport nhanh hơn.

**Throughput — kiểm tra ngày 17/10/2026**

Đã đối chiếu 17 full runs với `local_sgd_timeline.csv`, `train.csv` và `summary.json`. Mỗi run có 45.000 training samples/epoch × 40 epochs = **1.800.000 processed samples** trên toàn bộ 5 ranks. Từng sync đã kiểm tra tổng samples bằng tổng samples của 5 ranks. Đây là lượt xử lý samples, không phải số ảnh unique.

Chọn throughput chính cho benchmark là `1.800.000 / training_time_seconds`: gồm validation và overhead bên trong training loop; không gồm startup/preflight và final test. Đây không phải throughput của toàn bộ shell job kể từ launcher.

| Method | Seed42 (ảnh/s) | Seed43 (ảnh/s) | Seed44 (ảnh/s) | Mean ± sample SD (ảnh/s) | Tăng throughput so mean FP32 (%) |
|---|---:|---:|---:|---:|---:|
| FP32 | 687,04 | 673,40 | 679,00 | 679,81 ± 6,85 | +0,00 |
| Whole FP16 AllGather | 721,16 | 711,65 | 735,21 | 722,67 ± 11,86 | +6,30 |
| Whole FP16 AllReduce | 697,22 | 686,67 | 729,69 | 704,53 ± 22,42 | +3,64 |
| Delta FP16 AllGather | 730,50 | 744,55 | 701,65 | 725,56 ± 21,87 | +6,73 |
| Delta FP16 AllReduce | 668,25 | 688,99 | 661,00 | 672,75 ± 14,53 | -1,04 |

Mean trong bảng là trung bình arithmetic của ba throughput cấp run, mỗi seed một giá trị. Không phải trung bình các rates của từng epoch. Phần trăm throughput tăng khác phần trăm thời gian giảm vì throughput là nghịch đảo của thời gian.

**Throughput trong các epoch training, loại thời gian validation**

Tính `1.800.000 / sum(train.csv.train_time)`. Metric này vẫn gồm fetch, local compute/optimizer, chờ và model sync trong các epoch train. Validation/checkpoint ở rank0 có thể ảnh hưởng thời điểm workers sẵn sàng cho epoch kế tiếp, nên đây là phạm vi timer đã ghi, không phải thí nghiệm tắt validation.

| Method | Seed42 (ảnh/s) | Seed43 (ảnh/s) | Seed44 (ảnh/s) |
|---|---:|---:|---:|
| FP32 | 744,27 | 728,79 | 734,94 |
| Whole FP16 AllGather | 785,03 | 771,41 | 801,96 |
| Whole FP16 AllReduce | 755,23 | 742,61 | 794,82 |
| Delta FP16 AllGather | 795,83 | 811,49 | 760,45 |
| Delta FP16 AllReduce | 721,63 | 748,70 | 712,93 |

**Throughput paired EF seed43**

| Method | Training loop (ảnh/s) | Trong epoch train (ảnh/s) | Test accuracy (%) |
|---|---:|---:|---:|
| Delta FP16 AllGather | 690,70 | 747,04 | 65,86 |
| Delta FP16 AllGather + EF | 713,83 | 774,18 | 65,34 |

EF có throughput loop cao hơn 3,35% trong paired run, tương ứng thời gian giảm 3,24%, nhưng test accuracy giảm 0,52 điểm phần trăm. Chênh lệch thời gian đi cùng gather-wait thấp hơn; không quy mức tăng throughput cho việc giảm payload vì cả hai payload bằng nhau.

**Các throughput trong log không có cùng phạm vi đo**

- Dòng `Epoch [...] ... Tput: ... img/s`: code dùng global batch / `step_time`, trong khi `step_time` kết thúc ở `t_compute_end`, trước gather/model sync. Đây là rate quy đổi từ fetch + local compute của rank0; không phải throughput cluster đã tính toàn bộ đồng bộ. Giá trị quanh 1.200 img/s không mâu thuẫn với throughput loop quanh 700 ảnh/s.
- Cột `train.csv.samples_per_sec`: 45.000 / thời gian train của từng epoch. Đã kiểm tra khớp trong sai số làm tròn của CSV. Muốn gộp nhiều epochs, chia tổng samples cho tổng thời gian; không lấy trung bình arithmetic các rates.
- Cột `local_sgd_timeline.csv.round_throughput`: samples của cửa sổ chia elapsed time của cửa sổ. H40 và partial windows cuối epoch khác nhau, nên không lấy trung bình arithmetic các window rates làm throughput cả run.
- Ví dụ FP32 seed42: trung bình trực tiếp epoch rates là 770,09 ảnh/s; gộp đúng timer epoch train là 744,27 ảnh/s; tính toàn training loop là 687,04 ảnh/s.

Các số liệu trên là quan sát cùng recipe đã chạy, chưa isolate riêng lợi ích network khỏi biến động chờ rank và CPU-side overhead. Xem [evidence throughput JSON](fp16_throughput_20261017.json) để lấy số chưa làm tròn và nguồn từng run.

**Đối chiếu throughput với bảng H — 17/10/2026**

Bảng H dùng throughput trong epoch train: `Q_train = 45.000 / mean(train_time)` = `tổng samples / sum(train_time)`. Timer này gồm data fetch, compute/optimizer, chờ và sync MPI, nhưng loại timer validation. Các giá trị đều là lượt ảnh xử lý trên toàn cluster mỗi giây. Vì các epoch có cùng 45.000 samples, dùng mean thời gian là đúng; không lấy mean trực tiếp các throughput epoch.

Bảng training-loop phía trên dùng phạm vi thời gian rộng hơn. Hai metric cần có tên riêng khi đưa vào báo cáo: **throughput train (không gồm validation)** và **throughput training loop (gồm validation)**.

Cả 12 throughput trong bảng H người dùng cung cấp khớp dữ liệu gốc khi làm tròn một chữ số thập phân:

| Method | Epochs | Avg train/epoch (s) | Throughput train (ảnh/s) | Throughput loop (ảnh/s) | Total loop (phút) |
|---|---:|---:|---:|---:|---:|
| AllReduce | 20 | 291,98 | 154,12 | 151,50 | 99,01 |
| H=4 | 20 | 113,26 | 397,31 | 380,18 | 39,46 |
| H=8 | 20 | 78,15 | 575,80 | 540,03 | 27,78 |
| H=16 | 20 | 71,98 | 625,19 | 583,18 | 25,72 |
| H=24 | 20 | 65,62 | 685,81 | 635,87 | 23,59 |
| H=32 | 20 | 59,84 | 751,98 | 691,59 | 21,69 |
| H=40 | 20 | 64,43 | 698,48 | 642,41 | 23,35 |
| H=24 | 40 | 64,82 | 694,21 | 642,91 | 46,66 |
| H=40 | 40 | 65,59 | 686,03 | 636,94 | 47,10 |
| H=80 | 40 | 54,89 | 819,84 | 751,14 | 39,94 |
| H=120 | 40 | 57,32 | 785,04 | 722,25 | 41,54 |
| H=150 | 40 | 46,71 | 963,48 | 869,80 | 34,49 |

**FP16: throughput train theo cùng công thức của bảng H**

| Method | Seed42 (ảnh/s) | Seed43 (ảnh/s) | Seed44 (ảnh/s) | Mean ± sample SD (ảnh/s) | So mean FP32 (%) |
|---|---:|---:|---:|---:|---:|
| FP32 | 744,27 | 728,79 | 734,94 | 736,00 ± 7,79 | +0,00 |
| Whole FP16 AllGather | 785,03 | 771,41 | 801,96 | 786,13 ± 15,31 | +6,81 |
| Whole FP16 AllReduce | 755,23 | 742,61 | 794,82 | 764,22 ± 27,24 | +3,83 |
| Delta FP16 AllGather | 795,83 | 811,49 | 760,45 | 789,26 ± 26,15 | +7,24 |
| Delta FP16 AllReduce | 721,63 | 748,70 | 712,93 | 727,75 ± 18,65 | -1,12 |

| Method | Avg train/epoch seed42 (s) | Seed43 (s) | Seed44 (s) | Mean (s) |
|---|---:|---:|---:|---:|
| FP32 | 60,46 | 61,75 | 61,23 | 61,15 |
| Whole FP16 AllGather | 57,32 | 58,34 | 56,11 | 57,26 |
| Whole FP16 AllReduce | 59,58 | 60,60 | 56,62 | 58,93 |
| Delta FP16 AllGather | 56,54 | 55,45 | 59,18 | 57,06 |
| Delta FP16 AllReduce | 62,36 | 60,10 | 63,12 | 61,86 |

Mean ± SD ở đây tính trên ba throughput cấp run, mỗi seed một giá trị. So mean FP32 là tỷ lệ hai mean throughput; không phải tỷ lệ giảm thời gian.

H40 40e trong bảng cũ là run `local_sgd_h40_300_40e_20261007_105137`: throughput train **686,03 ảnh/s**, total loop **47,10 phút**. FP32 seed42 trong sweep FP16 là run khác `fp16_sweep_uydYAoMC/fp32`: throughput train **744,27 ảnh/s**, total loop **43,67 phút**. Giá trị **687,04 ảnh/s** của run sau là throughput loop, gần 686,03 về số nhưng khác phạm vi timer. Không dùng hai con số này làm cùng một baseline.

Theo metric train, Delta AllGather trung bình chỉ hơn Whole AllGather khoảng 0,40%; chưa đủ để khẳng định delta đem lại lợi ích throughput ổn định. EF paired seed43 theo cùng metric là 747,04 → 774,18 ảnh/s; đây vẫn chỉ là một paired run và có biến động gather-wait.

Nguồn từng run và số chưa làm tròn: [throughput comparison audit](throughput_comparison_audit_20261017.json), [FP16 per-run evidence](fp16_throughput_20261017.json).
