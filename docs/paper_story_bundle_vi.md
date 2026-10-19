# Câu chuyện paper hiện tại: hiệu quả và giới hạn của FP16 communication trên cụm CPU/GPU dị thể

Đây là bản diễn giải dựa trên log hiện có, phục vụ viết paper và trao đổi kết quả. Đây chưa phải tuyên bố một thuật toán mới đã vượt các baseline về time-to-accuracy. File này đi cùng raw logs, catalog và evidence trong gói ZIP.

## 1. Câu hỏi nghiên cứu

**Trên cụm commodity CPU/GPU dị thể và mạng Ethernet 100 Mbps, giảm communication bằng Local SGD và FP16 giúp huấn luyện nhanh hơn đến mức nào, và điều gì giới hạn lợi ích khi xét đồng thời thời gian và chất lượng model?**

Baseline hệ thống ban đầu là gradient AllReduce mỗi bước, đã dùng allocation không đều. Baseline trực tiếp của thí nghiệm precision là **FP32 Local SGD H40**. Cần phân biệt hai baseline này: một cơ chế có thể giảm mạnh chi phí so với AllReduce mỗi bước nhưng chỉ cải thiện nhỏ so với H40 đã giảm tần suất sync.

## 2. Bối cảnh và recipe

- ViT-Tiny trên CIFAR-10, 5 máy; rank0 dùng hai GPU Titan Z, ranks1–4 dùng CPU.
- Allocation [266, 6, 6, 6, 16], tổng batch 300. Đây không phải phép chia đều batch.
- Full FP16 comparisons: 40 epochs, H40, 150 local optimizer steps/epoch, sample-weighted model averaging, AdamW, LR cố định 0,001; optimizer state giữ local.
- 160 syncs/rank, gồm các cửa sổ local steps 40/40/40/30 mỗi epoch; 1.800.000 processed samples/run trên cả cluster.
- Training và optimizer giữ FP32. FP16 trong câu chuyện này là precision của communication, không phải mixed-precision compute trên GPU.
- Mỗi method có một training trajectory riêng. Test accuracy đo sau khi nạp checkpoint tốt nhất theo validation.

Bộ thống kê chính là một sweep đủ năm methods trên mỗi seed42/43/44, tổng 15 full runs. Paired EF seed43 có hai full runs riêng. Các lần chạy lại cùng seed, smoke và diagnostic không được dùng để tăng số seeds.

## 3. Quan sát thứ nhất: giảm số lần sync xử lý chi phí lớn nhất của baseline ban đầu

Ở baseline AllReduce 20e, rank0 có 3.000 steps; elapsed MPI AllReduce là **4.820,39/5.831,44 giây**, tương đương **82,66% tổng step**. Microbenchmark khoảng 1,6 giây/collective và bandwidth payload lớn khoảng 11,2 MB/s phù hợp với mạng 100 Mbps.

Local SGD H40 chỉ sync bốn lần/epoch. Các runs cũ 20e ghi total loop **99,01 phút** cho AllReduce mỗi bước và **23,35 phút** cho H40, tức nhanh hơn khoảng **4,24 lần ở cùng số epochs**. Không dùng con số này làm speedup tới cùng accuracy: Local SGD thay đổi cả quy trình tối ưu và hai runs cũ chưa phải đối chứng nhiều seeds cho time-to-target.

Luận điểm hệ thống có thể trình bày: **trên mạng này, tránh trao đổi model/gradient sau mọi bước mang lại lợi ích lớn hơn so với chỉ thay đổi định dạng số của một lần truyền.**

## 4. Quan sát thứ hai: sau H40, thời gian còn lại không thuần do truyền bytes

FP32 H40, trung bình ba seeds, xét theo toàn bộ training-loop timer rank0:

| Thành phần | Tổng trung bình (s) | Tỷ trọng loop |
|---|---:|---:|
| Compute + optimizer | 1.591,07 | 60,09% |
| Gather sample counts trước model sync | 540,72 | 20,42% |
| Toàn bộ model sync | 291,24 | 11,00% |

Phần còn lại gồm fetch, validation và các overhead khác. Gather ở đây không phải dedicated barrier, và elapsed MPI không phải wire time thuần.

Từ 85 trace từng rank của 17 full runs, đã xác nhận đủ số steps/syncs. Trong ba runs FP32, bỏ epoch khởi động, CPU ranks2/3 có tổng `data + compute` lớn nhất trong **468/468 cửa sổ**. Allocation hiện tại vẫn lệch thời gian local work. Tuy nhiên, chưa có timeline readiness căn chỉnh giữa máy để khẳng định hai ranks đó gây ra toàn bộ gather-wait hoặc bỏ chúng sẽ nhanh hơn.

Một cảnh báo quan trọng khi đọc log: worker có thể gửi sample counts về root rồi chờ ở collective kế tiếp. Rank4 có local compute thấp hơn ranks2/3 nhưng elapsed payload MPI cao nhất. Không suy ra worker có MPI time cao nhất là compute straggler, và không cộng maxima của các ranks khác nhau thành critical path.

Luận điểm có bằng chứng: **cần xem toàn bộ quy trình sync và độ lệch giữa các ranks, thay vì chỉ tối ưu một cột payload MPI.**

## 5. Quan sát thứ ba: FP16 AllGather có lợi về thời gian 40 epochs, delta chưa có lợi ích bổ sung rõ

| Method | Test accuracy TB ± sample SD (%) | Mean loop (phút) | Thay đổi thời gian so FP32 H40 |
|---|---:|---:|---:|
| FP32 | 66,95 ± 2,17 | 44,13 | Baseline |
| Whole FP16 AllGather | 66,30 ± 1,29 | 41,52 | Giảm 5,92% |
| Whole FP16 AllReduce | 65,95 ± 0,66 | 42,61 | Giảm 3,45% |
| Delta FP16 AllGather | 65,94 ± 0,32 | 41,37 | Giảm 6,25% |
| Delta FP16 AllReduce | 66,26 ± 0,50 | 44,61 | Tăng 1,07% |

Mean time tính từ số giây chưa làm tròn. Payload FP32 là 10.774.312 bytes/rank/sync, FP16 là 5.387.156 bytes/rank/sync. Đây là kích thước một buffer; AllGather và AllReduce không có cùng tổng network traffic chỉ vì cùng buffer bytes.

Delta AllGather có mean loop thấp nhất trong năm methods, nhưng so với Whole AllGather chỉ giảm **8,85 giây, khoảng 0,36%**. Mean test của Delta AllGather thấp hơn Whole AllGather khoảng **0,36 điểm phần trăm**; chênh lệch theo seed là +0,69/−1,69/−0,09 điểm phần trăm. Chưa có cơ sở nhận delta thắng Whole FP16 về tốc độ hoặc accuracy một cách nhất quán.

Mặc dù Delta AllReduce có payload MPI rank0 ngắn hơn Delta AllGather, gather trước sync dài hơn và total loop chậm hơn. Transport/checks/host overhead và rank waiting cần được đo như một quy trình hoàn chỉnh. FP32 AllReduce so FP16 AllGather còn thay cả precision lẫn collective; phép so này chưa isolate riêng tác dụng quantizer.

## 6. Sai số cập nhật tốt hơn chưa đủ để bảo đảm accuracy tốt hơn

Trong same-input diagnostic seed43, mean global error chuẩn hóa theo FP32 global update là **0,48749% với Whole FP16 AllGather** và **0,02045% với Delta FP16 AllGather**, thấp hơn khoảng **23,84 lần**. Đây là 360 steps/10 rounds, trajectory thực FP32; không dùng accuracy/timing của diagnostic để kết luận shadow method tốt hơn trong full training.

Kết quả full runs chưa cho cải thiện accuracy nhất quán của delta. Điều này làm rõ khoảng cách giữa **độ chính xác số học của phép trao đổi** và **chất lượng cuối của trajectory huấn luyện**; chưa xác định được nguyên nhân của khoảng cách bằng một đối chứng riêng.

EF trên paired seed43 cũng chưa cho lợi ích chất lượng:

- Test: **65,86% → 65,34%**, giảm 0,52 điểm phần trăm.
- Training: **2.606,04 → 2.521,61 giây**, giảm 84,43 giây.
- Gather rank0 giảm 113,58 giây; compute tăng 24,70 giây và model sync tăng 4,41 giây.
- Payload không đổi; thêm residual FP32 10,77 MB/rank.

Không quy chênh lệch thời gian này cho EF làm transport nhanh hơn. Một paired seed cũng chưa đủ kết luận EF luôn kém. Kết quả phù hợp cách báo cáo một negative/mixed result thay vì nhận EF là cải tiến đã chứng minh.

## 7. Đoạn kể ngắn có thể dùng trong phần kết quả

> Thực nghiệm trên cụm năm máy CPU/GPU dị thể nối Ethernet 100 Mbps cho thấy đồng bộ mỗi bước gây chi phí lớn. Periodic model averaging giảm mạnh số lần trao đổi và thời gian hoàn thành cùng số epochs. Sau bước này, local compute và chờ giữa các ranks chiếm phần đáng kể của runtime. Các biến thể FP16 AllGather giảm khoảng 6% thời gian training 40 epochs so với FP32 Local SGD H40 trong ba seeds, với tradeoff chất lượng được báo cáo đầy đủ. Delta representation giảm sai số cập nhật trên cùng inputs nhưng chưa đem lại lợi ích accuracy nhất quán so với Whole FP16; error feedback cũng chưa cải thiện quality trong paired seed43. Kết quả cho thấy payload size và local quantization error riêng lẻ chưa đủ dự đoán hiệu quả huấn luyện tổng thể trên cụm dị thể.

## 8. Có thể claim và chưa thể claim

**Có thể claim theo phạm vi dữ liệu:**

1. AllReduce mỗi bước có elapsed communication lớn trong baseline cụ thể đã đo; Local SGD giảm mạnh fixed-budget training time trong các runs được nêu.
2. FP16 giảm 50% buffer payload; FP16 AllGather có mean 40e loop time thấp hơn FP32 H40 khoảng 6% trong ba seeds đã chạy.
3. Rank2/3 có local work dài nhất sau khởi động trong các cửa sổ đã kiểm tra.
4. Delta giảm error trong diagnostic, nhưng lợi ích cuối về quality/speed chưa nhất quán; EF seed43 là một kết quả chưa ủng hộ giả thuyết cải thiện quality.

**Chưa có cơ sở claim:**

- Delta/EF là thuật toán mới hoặc SOTA; Local SGD, delta quantization và EF đều có tiền lệ.
- Giảm thời gian 40e 6,25% đồng nghĩa giảm time-to-accuracy 6,25%, hoặc FP16 giữ chất lượng tương đương FP32 theo nghĩa thống kê.
- Sample SD nhỏ hơn trên ba seeds chứng minh ổn định hơn nói chung.
- Mọi gather-wait do CPU ranks2/3, mọi MPI elapsed là wire time, hoặc bỏ CPU chắc chắn giúp nhanh hơn.
- Sai số quantization là nguyên nhân chính giới hạn accuracy; current data cũng chưa phân biệt drift, horizon/LR và các yếu tố optimization khác.
- Tài liệu profile/benchmark lịch sử dùng cùng evaluation/recipe với toàn bộ 17 runs chính.

## 9. Khoảng trống để câu chuyện thành một paper về cơ chế cải tiến

Hiện kết quả đủ cho một câu chuyện phân tích thực nghiệm về **tradeoff communication/runtime/quality**. Để nhận một cơ chế mới tốt hơn baseline, cần thêm:

1. Timeline từng round có thể đối chiếu giữa các ranks; tách local work, readiness, get/set weights, codec, guards/norm/hash, scalar/payload collectives và telemetry I/O.
2. Một giả thuyết cơ chế được kiểm tra bằng ablation, giữ recipe và ngân sách cố định. Ví dụ kiểm tra allocation giảm readiness spread nhưng giữ H/communication mode như control; đây là hướng kiểm nghiệm, không phải cải tiến đã đạt.
3. Time-to-validation-target đo trực tiếp từ training-loop start; target đặt trước, giữ các runs chưa đạt trong báo cáo. Không chọn threshold/seed vì thuận lợi sau khi nhìn kết quả.
4. Paired seeds và uncertainty cho finalists; tuning bằng validation, test dành cho đánh giá cuối.

Gói ZIP chứa cả các runs lịch sử để người đọc tự kiểm tra phạm vi. Các runs ngoài bộ 17 đã audit không được tự gộp thành thêm seeds hay bằng chứng cùng recipe.

## 10. Nguồn trong ZIP

- `04_PRIMARY_17_RUNS.csv`: đường dẫn log/train.csv và số liệu từng run chính.
- `workspace/HeteroViT-MPI/docs/bottleneck_story_20261018_vi.md`: audit chi tiết.
- `workspace/HeteroViT-MPI/docs/bottleneck_audit_20261018.json`: evidence tính từ 17 runs/85 traces.
- `workspace/HeteroViT-MPI/docs/fp16_results_by_seed_20261015_vi.md`: bảng từng seed, throughput và nguồn.
- `workspace/HeteroViT-MPI/profiles/`: raw compute/network/subset profiles; đọc cùng network audit.
- `workspace/HeteroViT-MPI/results/fp16_sweep_seed43_E2dtk5hD/diagnostic_summary.json`: diagnostic Whole/Delta.
- `workspace/HeteroViT-MPI/results/fp16_sweep_seed43_2tyCNQ2H/`: paired EF/control.
- `03_RUN_CATALOG.csv`, `05_TRAIN_CSV_LOCATIONS.csv`: phân biệt canonical runs, worker telemetry và bản sao.
- `source_snapshots/` và `09_SOURCE_SNAPSHOTS.csv`: source theo các sweep, giữ cả những phiên bản lịch sử.

Các nguồn trên hỗ trợ số liệu thực nghiệm của dự án. Trước khi nộp paper, cần bổ sung related work và xác minh citation theo phạm vi đóng góp cuối cùng; file này không thay thế một literature review.
