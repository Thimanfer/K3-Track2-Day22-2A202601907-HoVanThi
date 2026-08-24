# BÁO CÁO THỰC NGHIỆM BÀI LAB DAY 22
## Giám sát Pipeline RAG với LangSmith, Quản lý Phiên bản Prompt Hub, Đánh giá Định lượng RAGAS và Bảo mật Guardrails AI

- **Họ và tên học viên:** Hồ Văn Thi
- **Mã số học viên:** 2A202601907
- **Môn học / Khóa đào tạo:** K3-Track 2 — LLM & RAG Application Engineering
- **Ngày thực hiện:** 24/08/2026

---

## 1. Mục tiêu và Tổng quan Bài Lab

Trong bài thực hành này, em hướng tới việc xây dựng một hệ thống RAG (Retrieval-Augmented Generation) hoàn chỉnh ở cấp độ production-ready, tập trung vào 4 khía cạnh cốt lõi của kỹ thuật LLMOps:
1. **Khả năng quan sát (Observability):** Tích hợp LangSmith để theo dõi từng span/node, đo lường độ trễ (latency), lượng token tiêu thụ và bắt lỗi pipeline theo thời gian thực.
2. **Quản lý vòng đời Prompt (Prompt Lifecycle & Hub):** Phiên bản hóa Prompt trên LangSmith Hub (`hovanthi-rag-prompt-v1` và `hovanthi-rag-prompt-v2`), kết hợp thuật toán phân phối lưu lượng (A/B testing) tất định.
3. **Đánh giá chất lượng định lượng (RAG Evaluation with RAGAS):** Đánh giá khách quan trên bộ 50 cặp QA pairs với 4 tiêu chuẩn công nghiệp (`faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`).
4. **Bảo mật và Kiểm soát chất lượng dữ liệu (Guardrails AI):** Xây dựng các Custom Validator giúp tự động ẩn danh thông tin cá nhân nhạy cảm (PII) và chuẩn hóa cú pháp JSON đầu ra.

---

## 2. Bảng Tổng hợp Minh chứng Nộp bài (Deliverables Matrix)

| Tên tệp minh chứng | Định dạng | Mô tả chi tiết nội dung minh chứng | Trạng thái |
|---|:---:|---|:---:|
| [`01_langsmith_traces.png`](file:///c:/Users/HP/OneDrive/Desktop/lab_Track%202/K3-Track2-Day22-2A202601907-HoVanThi/evidence/01_langsmith_traces.png) | Image (PNG) | Ảnh chụp giao diện LangSmith Dashboard hiển thị danh sách 50 traces truy vấn của project `day22-lab`. | ✅ Hoàn thành |
| [`02_prompt_hub.png`](file:///c:/Users/HP/OneDrive/Desktop/lab_Track%202/K3-Track2-Day22-2A202601907-HoVanThi/evidence/02_prompt_hub.png) | Image (PNG) | Ảnh chụp LangSmith Prompt Hub chứa 2 phiên bản `hovanthi-rag-prompt-v1` và `hovanthi-rag-prompt-v2`. | ✅ Hoàn thành |
| [`02_ab_routing_log.txt`](file:///c:/Users/HP/OneDrive/Desktop/lab_Track%202/K3-Track2-Day22-2A202601907-HoVanThi/evidence/02_ab_routing_log.txt) | Text Log | Console log của quá trình định tuyến tất định (Deterministic Routing) trên 50 requests. | ✅ Hoàn thành |
| [`03_ragas_scores.png`](file:///c:/Users/HP/OneDrive/Desktop/lab_Track%202/K3-Track2-Day22-2A202601907-HoVanThi/evidence/03_ragas_scores.png) | Image (PNG) | Ảnh chụp màn hình terminal in bảng so sánh 4 chỉ số RAGAS giữa 2 phiên bản Prompt. | ✅ Hoàn thành |
| [`03_ragas_report.json`](file:///c:/Users/HP/OneDrive/Desktop/lab_Track%202/K3-Track2-Day22-2A202601907-HoVanThi/evidence/03_ragas_report.json) | JSON Data | Dữ liệu xuất chi tiết điểm số RAGAS của 50 QA pairs và kết quả đạt chuẩn điểm thưởng. | ✅ Hoàn thành |
| [`04_pii_demo_log.txt`](file:///c:/Users/HP/OneDrive/Desktop/lab_Track%202/K3-Track2-Day22-2A202601907-HoVanThi/evidence/04_pii_demo_log.txt) | Text Log | Console log kiểm thử PII Detector khử 4 loại dữ liệu nhạy cảm (Email, Phone, SSN, Credit Card). | ✅ Hoàn thành |
| [`04_json_demo_log.txt`](file:///c:/Users/HP/OneDrive/Desktop/lab_Track%202/K3-Track2-Day22-2A202601907-HoVanThi/evidence/04_json_demo_log.txt) | Text Log | Console log kiểm thử JSON Formatter tự sửa lỗi cú pháp phổ biến từ LLM. | ✅ Hoàn thành |

---

## 3. Chi tiết Quá trình Triển khai & Phân tích Kỹ thuật

### Bước 1 — Xây dựng RAG Pipeline & Tích hợp LangSmith Tracing
- **Thiết kế Pipeline:** Em sử dụng LangChain Expression Language (LCEL) để kết nối FAISS VectorStore (`k=3`), ChatPromptTemplate và Google Gemini Model (`gemini-3.1-flash-lite`).
- **Tracing Decorator:** Áp dụng decorator `@traceable(name="rag-query", tags=["rag", "step1"])` ở cấp độ chain và function, giúp ghi nhận đầy đủ tree trace (gồm retriever span, prompt formatting, LLM call, token usage, latency và exception handling).
- **Kết quả thực nghiệm:** 50/50 câu hỏi mẫu (`SAMPLE_QUESTIONS`) đã được chạy thành công và đẩy dữ liệu lên LangSmith project `day22-lab` mà không bị gián đoạn hay mất mát trace.

### Bước 2 — Thiết kế Prompt Hub & Thuật toán A/B Testing Tất định
- **Prompt V1 (Ngắn gọn, Trực diện):** Hướng dẫn mô hình đóng vai trò trợ lý thân thiện, trả lời cô đọng trong 2-4 câu, đi thẳng vào câu trả lời từ context.
- **Prompt V2 (Chuyên gia Phân tích, Có cấu trúc):** Yêu cầu mô hình đóng vai Senior AI Technical Analyst, cấu trúc câu trả lời bắt buộc thành 3 phần rõ ràng:
  1. *Tổng quan câu trả lời*
  2. *Chi tiết & Dẫn chứng dữ liệu từ Context*
  3. *Lưu ý kỹ thuật & Mức độ tin cậy*
- **Thuật toán Định tuyến Tất định (Deterministic Hash Routing):**
  - Để đảm bảo cùng một `request_id` (hoặc `user_id`) luôn luôn được nhận cùng một phiên bản prompt mà không cần lưu state phức tạp vào cơ sở dữ liệu, em sử dụng mã băm MD5:
  $$\text{bucket} = \text{int}(\text{MD5}(\text{request\_id}).\text{hexdigest}(), 16) \pmod{100}$$
  - Phân vùng: $\text{bucket} < 50 \implies \text{V1}$, ngược lại $\implies \text{V2}$.
  - **Kết quả phân phối thực tế (50 câu hỏi):**
    - **Prompt V1:** 19 câu (~38.0%)
    - **Prompt V2:** 31 câu (~62.0%)
    - Tỷ lệ phân phối ổn định và tuân thủ nguyên tắc tất định tuyệt đối.

### Bước 3 — Đánh giá Định lượng với RAGAS (50 Cặp QA Pairs)
Em đã thực hiện đánh giá độc lập 50 câu hỏi trên cả 2 phiên bản prompt với ground-truth reference để đo lường 4 chỉ số cốt lõi:

```text
====================================================================
         BẢNG SO SÁNH KẾT QUẢ ĐÁNH GIÁ RAGAS (50 QA PAIRS)        
  Học viên: Hồ Văn Thi | MSSV: 2A202601907
====================================================================
  Chỉ số đánh giá (RAGAS Metric)          V1        V2     So sánh
====================================================================
  faithfulness                        0.8986    0.9016  ← V2 thắng (⭐ Đạt chuẩn điểm thưởng ≥ 0.90)
  answer_relevancy                    0.9271    0.8980  ← V1 thắng
  context_recall                      0.9780    0.9780    Hòa nhau
  context_precision                   0.9904    0.9904    Hòa nhau
====================================================================
```

#### Phân tích & Đánh giá Chuyên sâu:
1. **Về chỉ số Faithfulness (Độ trung thực):**
   - Prompt V2 đạt **0.9016** (vượt ngưỡng điểm thưởng 0.90), cao hơn V1 (**0.8986**).
   - *Lý do:* Ràng buộc cấu trúc 3 phần và chỉ thị cấm suy đoán của V2 buộc mô hình phải bám sát từng sự kiện trong Context, hạn chế tối đa hiện tượng ảo giác (hallucination).
2. **Về chỉ số Answer Relevancy (Độ phù hợp của câu trả lời):**
   - Prompt V1 đạt **0.9271**, cao hơn V2 (**0.8980**).
   - *Lý do:* Prompt V1 trả lời ngắn gọn, trực diện, không chèn các tiêu đề mục hay các câu giải thích phụ, do đó độ tương đồng ngữ nghĩa (embedding cosine similarity) giữa câu hỏi và câu trả lời tập trung cao hơn.
3. **Về Context Recall (0.9780) & Context Precision (0.9904):**
   - Hai chỉ số này đạt mức tiệm cận tuyệt đối ở cả 2 phiên bản vì sử dụng chung cơ chế retrieval FAISS index với text chunking tối ưu (`chunk_size=500, chunk_overlap=50, k=3`).

### Bước 4 — Tầng Bảo mật & Chuẩn hóa Định dạng với Guardrails AI
Em xây dựng 2 Custom Validator kế thừa từ `Validator` của Guardrails AI:
1. **`PIIDetector` (`custom/pii-detector`):**
   - Sử dụng Regex Engine nhận diện 4 định dạng PII: Email, Số điện thoại (chuẩn Việt Nam & Quốc tế), Số an sinh xã hội (SSN), Số thẻ tín dụng (Credit Card).
   - Áp dụng chiến lược `OnFailAction.FIX` để tự động thay thế dữ liệu nhạy cảm bằng nhãn `[TYPE_REDACTED]`, đảm bảo dữ liệu người dùng không bị rò rỉ ra bên ngoài.
2. **`JSONFormatter` (`custom/json-formatter`):**
   - Xử lý các lỗi cú pháp kinh điển của LLM: lọc bỏ markdown code fences (` ```json `), chuyển single quotes sang double quotes, loại bỏ dấu phẩy thừa ở cuối object/array (trailing commas).
   - Khi đầu vào hoàn toàn không phải JSON, validator tự động bọc chuỗi vào object `{"response": "..."}` hợp lệ để ứng dụng backend không bị crash `JSONDecodeError`.

---

## 4. Những Thách thức Kỹ thuật và Giải pháp Khắc phục

Trong quá trình thực nghiệm, em đã đối mặt và giải quyết các bài toán kỹ thuật thực tế sau:

1. **Vấn đề Rate Limit & Quota trên Free Tier (HTTP 429 Resource Exhausted):**
   - *Hiện tượng:* Khi embedding đồng loạt 107 text chunks hoặc gửi liên tục 50 queries trong 1 phút, Gemini API bị chặn bởi rate-limit 15 req/min.
   - *Giải pháp:* Em đã triển khai:
     - **FAISS Local Caching:** Lưu vectorstore xuống đĩa (`data/faiss_index`) sau lần build đầu tiên, những lần chạy sau chỉ tải từ cache trong vài mili-giây.
     - **Pacing & Exponential Backoff Retry:** Thêm độ trễ điều tiết (`time.sleep(3.5)`) giữa các truy vấn kết hợp vòng lặp retry tự động chờ khi gặp mã 429.
2. **Tương thích Model & Môi trường Windows UTF-8:**
   - Cấu hình chuẩn `sys.stdout.reconfigure(encoding="utf-8")` trên toàn bộ các entrypoint scripts để ngăn chặn lỗi `UnicodeEncodeError` trên PowerShell/cmd Windows.
   - Chuyển sang model thế hệ mới `gemini-3.1-flash-lite` và `models/gemini-embedding-001` tương thích hoàn hảo với LangChain 0.3.x.

---

## 5. Kết luận

Qua bài lab Day 22, em đã nắm vững và làm chủ toàn bộ quy trình xây dựng, kiểm thử, giám sát và bảo mật ứng dụng LLM trong thực tế. Toàn bộ code đã được tổ chức khoa học, chạy kiểm thử thành công 100%, tạo đầy đủ 7 file bằng chứng cùng báo cáo JSON đạt điểm số tối đa.
