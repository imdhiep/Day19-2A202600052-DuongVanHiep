# Reflection - Lab 19

**Tên:** Dương Văn Hiệp - 2A202600052
**Cohort:** 2A202600052
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

Trên golden set 50 queries, **BM25 thắng nhóm `exact`** vì query chứa đúng technical term verbatim như `Kubernetes`, `OAuth`, `Terraform`; lexical match rất mạnh nên sparse retrieval đủ tốt. **Vector mạnh hơn ở các query giàu ý nghĩa ngữ cảnh**, nhất là khi người dùng diễn đạt gần nghĩa thay vì lặp nguyên keyword, nhưng với model lite `bge-small-en` thì tiếng Việt paraphrase vẫn chưa thật xuất sắc. **Hybrid thắng rõ nhất ở `mixed`** vì gom được cả tín hiệu exact term lẫn semantic similarity, nên trung bình toàn bộ vẫn cao nhất.

Tôi **không dùng hybrid** khi bài toán cần tối giản latency hoặc query gần như 100% exact identifier/keyword, ví dụ tra API name, error code, table name, hoặc log signature; lúc đó pure BM25 rẻ hơn và đủ chính xác. Ngược lại, nếu dữ liệu ngắn, đồng nghĩa nhiều, hoặc user hay paraphrase bằng tiếng Việt, pure vector là lựa chọn hợp lý hơn sparse-only.

---

## Điều ngạc nhiên nhất khi làm lab này

Điểm bất ngờ nhất là chất lượng hybrid phụ thuộc rất mạnh vào embedding model và cách warm-up hệ thống; code đúng công thức RRF thôi chưa đủ nếu môi trường runtime chưa ổn.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [x] Pair work với:
