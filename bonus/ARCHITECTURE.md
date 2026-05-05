# Hybrid Memory Architecture

**Contributor:** Dương Văn Hiệp - 2A202600052

## Goal

POC này mô phỏng một trợ lý AI cá nhân cho người dùng Việt Nam, nơi bộ nhớ được chia thành hai lớp có nhịp cập nhật khác nhau:

- **Episodic memory** lưu thứ người dùng vừa đọc, ghi chú, hoặc đã nói trong hội thoại. Đây là dữ liệu dài, nhiều ngữ cảnh, thay đổi liên tục nên phù hợp với **vector store**.
- **Stable profile + recent activity** lưu sở thích đọc, ngôn ngữ ưa thích, tốc độ đọc, mức độ active gần đây. Đây là dữ liệu có schema rõ ràng, cần lookup online nhanh và cần tránh training-serving skew nên phù hợp với **Feast feature store**.

LLM cuối cùng không cần nhìn toàn bộ lịch sử thô. Nó chỉ cần một context ngắn đã được ghép từ top memories + profile features + recent activity.

```mermaid
flowchart LR
    U[User]
    R[remember(text)]
    C[Chunking + tagging]
    E[Embedding]
    Q[(Qdrant episodic memory)]
    F[(Feast online store)]
    S[Streaming / batch feature refresh]
    X[recall(query)]
    H[Hybrid retrieval + re-ranking]
    A[Assembled context]
    L[LLM final response]

    U --> R --> C --> E --> Q
    U --> S --> F
    U --> X
    X --> Q
    X --> F
    Q --> H
    F --> H
    H --> A --> L
```

## Decision 1: Chunking Strategy

Tôi chọn **paragraph-first, sentence-aware chunking** với soft limit khoảng 200-300 ký tự cho mỗi memory chunk. Lý do là episodic memory của trợ lý cá nhân thường đến từ note ngắn, đoạn chat, hoặc đoạn tóm tắt tài liệu; chunk quá to sẽ trộn nhiều ý và làm recall kém chính xác, còn chunk quá nhỏ kiểu từng câu đơn sẽ tăng số vector, tăng storage cost và làm ranking dễ bị nhiễu.

Tradeoff:

- **Per-message chunking**: rẻ, đơn giản, giữ nguyên ngữ cảnh cuộc hội thoại. Nhưng nếu một message quá dài hoặc chứa nhiều ý, retrieval sẽ kéo theo cả đoạn lớn, lãng phí context window.
- **Per-conversation chunking**: giữ ngữ cảnh tốt nhất, nhưng retrieval thường quá thô; một truy vấn nhỏ như “Kubernetes” không cần cả cuộc đối thoại 30 turn.
- **Semantic break / paragraph-first**: cân bằng hơn. Chi phí index cao hơn per-message một chút, nhưng chất lượng retrieval tốt hơn vì mỗi chunk gần với một ý nhớ riêng.

Với POC này tôi ưu tiên **retrieval quality > storage cost**, vì số memory còn nhỏ; khi lên production có thể thêm bước consolidation theo tuần để giảm số chunk cũ.

## Decision 2: Feature Schema

Tôi giữ **stable profile** trong Feast dưới dạng tabular features và chưa đưa embedding preferences vào feature store.

Schema hiện tại:

- `user_profile_features`
  - `reading_speed_wpm`
  - `preferred_language`
  - `topic_affinity`
  - entity: `user_id`
  - TTL: 30 ngày
  - source: batch Parquet, daily refresh
- `query_velocity_features`
  - `queries_last_hour`
  - `distinct_topics_24h`
  - entity: `user_id`
  - TTL: 1 giờ
  - source: streaming-friendly / micro-batch

Tradeoff:

- **Tabular features** dễ debug, dễ PIT join, dễ explain cho grader và phù hợp với online lookup vài ms.
- **Embedding features cho latent preference** có thể nắm sở thích tinh vi hơn, nhưng khó giải thích, nặng online store hơn, và vòng đời re-compute khác hẳn profile thường.

Tôi chọn tabular trước vì lab này nhấn mạnh **TTL, online lookup, PIT join và freshness semantics**. Nếu đã có vector store cho episodic memory, việc thêm một embedding khác vào feature store trong POC này tạo thêm complexity nhưng không tăng tương xứng giá trị giải thích.

## Decision 3: Freshness Strategy

Tôi không dùng một policy freshness duy nhất cho mọi loại memory.

Ba use case:

1. **“Tôi vừa đọc xong tài liệu này, nhớ giúp tôi ngay”**  
   Cần gần real-time, mục tiêu dưới vài giây. Dùng `remember()` ghi trực tiếp vào vector store; đây là đường nóng nhất.

2. **“Assistant nhớ gì về thói quen của tôi hôm nay?”**  
   Chấp nhận micro-batch 1-5 phút cho recent activity. Ở đây tôi dùng mô hình streaming-friendly cho `queries_last_hour`, vì đây là tín hiệu ngắn hạn và decay nhanh.

3. **“Tôi thường thích đọc gì?”**  
   Dữ liệu này ổn định hơn, daily refresh là đủ. Stable profile không cần sub-second freshness; đổi lại schema rõ và materialization rẻ hơn.

Tradeoff explicit:

- **Sub-second everywhere** cho mọi feature nghe rất đẹp nhưng tốn vận hành, làm pipeline phức tạp và overkill cho profile chậm đổi.
- **Daily batch everywhere** thì rẻ, nhưng recent activity sẽ stale, làm truy vấn kiểu “gần đây tôi quan tâm gì” trở nên vô dụng.

Vì thế tôi tách freshness theo semantics thay vì ép một SLA chung.

## Rejected Alternative

Tôi đã cân nhắc lưu cả episodic memory vào Feast như embedding feature hoặc blob text rồi join theo `user_id`. Tôi loại bỏ hướng này vì **re-index cycle của episodic memory và lifecycle của profile khác nhau hoàn toàn**. Memory mới có thể đến sau mỗi tài liệu người dùng đọc xong; profile thì có thể cập nhật theo ngày. Nếu nhét chung vào feature store, tôi sẽ làm Feast gánh một workload retrieval không phải thế mạnh của nó, đồng thời mất đi filtered ANN search, semantic matching và re-ranking linh hoạt.

## Vietnamese-Context Considerations

Người dùng Việt Nam thường:

- **code-switch** giữa tiếng Việt và tiếng Anh kỹ thuật: “Kubernetes autoscaling”, “cloud security”, “least privilege”
- gõ **không dấu** hoặc typo ngữ âm: “tu dong mo rong ha tang”
- dùng truy vấn ngắn nhưng giàu domain jargon

Vì vậy tôi chọn hai nguyên tắc:

1. **Tokenizer đơn giản cho lexical overlap, embedding lo phần semantic chính.** Với POC, whitespace/regex tokenization là đủ để giữ code sáng. Trong production tôi sẽ cân nhắc `pyvi` hoặc `underthesea`, nhưng phải trade off tốc độ và khả năng xử lý text vi/en mix.
2. **Ưu tiên metadata dễ explain thay vì black-box personalization.** Với người dùng Việt Nam và bối cảnh dữ liệu cá nhân, yếu tố riêng tư rất quan trọng; cần minh bạch memory nào được nhớ và vì sao được nhắc lại. Điều này cũng phù hợp tinh thần của Decree 13: tối thiểu hóa dữ liệu và kiểm soát mục đích sử dụng.

## Link Back To Lab Concepts

- **Vector store**: episodic memory search
- **RRF / hybrid thinking**: semantic score được bổ sung lexical overlap và profile affinity boost
- **Feature store**: stable profile + online lookup
- **TTL**: profile 30 ngày, recent activity 1 giờ
- **Streaming concept**: `queries_last_hour` là feature có freshness nhanh hơn profile
- **PIT join**: cần cho training/offline evaluation nếu sau này học một memory ranker

## What This POC Doesn't Handle Yet

POC này chưa giải quyết đầy đủ multi-user isolation ở mức mã hóa, memory CRUD, deletion requests, encryption at rest, cross-device sync, hay consolidation/forgetting policy sau 30-90 ngày. Nó cũng chưa có một streaming pipeline thật; recent activity vẫn được materialize theo batch nhỏ. Tuy vậy, design đã tách đúng trách nhiệm giữa vector store và feature store, nên những phần mở rộng này có chỗ để phát triển mà không phải thay kiến trúc gốc.

## Vibe-Coding Note

Prompt hiệu quả nhất là prompt ép AI “giữ Qdrant cho episodic, Feast cho profile, và viết rõ tradeoff X vs Y”. Prompt fail nhiều nhất là prompt kiểu “thiết kế AI memory architecture” quá rộng; nó trả lời đẹp nhưng mơ hồ, thiếu TTL, freshness và Vietnamese-context cụ thể.
