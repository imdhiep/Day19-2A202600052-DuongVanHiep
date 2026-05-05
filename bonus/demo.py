from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent


def main() -> None:
    agent = HybridMemoryAgent()
    memories = [
        (
            "Mình đã đọc tài liệu về Kubernetes Horizontal Pod Autoscaler, "
            "cluster autoscaling và cách tăng replica theo CPU để tránh nghẽn giờ cao điểm."
        ),
        (
            "Ghi chú hôm nay: AWS Lambda phù hợp cho serverless burst traffic, "
            "nhưng cần theo dõi cold start và chi phí khi request tăng đột biến."
        ),
        (
            "Đã lưu một checklist cloud security gồm TLS mọi hop, IAM least privilege, "
            "secret rotation, network segmentation và zero-trust cho workload nội bộ."
        ),
        (
            "Tài liệu hạ tầng nhấn mạnh Terraform + VPC riêng biệt cho multi-tenant deployment, "
            "kèm autoscaling group để mở rộng linh hoạt theo lưu lượng."
        ),
        (
            "Mình cũng xem một note so sánh Kubernetes autoscaling với EC2 auto scaling; "
            "kết luận là Kubernetes tốt hơn khi cần scale container nhanh và theo workload thực tế."
        ),
    ]
    for memory in memories:
        agent.remember(memory)

    queries = [
        "Tôi đã đọc gì về Kubernetes?",
        "Recommend đọc gì tiếp",
        "Tôi đang quan tâm gì gần đây?",
        "Tài liệu về tự động mở rộng hạ tầng?",
        "Cho tôi summary cloud security",
    ]
    for idx, query in enumerate(queries, start=1):
        print(f"\n=== Query {idx} ===")
        print(agent.recall(query))


if __name__ == "__main__":
    main()
