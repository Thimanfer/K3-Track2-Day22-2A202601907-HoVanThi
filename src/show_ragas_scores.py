"""
Tiện ích hiển thị bảng điểm RAGAS Evaluation đã được đánh giá.
Chạy lệnh:
    python src/show_ragas_scores.py
"""
import sys
import json
from pathlib import Path

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

report_path = Path(__file__).parent.parent / "evidence" / "03_ragas_report.json"
if not report_path.exists():
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"

if not report_path.exists():
    print(f"❌ Chưa tìm thấy file báo cáo {report_path}. Hãy chạy python src/03_ragas_evaluation.py trước.")
    sys.exit(1)

data = json.loads(report_path.read_text(encoding="utf-8"))

v1 = data.get("prompt_v1_scores", {})
v2 = data.get("prompt_v2_scores", {})

print("\n" + "=" * 68)
print(f"  {'BẢNG SO SÁNH KẾT QUẢ ĐÁNH GIÁ RAGAS (50 QA PAIRS)':^64}")
print(f"  Học viên: {data.get('student_name', 'Hồ Văn Thi')} | MSSV: {data.get('student_id', '2A202601907')}")
print("=" * 68)
print(f"  {'Chỉ số đánh giá (RAGAS Metric)':32s}  {'V1':>8}  {'V2':>8}  {'So sánh':>10}")
print("=" * 68)

metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
for m in metrics:
    s1 = v1.get(m, 0.0)
    s2 = v2.get(m, 0.0)
    winner = "← V1 thắng" if s1 > s2 else ("← V2 thắng" if s2 > s1 else "Hòa nhau")
    print(f"  {m:32s}  {s1:>8.4f}  {s2:>8.4f}  {winner:>10}")

print("=" * 68)
best_faith = max(v1.get("faithfulness", 0.0), v2.get("faithfulness", 0.0))
if best_faith >= 0.9:
    print(f"  🌟 XUẤT SẮC: Faithfulness = {best_faith:.4f} ≥ 0.9 (Đạt chuẩn điểm thưởng tối đa!)")
elif best_faith >= 0.8:
    print(f"  ✅ ĐẠT YÊU CẦU: Faithfulness = {best_faith:.4f} ≥ 0.8")
print("=" * 68 + "\n")
