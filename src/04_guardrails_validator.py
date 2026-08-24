"""
Bước 4 — Guardrails AI Validators
====================================
NHIỆM VỤ:
  1. Xây dựng PIIDetector: phát hiện & redact email, số điện thoại, SSN, số thẻ tín dụng
  2. Xây dựng JSONFormatter: tự động sửa JSON lỗi
  3. Bọc mỗi validator trong Guard và test với các mẫu đầu vào
  4. Chạy demo với 6 trường hợp PII và 5 trường hợp JSON

DELIVERABLE: Tất cả test cases pass (PII bị redact, JSON được sửa thành công)

CÁC KHÁI NIỆM CHÍNH:
  - @register_validator     — khai báo custom validator class
  - Validator.validate()    — implement logic kiểm tra + sửa
  - OnFailAction.FIX        — thay thế output thay vì raise error
  - Guard().use(validator)  — gắn validator instance vào guard
  - guard.validate(text)    → ValidationOutcome
      .validation_passed    — bool
      .validated_output     — output đã được xử lý

⚠️  QUAN TRỌNG: on_fail phải truyền vào CONSTRUCTOR của VALIDATOR, KHÔNG phải Guard.use()
    SAI  : Guard().use(PIIDetector, on_fail=OnFailAction.FIX)   ← TypeError
    ĐÚNG : Guard().use(PIIDetector(on_fail=OnFailAction.FIX))   ← correct
"""

import sys
import re
import json
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from guardrails import Guard
from guardrails.validators import Validator, register_validator, PassResult, FailResult

try:
    from guardrails.hub import OnFailAction
except ImportError:
    from guardrails.validator_base import OnFailAction


# ── 1. PII Detector Validator ──────────────────────────────────────────────
@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Phát hiện và redact Personally Identifiable Information (PII).

    Các pattern được phát hiện:
      EMAIL       : xxx@xxx.xxx
      PHONE       : (123) 456-7890 hoặc 123-456-7890 hoặc 555-123-4567
      SSN         : 123-45-6789
      CREDIT_CARD : 1234 5678 9012 3456 (hoặc dấu gạch nối)
    """

    # Regex patterns cho từng loại PII
    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE":       r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\b\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict = None):
        """
        Tìm PII trong value; nếu phát hiện, redact và trả về FailResult với fix_value đã xử lý
        khi kết hợp cùng OnFailAction.FIX.
        """
        redacted_text = value
        found_pii = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, redacted_text)
            if matches:
                found_pii.append(pii_type)
                redacted_text = re.sub(pattern, f"[{pii_type}_REDACTED]", redacted_text)

        if found_pii:
            print(f"  ⚠️  Đã phát hiện và redact {len(found_pii)} loại PII: {list(set(found_pii))}")
            return FailResult(
                error_message=f"Phát hiện PII: {', '.join(set(found_pii))}",
                fix_value=redacted_text,
            )

        return PassResult(value_override=value)


# ── 2. JSON Formatter Validator ────────────────────────────────────────────
@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validate và tự động sửa JSON lỗi.

    Các lỗi có thể sửa tự động:
      - Strip markdown code fences (``` hoặc ```json)
      - Thay single quotes → double quotes
      - Xóa trailing commas trước } hoặc ]
      - Re-serialize với json.dumps để định dạng chuẩn
    """

    @staticmethod
    def _repair(text: str) -> str:
        """
        Cố gắng sửa chuỗi JSON lỗi.
        """
        text = text.strip()

        # Xóa markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
        text = text.strip()

        # Thay single quotes cho keys và string values
        text = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', text)
        text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
        # Sửa single quotes trong array elements
        text = re.sub(r"\[\s*'([^']*)'", r'["\1"', text)
        text = re.sub(r",\s*'([^']*)'", r', "\1"', text)

        # Xóa trailing commas trước } hoặc ]
        text = re.sub(r",\s*([}\]])", r"\1", text)

        return text

    def validate(self, value: str, metadata: dict = None):
        """
        Thử parse value thành JSON.
        Nếu thất bại, gọi _repair() rồi thử lại.
        """
        # Thử parse trực tiếp
        try:
            parsed = json.loads(value)
            formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
            return PassResult(value_override=formatted)
        except (json.JSONDecodeError, TypeError):
            pass

        # Thử sửa JSON rồi parse lại
        try:
            repaired_text = self._repair(value)
            parsed = json.loads(repaired_text)
            formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
            print(f"  🔧 JSON đã được tự động sửa thành công")
            return FailResult(
                error_message="JSON đã được sửa lỗi định dạng tự động",
                fix_value=formatted,
            )
        except (json.JSONDecodeError, TypeError) as e:
            fallback = json.dumps({
                "error": "Không thể phân tích JSON",
                "raw_preview": str(value)[:200],
            }, ensure_ascii=False, indent=2)
            return FailResult(
                error_message=f"JSON không hợp lệ sau khi sửa: {e}",
                fix_value=fallback,
            )


# ── 3. Demo: PII Guard ─────────────────────────────────────────────────────
def demo_pii_guard() -> str:
    output_lines = []
    header = "\n" + "=" * 55 + "\n  Demo: PII Detection & Redaction\n" + "=" * 55
    print(header)
    output_lines.append(header)

    # Truyền on_fail=OnFailAction.FIX vào constructor của validator
    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",        "Contact John at john.doe@example.com for details."),
        ("Phone",        "Call our support line at (555) 867-5309."),
        ("SSN",          "Patient SSN is 123-45-6789 on file."),
        ("Credit Card",  "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",    "Email: alice@example.com, Phone: 555-123-4567, Card: 5500-0000-0000-0004"),
        ("Clean",        "No sensitive information in this text. Everything is public knowledge."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "✅ PASS (Clean)" if result.validation_passed and "[PII_REDACTED]" not in str(result.validated_output) else "🛡️ FIX (PII Redacted)"
        line_case = f"\n[{label}] {status}\n  Input:  {text}\n  Output: {result.validated_output}"
        print(line_case)
        output_lines.append(line_case)

    return "\n".join(output_lines)


# ── 4. Demo: JSON Guard ────────────────────────────────────────────────────
def demo_json_guard() -> str:
    output_lines = []
    header = "\n" + "=" * 55 + "\n  Demo: JSON Formatting & Repair\n" + "=" * 55
    print(header)
    output_lines.append(header)

    # Truyền on_fail=OnFailAction.FIX vào constructor của validator
    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",       '{"name": "Alice", "age": 30}'),
        ("Markdown fences",  '```json\n{"name": "Bob", "score": 95}\n```'),
        ("Single quotes",    "{'name': 'Charlie', 'active': true}"),
        ("Trailing comma",   '{"items": ["a", "b",], "total": 2,}'),
        ("Truly invalid",    "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "✅ PASS (Valid JSON)" if result.validation_passed else "🔧 FIX/FALLBACK"
        line_case = f"\n[{label}] {status}\n  Input:  {text[:60]}\n  Output: {str(result.validated_output)}"
        print(line_case)
        output_lines.append(line_case)

    return "\n".join(output_lines)


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Bước 4: Guardrails AI Validators")
    print("=" * 55)

    pii_log = demo_pii_guard()
    json_log = demo_json_guard()

    # Lưu evidence logs
    evidence_dir = Path(__file__).parent.parent / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    pii_log_file = evidence_dir / "04_pii_demo_log.txt"
    json_log_file = evidence_dir / "04_json_demo_log.txt"

    pii_log_file.write_text(pii_log, encoding="utf-8")
    json_log_file.write_text(json_log, encoding="utf-8")

    print(f"\n💾 Đã lưu evidence log:")
    print(f"   - {pii_log_file}")
    print(f"   - {json_log_file}")
    print("\n✅ Bước 4 hoàn thành xuất sắc!")


if __name__ == "__main__":
    main()
