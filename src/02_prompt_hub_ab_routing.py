"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import hashlib
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
PROMPT_V1_NAME = "hovanthi-rag-prompt-v1"
PROMPT_V2_NAME = "hovanthi-rag-prompt-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# V1: Phong cách ngắn gọn, thân thiện (2-4 câu)
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện và hữu ích. Chỉ sử dụng thông tin trong Context được cung cấp để trả lời câu hỏi. "
    "Hãy giữ câu trả lời súc tích, ngắn gọn trong khoảng 2-4 câu và đi thẳng vào trọng tâm. "
    "Nếu không tìm thấy thông tin trong Context, hãy nói rõ là không tìm thấy thông tin.\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

# V2: Phong cách chuyên gia phân tích có cấu trúc (3-5 câu kèm đề mục rõ ràng)
SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích kỹ thuật AI cao cấp. Hãy đọc kỹ Context và đưa ra câu trả lời có cấu trúc, chi tiết và chuẩn xác (khoảng 3-5 câu). "
    "Cấu trúc trả lời gồm:\n"
    "1. Tổng quan / Câu trả lời chính\n"
    "2. Các điểm chi tiết / Minh chứng từ dữ liệu\n"
    "3. Lưu ý kỹ thuật hoặc mức độ tin cậy\n"
    "Tuyệt đối chỉ dựa vào Context, không suy diễn hoặc bịa đặt ngoài tài liệu.\n\n"
    "Context:\n{context}"
)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client: Client):
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.
    """
    try:
        url_v1 = client.push_prompt(
            PROMPT_V1_NAME,
            object=PROMPT_V1,
            description="V1 – Phong cách ngắn gọn, thân thiện (Hồ Văn Thi - 2A202601907)",
        )
        print(f"✅ Đã push V1 lên Hub → {url_v1}")
    except Exception as e:
        print(f"⚠️  V1 push lỗi hoặc đã tồn tại: {e}")

    try:
        url_v2 = client.push_prompt(
            PROMPT_V2_NAME,
            object=PROMPT_V2,
            description="V2 – Phong cách chuyên gia, có cấu trúc (Hồ Văn Thi - 2A202601907)",
        )
        print(f"✅ Đã push V2 lên Hub → {url_v2}")
    except Exception as e:
        print(f"⚠️  V2 push lỗi hoặc đã tồn tại: {e}")


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub.
    Fallback về template local nếu Hub không khả dụng.
    """
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Đã pull '{PROMPT_V1_NAME}' từ LangSmith Prompt Hub")
    except Exception as e:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V1_NAME}' (Lý do: {e})")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Đã pull '{PROMPT_V2_NAME}' từ LangSmith Prompt Hub")
    except Exception as e:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V2_NAME}' (Lý do: {e})")

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.

    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic).
    """
    hash_int = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Chạy RAG chain với prompt version được chọn bởi router.
    """
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return {
        "question": question,
        "answer": answer,
        "version": version,
    }


# ── 7. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    client = Client(api_key=config.LANGSMITH_API_KEY)

    print("📤 Đang đồng bộ Prompts lên LangSmith Prompt Hub...")
    push_prompts_to_hub(client)

    print("\n📥 Đang tải Prompts từ Hub về ứng dụng...")
    prompts = pull_prompts_from_hub(client)

    print("\n🔨 Khởi tạo FAISS Vectorstore & LLM...")
    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    log_lines = []
    header_log = f"=== A/B Routing Log (50 Requests) ===\nV1 Prompt: {PROMPT_V1_NAME}\nV2 Prompt: {PROMPT_V2_NAME}\n"
    log_lines.append(header_log)

    print(f"\n🔀 Bắt đầu A/B Routing cho {len(SAMPLE_QUESTIONS)} câu hỏi...")
    v1_count, v2_count = 0, 0
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        request_id  = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        result = ask_ab(retriever, llm, prompt, question, version_tag)

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1

        line_out = f"[{i:02d}] [{request_id}] [prompt-{version_tag}] Q: {question[:55]}..."
        ans_preview = f"     A: {str(result['answer'])[:75]}..."
        print(line_out)
        print(ans_preview)

        log_lines.append(line_out)
        log_lines.append(f"     Full Answer: {result['answer']}")
        import time
        time.sleep(3.5)

    summary = f"\n📊 Thống kê Routing: V1={v1_count} câu ({v1_count/len(SAMPLE_QUESTIONS)*100:.1f}%) | V2={v2_count} câu ({v2_count/len(SAMPLE_QUESTIONS)*100:.1f}%) | Tổng={len(SAMPLE_QUESTIONS)}"
    print(summary)
    log_lines.append(summary)

    # Lưu evidence file
    evidence_dir = Path(__file__).parent.parent / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    log_file = evidence_dir / "02_ab_routing_log.txt"
    log_file.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"💾 Đã lưu log A/B routing vào: {log_file}")
    print("✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()
