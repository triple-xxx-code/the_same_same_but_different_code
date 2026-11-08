# pip install magic-pdf[full] transformers torch pdf2image pillow accelerate ollama --extra-index-url https://pytorch.org
# Системные зависимости:
#   Ubuntu: sudo apt install poppler-utils
#   macOS:  brew install poppler

# Примеры запуска:
#   python multi_pdf_parser.py --pdf my_doc.pdf
#   python multi_pdf_parser.py --pdf my_doc.pdf --mode mineru
#   python multi_pdf_parser.py --pdf my_doc.pdf --mode glm
#   python multi_pdf_parser.py --pdf my_doc.pdf --mode qwen
#   python multi_pdf_parser.py --pdf my_doc.pdf --mode qwen --ollama-model qwen3.8:27b

import os
import gc
import argparse
import traceback
from io import BytesIO
from typing import List, Tuple

import torch
from PIL import Image
from pdf2image import convert_from_path


# ============================================================
#  УТИЛИТЫ ДЛЯ УПРАВЛЕНИЯ ПАМЯТЬЮ GPU
# ============================================================
def free_gpu_memory(label: str = ""):
    """Освобождает кэш CUDA и вызывает сборщик мусора Python."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if label:
        print(f"🧹 Память GPU освобождена ({label})")


def report_gpu_memory():
    """Печатает текущее потребление VRAM (если CUDA доступна)."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   VRAM: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")


# ============================================================
#  1. ПАРСЕР: MINERU
# ============================================================
def parse_with_mineru(pdf_path: str, output_dir: str) -> bool:
    print("\n--- [1] Запуск парсинга через MinerU ---")
    try:
        from magic_pdf.pipe.UNIPipe import UNIPipe
        from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
    except ImportError:
        print("❌ magic-pdf не установлен. Пропускаю MinerU.")
        return False

    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, "mineru_images")
    os.makedirs(image_dir, exist_ok=True)          # DiskReaderWriter требует существующей папки
    image_writer = DiskReaderWriter(image_dir)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    print("🔄 Анализ структуры и математических формул...")
    try:
        pipe = UNIPipe(pdf_bytes, {"_pdf_type": "", "model_list": []}, image_writer)
        # В разных версиях magic-pdf методы называются по-разному
        if hasattr(pipe, "pipe_classify"):
            pipe.pipe_classify()
        elif hasattr(pipe, "pipe_analyze"):
            pipe.pipe_analyze()

        pipe.pipe_parse()

        # --- Markdown ---
        # В актуальных версиях magic-pdf используется pipe_mk_markdown, а не to_markdown
        if hasattr(pipe, "pipe_mk_markdown"):
            markdown_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
        elif hasattr(pipe, "to_markdown"):
            markdown_content = pipe.to_markdown()          # fallback для старых версий
        else:
            markdown_content = ""

        # --- Чистый текст ---
        # pipe.pipe_res — это обычно список, а не dict; .get() у него нет
        layout_dets = []
        if hasattr(pipe, "pdf_mid_data") and isinstance(pipe.pdf_mid_data, dict):
            layout_dets = pipe.pdf_mid_data.get("layout_dets", [])
        elif hasattr(pipe, "pipe_res") and isinstance(pipe.pipe_res, list):
            layout_dets = pipe.pipe_res

        txt_lines = [
            item["text"] for item in layout_dets
            if isinstance(item, dict) and "text" in item
        ]
        text_content = "\n".join(txt_lines)

        with open(os.path.join(output_dir, "output_mineru.md"), "w", encoding="utf-8") as f:
            f.write(markdown_content)
        with open(os.path.join(output_dir, "output_mineru.txt"), "w", encoding="utf-8") as f:
            f.write(text_content)

        print("🎉 Результаты MinerU сохранены в папку:", output_dir)
        return True

    except Exception as e:
        print(f"❌ Ошибка MinerU: {e}")
        traceback.print_exc()
        return False


# ============================================================
#  2. ПАРСЕР: GLM-OCR (локальная модель на GPU)
# ============================================================
def parse_with_glm(pdf_path: str, output_dir: str) -> bool:
    print("\n--- [2] Запуск парсинга через GLM-OCR (bfloat16 на GPU) ---")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("❌ transformers не установлен. Пропускаю GLM.")
        return False

    os.makedirs(output_dir, exist_ok=True)

    # ВАЖНО: проверьте актуальное имя модели на HuggingFace.
    # На момент написания "THUDM/glm-ocr" может не существовать;
    # распространённые варианты: THUDM/glm-4v-9b, THUDM/GLM-4.1V-9B-Thinking и т.п.
    model_id = "THUDM/glm-ocr"

    print(f"🔄 Загрузка модели {model_id} в видеопамять...")
    model = None
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()

        print("🔄 Конвертация PDF в изображения...")
        pages = convert_from_path(pdf_path, dpi=150)
        prompt = (
            "Распознай текст на этой странице. Верни результат в формате Markdown, "
            "бережно сохраняя структуру таблиц, списков и заголовков."
        )

        md_lines, txt_lines = [], []

        for i, page in enumerate(pages, start=1):
            print(f"⏳ Обработка страницы {i}/{len(pages)}...")
            image = page.convert("RGB")

            # --- ИСПРАВЛЕНИЕ: передаём images=[image], а не images=, ---
            inputs = model.build_conversation_input_ids(
                tokenizer, query=prompt, history=[], images=[image]
            )

            # --- ИСПРАВЛЕНИЕ: для GLM-4V нужен особый формат inputs ---
            inputs = {
                "input_ids":      inputs["input_ids"].unsqueeze(0).to("cuda"),
                "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to("cuda"),
                "attention_mask": inputs["attention_mask"].unsqueeze(0).to("cuda"),
                "images": [[inputs["images"][0].to("cuda").to(torch.bfloat16)]],
            }

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # --- ИСПРАВЛЕНИЕ: отрезаем prompt по длине input_ids, а не по batch size ---
            prompt_len = inputs["input_ids"].shape[-1]
            page_content = tokenizer.decode(
                outputs[0][prompt_len:], skip_special_tokens=True
            )

            md_lines.append(f"## Страница {i}\n\n{page_content}\n\n---\n\n")
            txt_lines.append(f"--- Страница {i} ---\n{page_content}\n\n")

        with open(os.path.join(output_dir, "output_glm.md"), "w", encoding="utf-8") as f:
            f.write("".join(md_lines))
        with open(os.path.join(output_dir, "output_glm.txt"), "w", encoding="utf-8") as f:
            f.write("".join(txt_lines))

        print("🎉 Результаты GLM-OCR сохранены в папку:", output_dir)
        return True

    except Exception as e:
        print(f"❌ Ошибка GLM-OCR: {e}")
        traceback.print_exc()
        return False

    finally:
        # --- ОСВОБОЖДЕНИЕ ПАМЯТИ GPU ---
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        free_gpu_memory("после GLM")


# ============================================================
#  3. ПАРСЕР: QWEN3.8 ЧЕРЕЗ OLLAMA
# ============================================================
def parse_with_qwen_ollama(
    pdf_path: str,
    output_dir: str,
    model_name: str = "qwen3.8:27b-bf16",
) -> bool:
    print(f"\n--- [3] Запуск парсинга через Ollama ({model_name}) ---")

    try:
        import ollama
    except ImportError:
        print("❌ ollama-python не установлен. Пропускаю Qwen.")
        return False

    os.makedirs(output_dir, exist_ok=True)

    print("🔄 Конвертация PDF в изображения...")
    try:
        pages = convert_from_path(pdf_path, dpi=150)
    except Exception as e:
        print(f"❌ Не удалось сконвертировать PDF: {e}")
        print("   Убедитесь, что установлен poppler (sudo apt install poppler-utils).")
        return False

    prompt = (
        "Переведи изображение этой страницы документа в структурированный текст. "
        "Если на странице есть картинки или схемы, подробно опиши их "
        "внутри блока [ИЛЛЮСТРАЦИЯ: ... описание ...]. "
        "Ответ верни строго в формате Markdown."
    )

    md_lines, txt_lines = [], []

    for i, page in enumerate(pages, start=1):
        print(f"⏳ Ollama обрабатывает страницу {i}/{len(pages)}...")
        # --- ИСПРАВЛЕНИЕ: приводим к RGB перед JPEG ---
        page_rgb = page.convert("RGB")
        img_buffer = BytesIO()
        page_rgb.save(img_buffer, format="JPEG")
        img_bytes = img_buffer.getvalue()

        try:
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                images=[img_bytes],
            )

            # --- ИСПРАВЛЕНИЕ: поддержка и dict, и Pydantic-объекта ---
            if isinstance(response, dict):
                page_content = response.get("response", "")
            else:
                page_content = getattr(response, "response", "") or ""

            md_lines.append(f"## Страница {i}\n\n{page_content}\n\n---\n\n")
            txt_lines.append(f"--- Страница {i} ---\n{page_content}\n\n")

        except Exception as e:
            print(f"❌ Ошибка Ollama на странице {i}: {e}")
            print(f"   Убедитесь, что модель загружена: ollama pull {model_name}")

    with open(os.path.join(output_dir, "output_qwen.md"), "w", encoding="utf-8") as f:
        f.write("".join(md_lines))
    with open(os.path.join(output_dir, "output_qwen.txt"), "w", encoding="utf-8") as f:
        f.write("".join(txt_lines))

    print("🎉 Результаты Qwen сохранены в папку:", output_dir)

    # --- ОСВОБОЖДЕНИЕ ПАМЯТИ OLLAMA ---
    # Ollama управляет памятью самостоятельно, но можно попросить выгрузить модель,
    # отправив фиктивный запрос с keep_alive=0 (модель сразу выгрузится из VRAM).
    try:
        ollama.generate(model=model_name, prompt="", keep_alive=0)
        print("🧹 Модель Ollama выгружена из памяти (keep_alive=0).")
    except Exception:
        pass  # не критично, если не удалось

    free_gpu_memory("после Qwen/Ollama")
    return True


# ============================================================
#  ОСНОВНОЙ БЛОК УПРАВЛЕНИЯ
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Универсальный PDF-парсер (MinerU, GLM-OCR, Qwen3.8 через Ollama)"
    )
    parser.add_argument("--pdf", type=str, default="example.pdf", help="Путь к PDF-файлу")
    parser.add_argument("--out", type=str, default="./parsed_results", help="Выходная папка")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["mineru", "glm", "qwen", "all"],
        help="Выбор парсера: mineru, glm, qwen или all (запустить все по очереди)",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="qwen3.8:27b-bf16",
        help="Тег модели Ollama для Qwen (по умолчанию qwen3.8:27b-bf16; "
             "для экономии VRAM используйте qwen3.8:27b)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"❌ Критическая ошибка: файл '{args.pdf}' не найден!")
        exit(1)

    print(f"🚀 Старт обработки файла: {args.pdf}")
    report_gpu_memory()

    results = {}

    if args.mode in ["mineru", "all"]:
        free_gpu_memory("перед MinerU")
        results["MinerU"] = parse_with_mineru(args.pdf, args.out)
        free_gpu_memory("после MinerU")

    if args.mode in ["glm", "all"]:
        free_gpu_memory("перед GLM")
        results["GLM-OCR"] = parse_with_glm(args.pdf, args.out)
        # внутри parse_with_glm уже вызывается free_gpu_memory в finally

    if args.mode in ["qwen", "all"]:
        free_gpu_memory("перед Qwen/Ollama")
        results["Qwen/Ollama"] = parse_with_qwen_ollama(
            args.pdf, args.out, model_name=args.ollama_model
        )
        # внутри parse_with_qwen_ollama уже вызывается free_gpu_memory

    # --- Итоговый отчёт ---
    print("\n" + "=" * 50)
    print("📊 ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 50)
    all_ok = True
    for name, ok in results.items():
        status = "✅ успешно" if ok else "❌ с ошибками / пропущен"
        print(f"  {name:15s}: {status}")
        if not ok:
            all_ok = False

    report_gpu_memory()
    free_gpu_memory("завершение работы")

    if all_ok:
        print("\n🏁 Все выбранные процессы успешно завершены!")
    else:
        print("\n⚠️ Некоторые процессы завершились с ошибками — проверьте лог выше.")