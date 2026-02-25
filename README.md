### USER
проект: распознавание аудио файла по голосам, запись разпознанного текста по предложениям, и использование нейросети llama3 для суммаризации разговора

### ASSISTANT
Это отличная идея для проекта. Создание системы, которая не просто переводит речь в текст, но и понимает, *кто* говорит, а затем анализирует смысл диалога — это полноценный пайплайн обработки данных (Data Pipeline).

Ниже представлен подробный план реализации, выбор технологий и пример кода на Python.

### 1. Архитектура проекта

Проект будет состоять из трех основных модулей:
1.  **Модуль диаризации и транскрибации (STT + Diarization):** Разделяет аудио на сегменты и определяет спикеров.
2.  **Модуль предобработки текста:** Форматирует текст по предложениям для удобства чтения.
3.  **Модуль LLM (Llama 3):** Принимает текст и генерирует саммари (краткое содержание).

### 2. Стек технологий (Open Source)

Для реализации мы будем использовать бесплатные и мощные библиотеки:

*   **Язык:** Python 3.9+
*   **Распознавание речи (STT):** `faster-whisper` (быстрая и точная версия Whisper от OpenAI).
*   **Диаризация (разделение голосов):** `pyannote.audio` (золотой стандарт для разделения спикеров).
*   **Нейросеть (LLM):** `Ollama` (для локального запуска Llama 3) или API (например, Groq, если нет мощной видеокарты).
*   **Обработка текста:** `nltk` или стандартные методы Python.

---

### 3. Пошаговая реализация

#### Шаг 1: Установка зависимостей

Вам понадобится установленный `ffmpeg` в системе.
Создайте виртуальное окружение и установите библиотеки:

```bash
pip install faster-whisper pyannote.audio nltk ollama
```

*Примечание: Для `pyannote.audio` вам потребуется бесплатно зарегистрироваться на Hugging Face, принять лицензионное соглашение для моделей `pyannote/speaker-diarization-3.1` и `pyannote/segmentation-3.1`, и получить **Access Token**.*

#### Шаг 2: Код проекта

Создайте файл `main.py`. Ниже приведен полный пример скрипта.

```python
import os
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from nltk.tokenize import sent_tokenize
import ollama

# --- КОНФИГУРАЦИЯ ---
AUDIO_FILE = "meeting_recording.mp3"  # Путь к вашему файлу
HF_TOKEN = "ВАШ_HUGGINGFACE_TOKEN"    # Токен от Hugging Face для pyannote
LLAMA_MODEL = "llama3"                # Модель в Ollama

# --- 1. ДИАРИЗАЦИЯ И ТРАНСКРИБАЦИЯ ---
def process_audio(audio_path):
    print(f"Загрузка аудио: {audio_path}...")
    
    # Инициализация пайплайна диаризации (требует GPU для скорости, но работает и на CPU)
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
    
    # Инициализация Whisper (можно выбрать 'large-v3' для лучшего качества)
    whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    print("Запуск диаризации и распознавания...")
    
    # Запускаем диаризацию
    diarization = diarization_pipeline(audio_path)
    
    # Запускаем транскрибацию с временными метками
    segments, info = whisper_model.transcribe(audio_path, word_timestamps=True)
    segments = list(segments)
    
    transcript_with_speakers = []

    # Сопоставляем сегменты речи со спикерами
    for segment in segments:
        start, end = segment.start, segment.end
        text = segment.text
        
        # Находим, кто говорил в это время
        speakers = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Проверяем пересечение временных интервалов
            if turn.end > start and turn.start < end:
                speakers.append(speaker)
        
        # Берем самого частого спикера в этом сегменте (упрощенно)
        speaker_label = speakers[0] if speakers else "UNKNOWN"
        
        transcript_with_speakers.append({
            "speaker": speaker_label,
            "text": text.strip(),
            "start": start,
            "end": end
        })

    return transcript_with_speakers

# --- 2. ФОРМАТИРОВАНИЕ ТЕКСТА ---
def format_text_for_llm(transcript_data):
    formatted_text = ""
    for item in transcript_
        # Разбиваем текст на предложения для чистоты
        sentences = sent_tokenize(item['text'], language='russian') 
        for sentence in sentences:
            formatted_text += f"{item['speaker']}: {sentence}\n"
    return formatted_text

# --- 3. СУММАРИЗАЦИЯ ЧЕРЕЗ LLAMA 3 ---
def summarize_conversation(text):
    print("Отправка текста в Llama 3 для анализа...")
    
    prompt = f"""
    Ты — профессиональный секретарь. Ниже приведен текст разговора между несколькими участниками.
    Твоя задача:
    1. Сделать краткое резюме (саммари) разговора (3-5 предложений).
    2. Выделить ключевые договоренности или задачи (bullet points).
    3. Определить тональность разговора.

    Текст разговора:
    {text}
    """

    response = ollama.chat(model=LLAMA_MODEL, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    
    return response['message']['content']

# --- ОСНОВНОЙ ЗАПУСК ---
if __name__ == "__main__":
    # Скачиваем модели nltk для токенизации (нужно сделать один раз)
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')

    if not os.path.exists(AUDIO_FILE):
        print(f"Файл {AUDIO_FILE} не найден!")
        exit()

    # 1. Обработка аудио
    raw_data = process_audio(AUDIO_FILE)
    
    # 2. Форматирование
    full_text = format_text_for_llm(raw_data)
    
    # Сохраняем полный текст в файл
    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    print("Текст сохранен в transcript.txt")

    # 3. Саммари
    # Внимание: Llama 3 имеет ограничение по контексту (обычно 8k токенов). 
    # Если файл очень длинный, текст нужно резать на куски.
    summary = summarize_conversation(full_text)
    
    print("\n--- РЕЗУЛЬТАТ (SUMMARY) ---")
    print(summary)
```

### 4. Важные нюансы и оптимизация

#### А. Ограничение контекста (Context Window)
Llama 3 (обычная версия) имеет окно контекста 8000 токенов. Это примерно 10-15 минут плотного разговора.
*   **Если файл длинный:** Вам нужно реализовать логику "Map-Reduce".
    1.  Разбить текст на куски по 5000 токенов.
    2.  Попросить Llama сделать саммари для каждого куска.
    3.  Склеить эти саммари и попросить Llama сделать итоговое саммари из них.

#### Б. Производительность (GPU vs CPU)
*   **Whisper и PyAnnote** очень требовательны к ресурсам. На CPU обработка 1 минуты аудио может занять 1-2 минуты. На GPU (NVIDIA) это займет несколько секунд.
*   **Llama 3:** Запуск через Ollama на CPU возможен, но генерация будет медленной. Для продакшена лучше использовать API (например, **Groq API** — они предоставляют очень быстрый доступ к Llama 3 бесплатно/недорого).

#### В. Альтернатива для API (Groq)
Если у вас нет мощной видеокарты, замените функцию `summarize_conversation` на запрос к Groq API:

```python
from groq import Groq

client = Groq(api_key="ВАШ_КЛЮЧ_GROQ")

def summarize_conversation_api(text):
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": f"Сделай саммари этого диалога: {text}"
        }],
        model="llama3-70b-8192", # Очень мощная и быстрая модель
    )
    return chat_completion.choices[0].message.content
```

### 5. Как улучшить проект в будущем?

1.  **Идентификация спикеров:** Сейчас PyAnnote говорит "SPEAKER_00", "SPEAKER_01". Можно добавить этап, где вы вручную (или через few-shot learning) присваиваете имена: "SPEAKER_00 = Иван".
2.  **Поиск по диалогу:** Сохраните результат в векторную базу данных (ChromaDB или FAISS). Тогда вы сможете спрашивать у чат-бота: "Что Иван сказал про бюджет?", и он найдет нужный фрагмент.
3.  **Веб-интерфейс:** Оберните этот скрипт в **Streamlit**, чтобы загружать файлы через браузер и видеть результат.

Этот план дает вам работающий прототип (MVP), который можно развивать в полноценный продукт.
