# Анализ проблемы с таймкодами при начальной тишине в аудио

## Описание проблемы

При транскрипции двухканальных записей (клиент/сотрудник), разделенных на отдельные дорожки, возникает проблема:
- В начале дорожки клиента есть переменная пауза (15-20 секунд тишины)
- ASR сервис получает файл с этой паузой
- Таймкоды в транскрипции начинаются с 0.0, не учитывая начальную паузу
- Необходимо, чтобы таймкоды отражали реальное время в исходной записи

## Текущая архитектура проекта

### Используемый движок
- **ASR_ENGINE**: `whisperx` (из `docker-compose.yml`)
- Альтернативный движок: `faster_whisper` (также доступен)

### Поток обработки аудио

1. **Эндпоинт `/asr`** (`app/webservice.py:55-108`)
   - Принимает `audio_file` через POST запрос
   - Параметры: `vad_filter`, `word_timestamps`, `diarize`, и др.
   - Вызывает `asr_model.transcribe()`

2. **Загрузка аудио** (`app/utils.py:97-127`)
   - Функция `load_audio()` обрабатывает входящий файл
   - Если `encode=True` (по умолчанию), использует ffmpeg:
   ```python
   ffmpeg.input("pipe:", threads=0)
       .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
   ```
   - **ВАЖНО**: В команде ffmpeg НЕТ фильтров типа:
     - `silenceremove` - удаление тишины
     - `-ss` - обрезка начала
     - `-t` - обрезка конца
     - `-af` с фильтрами тишины
   - Аудио декодируется полностью, без обрезки

3. **Транскрипция через WhisperX** (`app/asr_models/mbain_whisperx_engine.py:41-93`)
   - Вызывает `self.model['whisperx'].transcribe(audio, **options_dict)`
   - Затем выполняет выравнивание: `whisperx.align()`
   - При необходимости выполняет диаризацию: `whisperx.assign_word_speakers()`

4. **Транскрипция через Faster Whisper** (`app/asr_models/faster_whisper_engine.py:27-66`)
   - Вызывает `self.model.transcribe(audio, beam_size=5, **options_dict)`
   - Параметр `vad_filter` передается в `options_dict` только если `vad_filter=True`
   - **ПРОБЛЕМА**: `vad_filter` передается как булево значение, а не как параметр из запроса напрямую

## Анализ по пунктам рекомендаций

### 1.1. Проверка ffmpeg-декодирования на тримминг тишины

**СТАТУС**: ✅ **ПРОБЛЕМ НЕ ОБНАРУЖЕНО**

**Место в коде**: `app/utils.py:117-120`

```python
out, _ = (
    ffmpeg.input("pipe:", threads=0)
    .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
    .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=file.read())
)
```

**Выводы**:
- В команде ffmpeg отсутствуют фильтры удаления тишины
- Нет параметров `-ss`, `-t` для обрезки
- Нет `-af silenceremove=...`
- Аудио декодируется полностью, включая начальную тишину

**Заключение**: Проблема НЕ в ffmpeg-декодировании. Аудио целиком доходит до модели.

---

### 1.2. Анализ VAD в faster_whisper движке

**СТАТУС**: ⚠️ **ПОТЕНЦИАЛЬНАЯ ПРОБЛЕМА**

**Место в коде**: `app/asr_models/faster_whisper_engine.py:44-52`

```python
options_dict = {"task": task}
if language:
    options_dict["language"] = language
if initial_prompt:
    options_dict["initial_prompt"] = initial_prompt
if vad_filter:  # <-- ПРОБЛЕМА: проверяется булево значение
    options_dict["vad_filter"] = True
if word_timestamps:
    options_dict["word_timestamps"] = True
```

**Проблемы**:
1. `vad_filter` передается только если он `True`, но не передается явно как `False`
2. В библиотеке `faster_whisper` параметр `vad_filter` может работать некорректно - он может обрезать начальную тишину
3. Если `vad_filter=True`, модель может автоматически обрезать тишину в начале, что приведет к таймкодам, начинающимся с 0.0

**Рекомендации**:
- Если используется `faster_whisper` и `vad_filter=True`, это может быть причиной проблемы
- Нужно либо отключить `vad_filter`, либо добавить offset после транскрипции

**Текущая конфигурация**:
- В `docker-compose.yml` используется `ASR_ENGINE=whisperx`, поэтому `faster_whisper` не активен
- Но если переключиться на `faster_whisper`, проблема может проявиться

---

### 1.3. Анализ WhisperX движка

**СТАТУС**: ⚠️ **ПРОБЛЕМА ВОЗМОЖНА**

**Место в коде**: `app/asr_models/mbain_whisperx_engine.py:41-93`

**Особенности WhisperX**:
1. WhisperX имеет встроенный VAD (Voice Activity Detection)
2. WhisperX может автоматически обрезать тишину в начале аудио
3. После выравнивания (`whisperx.align()`) таймкоды могут начинаться с 0.0, даже если в начале была тишина

**Проблема**:
- WhisperX внутренне может обрезать начальную тишину
- Модель ставит первый сегмент относительно "первой уверенной речи" → 0.0
- Нет механизма для добавления offset к таймкодам

**Текущая реализация**:
```python
result = self.model['whisperx'].transcribe(audio, **options_dict)
# ... align и diarize ...
# Нет добавления offset к таймкодам
```

---

## Возможные решения

### Решение 1: Добавление offset после транскрипции

**Место реализации**: В методах `transcribe()` движков (`faster_whisper_engine.py` и `mbain_whisperx_engine.py`)

**Подход**:
1. После получения `result` от модели
2. Вычислить `initial_silence` (время до первой речи)
3. Добавить offset ко всем таймкодам сегментов

**Где добавить**:

**Для Faster Whisper** (`app/asr_models/faster_whisper_engine.py:60`):
```python
result = {"language": ..., "segments": segments, "text": text}
# ДОБАВИТЬ ЗДЕСЬ:
# offset = calculate_initial_silence(audio)  # или из параметров
# for seg in result["segments"]:
#     seg.start += offset
#     seg.end += offset
```

**Для WhisperX** (`app/asr_models/mbain_whisperx_engine.py:87`):
```python
result = whisperx.assign_word_speakers(diarize_segments, result)
result["language"] = language
# ДОБАВИТЬ ЗДЕСЬ:
# offset = calculate_initial_silence(audio)  # или из параметров
# for seg in result["segments"]:
#     seg["start"] += offset
#     seg["end"] += offset
#     if "words" in seg:
#         for word in seg["words"]:
#             word["start"] += offset
#             word["end"] += offset
```

### Решение 2: Вычисление initial_silence из аудио

**Реализация функции**:
```python
def calculate_initial_silence(audio_array, sample_rate=16000, silence_threshold=0.01, min_speech_duration=0.2):
    """
    Вычисляет время начальной тишины в аудио.
    
    Parameters:
    - audio_array: numpy array с аудио данными
    - sample_rate: частота дискретизации (по умолчанию 16000)
    - silence_threshold: порог RMS для определения тишины (0.01 = 1% от максимума)
    - min_speech_duration: минимальная длительность речи для фиксации начала (секунды)
    
    Returns:
    - offset: время в секундах до начала речи
    """
    # Вычислить RMS по окнам
    window_size = int(sample_rate * 0.1)  # 100ms окна
    rms_values = []
    
    for i in range(0, len(audio_array) - window_size, window_size):
        window = audio_array[i:i+window_size]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)
    
    # Найти порог тишины
    max_rms = max(rms_values) if rms_values else 0.01
    threshold = max_rms * silence_threshold
    
    # Найти первое окно выше порога
    speech_start_window = None
    for i, rms in enumerate(rms_values):
        if rms > threshold:
            speech_start_window = i
            break
    
    if speech_start_window is None:
        return 0.0
    
    # Вернуть время в секундах
    offset = (speech_start_window * window_size) / sample_rate
    return max(0.0, offset - min_speech_duration)
```

**Место добавления**: В `app/utils.py` или в каждом движке отдельно

### Решение 3: Параметр offset из запроса

**Добавить параметр в эндпоинт** (`app/webservice.py:55-90`):
```python
initial_offset: Union[float, None] = Query(
    default=None,
    description="Initial silence offset in seconds to add to all timestamps"
)
```

**Передать в движок**:
```python
result = asr_model.transcribe(
    load_audio(audio_file.file, encode),
    task,
    language,
    initial_prompt,
    vad_filter,
    word_timestamps,
    {
        "diarize": diarize,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "initial_offset": initial_offset  # <-- НОВЫЙ ПАРАМЕТР
    },
    output,
)
```

**Использовать в движках**: Добавить логику добавления offset в методы `transcribe()`

---

## Рекомендации по реализации

### Приоритет 1: Отключение VAD (если включен)

**Для Faster Whisper**:
- Убедиться, что `vad_filter=False` по умолчанию (уже так)
- Если используется `vad_filter=True`, это может обрезать тишину

**Для WhisperX**:
- WhisperX имеет встроенный VAD, который нельзя отключить через параметры
- Нужно добавлять offset после транскрипции

### Приоритет 2: Добавление offset к таймкодам

**Вариант A: Автоматическое вычисление**
- Вычислять `initial_silence` из аудио автоматически
- Добавлять offset ко всем таймкодам

**Вариант B: Ручной параметр**
- Добавить параметр `initial_offset` в API
- Позволить клиенту передавать известное значение offset

**Вариант C: Комбинированный**
- Если `initial_offset` передан - использовать его
- Если нет - вычислять автоматически

### Приоритет 3: Обработка word-level timestamps

Если используется `word_timestamps=True`, нужно также добавлять offset к таймкодам слов:
- В Faster Whisper: `segment.words[].start` и `segment.words[].end`
- В WhisperX: `segment["words"][]["start"]` и `segment["words"][]["end"]`

---

## Структура данных сегментов

### Faster Whisper
**Тип данных**: Объекты с атрибутами
```python
segment.start  # float, секунды
segment.end    # float, секунды
segment.text   # str
segment.words  # список объектов (если word_timestamps=True)
```

**Использование в коде**:
- `app/utils.py:83-84`: `segment.start`, `segment.end`
- `app/utils.py:45`: `format_timestamp(segment.start)`

### WhisperX
**Тип данных**: Словари
```python
segment["start"]  # float, секунды
segment["end"]    # float, секунды
segment["text"]   # str
segment["words"]  # список словарей (если есть выравнивание)
```

**Использование в коде**:
- WhisperX использует свои `ResultWriter` из библиотеки `whisperx.utils`
- Структура данных передается напрямую в writer'ы

### Важно для реализации offset

**Faster Whisper**:
- Нужно модифицировать атрибуты объектов: `segment.start += offset`
- Если есть `word_timestamps`, также: `word.start += offset`, `word.end += offset`

**WhisperX**:
- Нужно модифицировать словари: `segment["start"] += offset`
- Если есть `words`, также: `word["start"] += offset`, `word["end"] += offset`
- Если есть диаризация, структура может быть сложнее

---

## Выводы

1. ✅ **FFmpeg декодирование**: Проблемы нет, аудио декодируется полностью
2. ⚠️ **VAD в Faster Whisper**: Может обрезать тишину, если `vad_filter=True`
3. ⚠️ **WhisperX**: Встроенный VAD может обрезать начальную тишину
4. ✅ **Решение**: Добавить механизм offset к таймкодам после транскрипции

**Рекомендуемый подход**:
1. Добавить функцию `calculate_initial_silence()` в `app/utils.py`
2. Модифицировать методы `transcribe()` в обоих движках для добавления offset
3. Опционально: добавить параметр `initial_offset` в API для ручной передачи
4. Учесть различия в структуре данных между Faster Whisper (объекты) и WhisperX (словари)

## Детали реализации offset

### Место добавления offset для Faster Whisper

**Файл**: `app/asr_models/faster_whisper_engine.py`
**Строка**: После строки 60, перед строкой 62

```python
result = {"language": ..., "segments": segments, "text": text}

# ДОБАВИТЬ OFFSET ЗДЕСЬ
if options and options.get("initial_offset"):
    offset = options["initial_offset"]
elif options and options.get("auto_calculate_offset", False):
    offset = calculate_initial_silence(audio)
else:
    offset = 0.0

if offset > 0:
    for seg in result["segments"]:
        seg.start += offset
        seg.end += offset
        if hasattr(seg, 'words') and seg.words:
            for word in seg.words:
                word.start += offset
                word.end += offset

output_file = StringIO()
```

### Место добавления offset для WhisperX

**Файл**: `app/asr_models/mbain_whisperx_engine.py`
**Строка**: После строки 87, перед строкой 89

```python
result = whisperx.assign_word_speakers(diarize_segments, result)
result["language"] = language

# ДОБАВИТЬ OFFSET ЗДЕСЬ
if options and options.get("initial_offset"):
    offset = options["initial_offset"]
elif options and options.get("auto_calculate_offset", False):
    offset = calculate_initial_silence(audio)
else:
    offset = 0.0

if offset > 0:
    for seg in result["segments"]:
        seg["start"] += offset
        seg["end"] += offset
        if "words" in seg and seg["words"]:
            for word in seg["words"]:
                word["start"] += offset
                word["end"] += offset

output_file = StringIO()
```

### Функция calculate_initial_silence

**Файл**: `app/utils.py`
**Место**: После функции `load_audio()` (после строки 127)

```python
def calculate_initial_silence(audio_array, sample_rate=16000, silence_threshold=0.01, min_speech_duration=0.2):
    """
    Calculate initial silence duration in audio.
    
    Parameters:
    - audio_array: numpy array with audio data (float32, normalized)
    - sample_rate: sample rate (default 16000)
    - silence_threshold: RMS threshold for silence detection (0.01 = 1% of max)
    - min_speech_duration: minimum speech duration to confirm start (seconds)
    
    Returns:
    - offset: time in seconds until speech starts
    """
    import numpy as np
    
    if len(audio_array) == 0:
        return 0.0
    
    # Calculate RMS in windows
    window_size = int(sample_rate * 0.1)  # 100ms windows
    if window_size > len(audio_array):
        return 0.0
    
    rms_values = []
    for i in range(0, len(audio_array) - window_size, window_size):
        window = audio_array[i:i+window_size]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)
    
    if not rms_values:
        return 0.0
    
    # Find silence threshold
    max_rms = max(rms_values)
    threshold = max_rms * silence_threshold
    
    # Find first window above threshold
    speech_start_window = None
    consecutive_speech_windows = 0
    required_windows = int(min_speech_duration * 10)  # 100ms windows
    
    for i, rms in enumerate(rms_values):
        if rms > threshold:
            consecutive_speech_windows += 1
            if consecutive_speech_windows >= required_windows:
                speech_start_window = i - required_windows + 1
                break
        else:
            consecutive_speech_windows = 0
    
    if speech_start_window is None:
        return 0.0
    
    # Return time in seconds
    offset = (speech_start_window * window_size) / sample_rate
    return max(0.0, offset)
```

### Модификация API эндпоинта

**Файл**: `app/webservice.py`
**Строка**: Добавить параметр после строки 89

```python
initial_offset: Union[float, None] = Query(
    default=None,
    description="Initial silence offset in seconds to add to all timestamps. If not provided, will be calculated automatically if auto_calculate_offset=true"
),
auto_calculate_offset: bool = Query(
    default=False,
    description="Automatically calculate initial silence offset from audio"
),
```

**Строка**: Модифицировать вызов transcribe (строка 91-99)

```python
result = asr_model.transcribe(
    load_audio(audio_file.file, encode),
    task,
    language,
    initial_prompt,
    vad_filter,
    word_timestamps,
    {
        "diarize": diarize,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "initial_offset": initial_offset,
        "auto_calculate_offset": auto_calculate_offset
    },
    output,
)
```

