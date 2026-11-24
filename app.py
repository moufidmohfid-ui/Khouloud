import os
import time
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# إعداد المجلدات
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# يمكنك تغيير النموذج إلى gemini-3.6-flash أو gemini-4-hydrogen أو gemini-3.5-flash-lite
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

app = FastAPI(title="HookSum AI - ملخص الفيديو بالهوك")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


# نموذج البيانات المهيكلة التي نطلبها من الذكاء الاصطناعي
class SummaryClip(BaseModel):
    start_seconds: float = Field(description="وقت بداية المقطع بالثواني")
    end_seconds: float = Field(description="وقت نهاية المقطع بالثواني")
    title: str = Field(description="عنوان قصير وجذاب للمقطع بالعربية")
    description: str = Field(description="شرح مختصر لما يحدث في المقطع بالعربية")


class VideoAnalysisResult(BaseModel):
    video_title: str = Field(description="عنوان مقترح وجذاب للفيديو الملخص")
    hook_caption: str = Field(description="جملة هوك قوية ومثيرة للفضول تظهر على الشاشة في أول 3 ثوانٍ")
    hook_reason: str = Field(description="لماذا تم اختيار هذا المشهد كهوك جاذب للأنظار")
    hook_start_seconds: float = Field(description="وقت بداية مشهد الهوك بالثواني")
    hook_end_seconds: float = Field(description="وقت نهاية مشهد الهوك بالثواني (يفضل 4 إلى 8 ثوانٍ)")
    overall_summary: str = Field(description="ملخص نصي شامل ومكثف لمحتوى الفيديو بالكامل")
    key_clips: List[SummaryClip] = Field(description="قائمة مرتبة بأهم 3 إلى 6 مشاهد تلخص الفيديو")


def has_audio_stream(video_path: Path) -> bool:
    """التحقق من وجود مسار صوتي في الفيديو لتجنب أخطاء FFmpeg"""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video_path)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return bool(res.stdout.strip())
    except FileNotFoundError:
        return False


def create_hook_summary_video(source_path: Path, analysis: VideoAnalysisResult, output_path: Path) -> bool:
    """استخدام FFmpeg لقص مشهد الهوك ووضعه في البداية ثم دمج مشاهد التلخيص"""
    if shutil.which("ffmpeg") is None:
        return False

    job_tmp = OUTPUT_DIR / f"tmp_{uuid.uuid4().hex[:8]}"
    job_tmp.mkdir(exist_ok=True)
    audio_exists = has_audio_stream(source_path)

    # تجهيز قائمة المقاطع: أولاً مشهد الهوك، ثم مقاطع التلخيص
    segments = [(analysis.hook_start_seconds, analysis.hook_end_seconds)]
    for clip in analysis.key_clips:
        if clip.end_seconds > clip.start_seconds:
            segments.append((clip.start_seconds, clip.end_seconds))

    segment_files = []
    try:
        for idx, (start, end) in enumerate(segments):
            duration = max(1.0, end - start)
            seg_file = job_tmp / f"seg_{idx}.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start:.2f}",
                "-i", str(source_path),
                "-t", f"{duration:.2f}",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            ]
            if audio_exists:
                cmd += ["-c:a", "aac", "-ar", "44100", "-ac", "2"]
            else:
                cmd += ["-an"]
            cmd.append(str(seg_file))
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            segment_files.append(seg_file)

        # دمج جميع المقاطع في فيديو واحد
        concat_list = job_tmp / "list.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for seg in segment_files:
                f.write(f"file '{seg.as_posix()}'\n")

        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list), "-c", "copy", str(output_path)
        ]
        subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_path.exists()
    except subprocess.CalledProcessError:
        return False
    finally:
        shutil.rmtree(job_tmp, ignore_errors=True)


@app.get("/", response_class=HTMLResponse)
async def serve_home():
    index_file = STATIC_DIR / "index.html"
    return index_file.read_text(encoding="utf-8")


@app.post("/api/analyze")
async def analyze_video(
    file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
):
    effective_key = (api_key or "").strip() or os.getenv("GEMINI_API_KEY")
    if not effective_key:
        raise HTTPException(status_code=400, detail="يرجى إدخال مفتاح Gemini API أو ضبط GEMINI_API_KEY في البيئة.")

    if not file and not youtube_url:
        raise HTTPException(status_code=400, detail="يرجى رفع ملف فيديو أو إدخال رابط يوتيوب.")

    client = genai.Client(api_key=effective_key)
    local_video_path: Optional[Path] = None
    video_url_for_player = ""
    uploaded_gemini_file = None

    prompt = """
    أنت خبير مونتاج وصانع محتوى محترف (YouTube & Reels Strategist).
    قم بتحليل هذا الفيديو الطويل بالكامل وأنشئ خطة تلخيص تبدأ بهوك قوي ومشهد يخطف الأنظار.

    المطلوب منك:
    1. اختر أقوى مشهد بصري أو لحظة ذروة/مفاجأة/سؤال صادم في الفيديو ليكون هو "الهوك" (Hook) في البداية (مدته بين 4 إلى 8 ثوانٍ).
    2. اكتب جملة هوك عربية قصيرة ومثيرة للفضول لتظهر في بداية الفيديو.
    3. لخّص الفيديو بالكامل في فقرة واضحة ومفيدة.
    4. اختر أهم 3 إلى 6 مقاطع زمنية مرتبة ترتيباً منطقياً تلخص الفكرة الكاملة للفيديو، مع كتابة عنوان ووصف لكل مقطع.
    تأكد من دقة الطوابع الزمنية (start_seconds و end_seconds) بالثواني.
    """

    try:
        if file and file.filename:
            ext = Path(file.filename).suffix or ".mp4"
            unique_name = f"{uuid.uuid4().hex}{ext}"
            local_video_path = UPLOAD_DIR / unique_name
            with open(local_video_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            video_url_for_player = f"/uploads/{unique_name}"

            # رفع الفيديو إلى Gemini Files API وانتظار اكتمال المعالجة
            uploaded_gemini_file = client.files.upload(file=str(local_video_path))
            while uploaded_gemini_file.state and uploaded_gemini_file.state.name == "PROCESSING":
                time.sleep(3)
                uploaded_gemini_file = client.files.get(name=uploaded_gemini_file.name)

            if uploaded_gemini_file.state and uploaded_gemini_file.state.name == "FAILED":
                raise HTTPException(status_code=500, detail="فشلت معالجة الفيديو في Gemini API.")

            contents = [uploaded_gemini_file, prompt]
        else:
            # تحليل رابط يوتيوب مباشرة عبر Gemini
            yt_url = youtube_url.strip()
            contents = [
                types.Part.from_uri(file_uri=yt_url, mime_type="video/mp4"),
                prompt,
            ]

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VideoAnalysisResult,
                temperature=0.4,
            ),
        )

        analysis = VideoAnalysisResult.model_validate_json(response.text)

        # إذا كان الفيديو مرفوعاً محلياً، نحاول إنتاج فيديو ممنتج جاهز للتحميل بـ FFmpeg
        exported_video_url = None
        if local_video_path:
            out_name = f"summary_{uuid.uuid4().hex[:10]}.mp4"
            out_path = OUTPUT_DIR / out_name
            if create_hook_summary_video(local_video_path, analysis, out_path):
                exported_video_url = f"/outputs/{out_name}"

        return {
            "status": "success",
            "source_type": "local" if local_video_path else "youtube",
            "video_url": video_url_for_player,
            "youtube_url": youtube_url,
            "exported_video_url": exported_video_url,
            "analysis": analysis.model_dump(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء التحليل: {str(e)}")
    finally:
        # تنظيف الملف من سحابة Gemini بعد الانتهاء
        if uploaded_gemini_file:
            try:
                client.files.delete(name=uploaded_gemini_file.name)
            except Exception:
                pass
