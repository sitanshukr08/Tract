from datetime import timedelta

def format_timestamp(seconds: float) -> str:
    millis = int(round((seconds % 1) * 1000))
    td = timedelta(seconds=int(seconds))
    return f"{str(td).zfill(8)},{millis:03d}"

def generate_srt(segments: list) -> str:
    srt = ""
    for i, seg in enumerate(segments):
        start = format_timestamp(seg['start'])
        end = format_timestamp(seg['end'])
        srt += f"{i+1}\n{start} --> {end}\n{seg['text'].strip()}\n\n"
    return srt

def coerce_segment(seg) -> dict:
    if isinstance(seg, dict):
        return {"start": float(seg["start"]), "end": float(seg["end"]), "text": str(seg.get("text", ""))}
    return {"start": float(seg.start), "end": float(seg.end), "text": str(seg.text)}