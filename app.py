import os
import logging
# Suppress TensorFlow logging and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from mtcnn import MTCNN
from io import BytesIO
import tempfile, math, colorsys, pathlib, requests
try:
    from moviepy.editor import VideoFileClip, AudioFileClip
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False
from pillow_heif import register_heif_opener
register_heif_opener()

st.set_page_config(page_title="AutoBlur AI", page_icon="🕵️", layout="wide")

_css = pathlib.Path(__file__).parent / "style.css"
if _css.exists():
    st.markdown(f"<style>{_css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ── Emoji sticker list (Twemoji / iPhone style) ─────────────────────────────
# Value: emoji character (downloaded from Twemoji CDN) or "glitch" (PIL-drawn)
STICKERS = {
    "⚡ Глитч":       "glitch",
    "❤️ Красное":     "❤️",
    "🧡 Оранжевое":   "🧡",
    "💛 Желтое":      "💛",
    "💚 Зеленое":     "💚",
    "💙 Синее":       "💙",
    "💜 Фиолетовое":  "💜",
    "🖤 Черное":      "🖤",
    "🤍 Белое":       "🤍",
    "🤎 Коричневое":  "🤎",
    "💖 Сияющее":     "💖",
    "💗 Растущее":    "💗",
    "💓 Биение":      "💓",
    "💞 Вращающиеся": "💞",
    "💕 Два сердца":  "💕",
}
METHODS = ["🌀 Размытие", "🟦 Пикселизация", "🎭 Стикер"]

# ── Twemoji emoji image downloader ───────────────────────────────────────────
_EMOJI_CACHE_DIR = pathlib.Path(tempfile.gettempdir()) / "autoblur_twemoji"
_EMOJI_CACHE_DIR.mkdir(exist_ok=True)

@st.cache_data(show_spinner=False)
def get_twemoji(emoji_char: str):
    """Download a Twemoji PNG (72×72) from CDN and return as RGBA PIL Image."""
    # Build codepoint string, skipping variation selector U+FE0F
    cp = "-".join(f"{ord(c):x}" for c in emoji_char if ord(c) != 0xFE0F)
    cache = _EMOJI_CACHE_DIR / f"{cp}.png"
    if cache.exists():
        return Image.open(cache).convert("RGBA")
    url = f"https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/72x72/{cp}.png"
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content)).convert("RGBA")
            img.save(cache)
            return img
    except Exception:
        pass
    return None

# ── Only glitch kept as PIL-drawn sticker ────────────────────────────────────
def _sticker_glitch(w, h, roi_bgr=None):
    if roi_bgr is not None and roi_bgr.size > 0:
        rgb = cv2.cvtColor(cv2.resize(roi_bgr,(w,h)), cv2.COLOR_BGR2RGB)
        arr = np.dstack([rgb, np.full((h,w),240,np.uint8)])
        sh = max(4, w//10)
        arr[:, sh:, 0] = arr[:, :-sh, 0].copy()
        arr[:, :-sh, 2] = arr[:, sh:, 2].copy()
        arr[::3,:,:3] = np.clip(arr[::3,:,:3].astype(int)+60,0,255)
        return Image.fromarray(arr.astype(np.uint8), "RGBA")
    # Fallback neon
    img = Image.new("RGBA", (w, h), (4, 4, 18, 245))
    d = ImageDraw.Draw(img)
    for i, col in enumerate([(0,229,255),(168,85,247),(236,72,153)]):
        f = 1 - i*0.28
        rx2, ry2 = int(w/2*f), int(h/2*f)
        if rx2>3 and ry2>3:
            d.ellipse([w//2-rx2,h//2-ry2,w//2+rx2,h//2+ry2],
                      outline=(*col,210), width=max(2,min(w,h)//12))
    return img





# ── Core face effect engine ───────────────────────────────────────────────────
def _safe(x,y,w,h,iw,ih):
    x=max(0,x); y=max(0,y); w=min(iw-x,w); h=min(ih-y,h)
    return x,y,w,h

def apply_effect(img_cv, x,y,w,h, method, strength, sticker_key):
    ih, iw = img_cv.shape[:2]
    x,y,w,h = _safe(x,y,w,h,iw,ih)
    if w<=0 or h<=0: return img_cv
    cx, cy = x+w//2, y+h//2
    rx = int(w*0.58); ry = int(h*0.65)
    bx1=max(0,cx-rx); by1=max(0,cy-ry)
    bx2=min(iw,cx+rx); by2=min(ih,cy+ry)
    bw,bh = bx2-bx1, by2-by1
    if bw<=0 or bh<=0: return img_cv

    # Ellipse mask
    mask = np.zeros((ih,iw), np.uint8)
    cv2.ellipse(mask,(cx,cy),(rx,ry),0,0,360,255,-1)
    m_roi = mask[by1:by2, bx1:bx2]

    if "Размытие" in method:
        k = strength if strength%2!=0 else strength+1
        roi = img_cv[by1:by2, bx1:bx2].copy()
        proc = cv2.GaussianBlur(roi,(k,k),0)
        m3 = cv2.cvtColor(m_roi, cv2.COLOR_GRAY2BGR)
        img_cv[by1:by2, bx1:bx2] = np.where(m3==255, proc, roi)

    elif "Пикселизация" in method:
        roi = img_cv[by1:by2, bx1:bx2].copy()
        blk = max(2, min(bw,bh,strength))
        tmp = cv2.resize(roi,(max(1,bw//blk),max(1,bh//blk)),interpolation=cv2.INTER_LINEAR)
        proc = cv2.resize(tmp,(bw,bh),interpolation=cv2.INTER_NEAREST)
        m3 = cv2.cvtColor(m_roi, cv2.COLOR_GRAY2BGR)
        img_cv[by1:by2, bx1:bx2] = np.where(m3==255, proc, roi)

    elif "Стикер" in method and sticker_key:
        emoji_char = STICKERS.get(sticker_key, "")

        if emoji_char == "glitch":
            roi_bgr = img_cv[by1:by2, bx1:bx2].copy()
            patch = _sticker_glitch(bw, bh, roi_bgr)
            # Elliptical mask
            pmask = Image.new("L",(bw,bh),0)
            ImageDraw.Draw(pmask).ellipse([0,0,bw-1,bh-1], fill=255)
            patch.putalpha(pmask)
            base_pil = Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB)).convert("RGBA")
            base_pil.paste(patch,(bx1,by1),patch)
            img_cv = cv2.cvtColor(np.array(base_pil.convert("RGB")),cv2.COLOR_RGB2BGR)

        elif emoji_char:  # Twemoji iPhone emoji
            em_img = get_twemoji(emoji_char)
            if em_img is not None:
                # Size emoji to fit tightly over the face ellipse
                size = max(rx*2, ry*2)
                em_resized = em_img.resize((size, size), Image.LANCZOS)
                base_pil = Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB)).convert("RGBA")
                px = max(0, cx - size//2)
                py = max(0, cy - size//2)
                base_pil.paste(em_resized, (px, py), em_resized)
                img_cv = cv2.cvtColor(np.array(base_pil.convert("RGB")),cv2.COLOR_RGB2BGR)

    return img_cv


# ── MTCNN detector ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_detector(): return MTCNN()

def detect_faces(pil_img, conf=0.70):
    # Performance Optimization: Downscale for detection if too large
    max_dim = 960
    w, h = pil_img.size
    scale = 1.0
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        det_img = pil_img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    else:
        det_img = pil_img

    arr = np.array(det_img.convert("RGB"))
    dets = [d for d in get_detector().detect_faces(arr) if d["confidence"] >= conf]
    
    # Scale coordinates back
    if scale != 1.0:
        for d in dets:
            x, y, w_d, h_d = d["box"]
            d["box"] = [int(x / scale), int(y / scale), int(w_d / scale), int(h_d / scale)]
            if "keypoints" in d:
                for k in d["keypoints"]:
                    kx, ky = d["keypoints"][k]
                    d["keypoints"][k] = (int(kx / scale), int(ky / scale))
    return dets


# ── AI Quality Enhancer ───────────────────────────────────────────────────────
def enhance_frame(img_cv):
    """Professional enhancement pipeline: Denoise -> Sharpen -> Detail."""
    # 1. Denoise (Mild to preserve details)
    denoised = cv2.fastNlMeansDenoisingColored(img_cv, None, 7, 7, 5, 15)
    # 2. Detail Enhancement
    detailed = cv2.detailEnhance(denoised, sigma_s=10, sigma_r=0.15)
    # 3. Unsharp Mask (Contrast improvement)
    gaussian = cv2.GaussianBlur(detailed, (0, 0), 2.0)
    enhanced = cv2.addWeighted(detailed, 1.4, gaussian, -0.4, 0)
    return enhanced


# ── Preview: numbered ellipses ────────────────────────────────────────────────
def draw_boxes(pil_img, dets):
    img = pil_img.copy().convert("RGBA")
    ov  = Image.new("RGBA", img.size, (0,0,0,0))
    d   = ImageDraw.Draw(ov)
    for i,det in enumerate(dets):
        x,y,w,h = det["box"]
        x=max(0,x); y=max(0,y)
        cx,cy = x+w//2, y+h//2
        rx,ry = int(w*.58), int(h*.65)
        d.ellipse([cx-rx,cy-ry,cx+rx,cy+ry], outline=(78,205,196,220), width=3)
        d.rectangle([cx-13,cy-ry-22,cx+13,cy-ry-4], fill=(78,205,196,210))
        d.text((cx-9,cy-ry-21), f"#{i+1}", fill=(0,0,0,255))
    return Image.alpha_composite(img,ov).convert("RGB")


# ── Process image ─────────────────────────────────────────────────────────────
def process_image(pil_img, dets, selected, cfgs):
    img_cv = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    for i,det in enumerate(dets):
        if not selected[i]: continue
        x,y,w,h = det["box"]
        img_cv = apply_effect(img_cv,x,y,w,h, cfgs[i]["m"], cfgs[i]["s"], cfgs[i]["e"])
    return Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB))


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Настройки")
        st.markdown("---")
        st.markdown("#### Точность детекции")
        conf = st.slider("Мин. уверенность", 0.50, 0.99, 0.70, 0.05)
        st.markdown("---")
        st.markdown("#### Глобальный эффект")
        st.caption("Для видео и значение по умолчанию.")
        gm = st.selectbox("Метод", METHODS)
        if "Размытие" in gm:   gs=st.slider("Интенсивность",5,99,51,2); ge=None
        elif "Пикселизация" in gm: gs=st.slider("Размер блока",3,50,12); ge=None
        else:
            gs=20
            ge = st.selectbox("Стикер", list(STICKERS.keys()))
        st.markdown("---")
        st.markdown("#### ✨ Улучшение")
        enhance = st.toggle("AI Enhancer", value=False, help="Улучшает качество, удаляет шум и делает картинку четче.")
        st.markdown("---")
        st.markdown("#### ⚡ Режим работы")
        speed_mode = st.radio("Скорость", ["Качество", "Турбо"], 
                              help="Турбо режим ускоряет видео в 3 раза, пропуская детекцию в промежуточных кадрах.")
    return gm, gs, ge, conf, speed_mode == "Турбо", enhance


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB: IMAGE
# ═══════════════════════════════════════════════════════════════════════════════
def tab_image(gm, gs, ge, conf, turbo, enhance):
    up = st.file_uploader("", type=["jpg","jpeg","png","webp","bmp","heic"], key="img_up",
                          label_visibility="collapsed")
    if not up:
        st.markdown(
            '<div style="text-align:center;padding:3rem;border:1.5px dashed rgba(255,255,255,.1);'
            'border-radius:14px;color:#64748B"><div style="font-size:2.5rem">🖼️</div>'
            '<p>JPG · PNG · WEBP · BMP · HEIC</p></div>', unsafe_allow_html=True)
        return

    try:
        img = Image.open(up)
    except Exception as e:
        st.error(f"❌ Ошибка открытия файла: {up.name}. Пожалуйста, используйте стандартные форматы (JPG, PNG).")
        return

    # Enhancement
    if enhance:
        with st.spinner("✨ Улучшение качества…"):
            img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            img_cv = enhance_frame(img_cv)
            img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    with st.spinner("🔍 Обнаружение лиц…"):
        dets = detect_faces(img, conf)

    if not dets:
        st.warning("Лица не обнаружены. Снизьте порог точности или загрузите другое фото.")
        st.image(img, use_container_width=True)
        return

    n = len(dets)
    c1,c2,c3 = st.columns(3)
    c1.metric("👥 Лиц", n)
    c2.metric("🖼 Размер", f"{img.width}×{img.height}")
    c3.metric("🎯 Уверенность", f"{max(d['confidence'] for d in dets):.0%}")

    st.image(draw_boxes(img,dets), caption="Обнаруженные лица", use_container_width=True)
    st.markdown("### 🎯 Настройки для каждого лица")

    selected, cfgs = [], []
    cols_n = min(n, 4)
    for row_s in range(0, n, cols_n):
        cols = st.columns(cols_n)
        for ci in range(cols_n):
            fi = row_s+ci
            if fi >= n: break
            det = dets[fi]
            fx,fy,fw,fh = det["box"]
            fx=max(0,fx); fy=max(0,fy)
            with cols[ci]:
                pad = max(12, min(fw,fh)//4)
                thumb = img.crop((max(0,fx-pad),max(0,fy-pad),
                                  min(img.width,fx+fw+pad),min(img.height,fy+fh+pad)))
                st.image(thumb, width=130)
                sel = st.checkbox(f"Скрыть #{fi+1}", value=True, key=f"sel_{fi}")
                selected.append(sel)
                if sel:
                    m = st.selectbox("Метод", METHODS,
                                     index=METHODS.index(gm), key=f"m_{fi}")
                    if "Размытие" in m:
                        s=st.slider("Сила",5,99,51,2,key=f"s_{fi}"); e=None
                    elif "Пикселизация" in m:
                        s=st.slider("Блок",3,50,12,key=f"s_{fi}"); e=None
                    else:
                        s=20
                        def_e = ge if ge and ge in STICKERS else list(STICKERS.keys())[0]
                        e=st.selectbox("Стикер",list(STICKERS.keys()),
                                       index=list(STICKERS.keys()).index(def_e),key=f"e_{fi}")
                    cfgs.append({"m":m,"s":s,"e":e})
                else:
                    cfgs.append({"m":gm,"s":gs,"e":ge})

    st.markdown("---")
    if st.button("🚀 Применить", type="primary", use_container_width=True):
        with st.spinner("✨ Анонимизация…"):
            result = process_image(img, dets, selected, cfgs)

        col1,col2 = st.columns(2)
        with col1:
            st.subheader("📷 Оригинал")
            st.image(img, use_container_width=True)
        with col2:
            st.subheader("🔒 Результат")
            st.image(result, use_container_width=True)

        # High-quality JPEG download (98 quality = near-lossless, smaller file)
        buf_jpg = BytesIO()
        result.convert("RGB").save(buf_jpg, format="JPEG", quality=98, subsampling=0)
        # Lossless PNG download
        buf_png = BytesIO()
        result.save(buf_png, format="PNG")
        dc1, dc2 = st.columns(2)
        with dc1:
            st.download_button("⬇️ Скачать JPEG (высокое качество)",
                               buf_jpg.getvalue(), "autoblur.jpg", "image/jpeg",
                               use_container_width=True)
        with dc2:
            st.download_button("⬇️ Скачать PNG (без потерь)",
                               buf_png.getvalue(), "autoblur.png", "image/png",
                               use_container_width=True)
        st.success(f"✅ Скрыто {sum(selected)} из {n} лиц")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB: VIDEO
# ═══════════════════════════════════════════════════════════════════════════════
def tab_video(gm, gs, ge, conf, turbo, enhance):
    st.info("⏱ MTCNN обрабатывает каждый кадр. Длинные видео займут несколько минут.")
    up = st.file_uploader("", type=["mp4","avi","mov","mkv"],
                          key="vid_up", label_visibility="collapsed")
    if not up:
        st.markdown(
            '<div style="text-align:center;padding:3rem;border:1.5px dashed rgba(255,255,255,.1);'
            'border-radius:14px;color:#64748B"><div style="font-size:2.5rem">🎬</div>'
            '<p>MP4 · AVI · MOV · MKV</p></div>', unsafe_allow_html=True)
        return

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tf.write(up.read()); tf.flush(); tf.close()
    st.video(tf.name)

    c1, c2 = st.columns(2)
    v_upscale = c1.toggle("🚀 Upscale (HD+)", value=False, help="Увеличивает разрешение видео в 1.5 раза для четкости.")
    v_high_q  = c2.toggle("💎 Ultra Quality", value=True, help="Использует высокий битрейт для сохранения всех деталей.")

    if st.button("🚀 Обработать видео", type="primary", use_container_width=True):
        cap = cv2.VideoCapture(tf.name)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh_v  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if v_upscale:
            fw = int(fw * 1.5)
            fh_v = int(fh_v * 1.5)

        out_path = tf.name.replace(".mp4","_out.mp4")
        # Try H264 first for browser playback
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (fw,fh_v))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (fw,fh_v))

        det_obj = get_detector()
        prog = st.progress(0, text="⏳ Обработка…")
        n_frame = 0
        last_faces = []

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Optimization: Frame skipping in Turbo mode
            if not turbo or n_frame % 3 == 0:
                # Downscale frame for detection
                h_f, w_f = frame.shape[:2]
                max_dim = 640 if turbo else 800
                scale = 1.0
                if max(w_f, h_f) > max_dim:
                    scale = max_dim / max(w_f, h_f)
                    det_frame = cv2.resize(frame, (int(w_f * scale), int(h_f * scale)))
                else:
                    det_frame = frame
                
                rgb = cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB)
                current_faces = [d for d in det_obj.detect_faces(rgb) if d["confidence"] >= conf]
                
                # Scale back
                if scale != 1.0:
                    for d in current_faces:
                        x, y, w_d, h_d = d["box"]
                        d["box"] = [int(x / scale), int(y / scale), int(w_d / scale), int(h_d / scale)]
                last_faces = current_faces
            
            for d in last_faces:
                x,y,w,h = d["box"]
                frame = apply_effect(frame,x,y,w,h,gm,gs,ge)
            
            # Upscale frame if requested
            if v_upscale:
                frame = cv2.resize(frame, (fw, fh_v), interpolation=cv2.INTER_LANCZOS4)
            
            # Post-processing Enhancement (Video)
            if enhance:
                # To keep video performance, we only apply a faster version
                frame = cv2.detailEnhance(frame, sigma_s=5, sigma_r=0.1)

            writer.write(frame)
            n_frame += 1
            if total > 0 and n_frame % 5 == 0:
                prog.progress(min(n_frame/total, 1.0), text=f"Кадр {n_frame}/{total}")

        cap.release(); writer.release()
        
        # Audio Preservation Step
        final_out = out_path.replace("_out.mp4", "_final.mp4")
        if HAS_MOVIEPY:
            try:
                with st.spinner("🎵 Сохранение аудио…"):
                    # Use moviepy to add audio from original tf.name to out_path
                    orig_clip = VideoFileClip(tf.name)
                    if orig_clip.audio is not None:
                        processed_clip = VideoFileClip(out_path)
                        final_clip = processed_clip.set_audio(orig_clip.audio)
                        # High quality bitrate logic
                        br = "8000k" if v_high_q else "3000k"
                        final_clip.write_videofile(final_out, codec="libx264", audio_codec="aac", 
                                                   bitrate=br, verbose=False, logger=None)
                        processed_path = final_out
                    else:
                        processed_path = out_path
                    orig_clip.close()
            except Exception as e:
                st.warning(f"⚠️ Не удалось сохранить аудио: {e}")
                processed_path = out_path
        else:
            processed_path = out_path

        prog.progress(1.0, text="✅ Готово!")

        with open(processed_path,"rb") as vf:
            vbytes = vf.read()

        st.subheader("🎬 Результат")
        st.video(vbytes)
        st.download_button("⬇️ Скачать видео", vbytes,
                           "autoblur_video.mp4","video/mp4", use_container_width=True)
        st.success("✅ Видео обработано!")

        try: os.unlink(tf.name)
        except: pass
        try: os.unlink(out_path)
        except: pass
        if processed_path == final_out:
            try: os.unlink(final_out)
            except: pass


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown(
        '<h1>🕵️ AutoBlur AI</h1>'
        '<p style="color:#64748B;margin-bottom:1.5rem">Профессиональная анонимизация лиц '
        '· <b style="color:#4ECDC4">MTCNN Deep Learning</b></p>',
        unsafe_allow_html=True)

    gm, gs, ge, conf, turbo, enhance = sidebar()
    t1, t2 = st.tabs(["🖼️  Изображение", "🎬  Видео"])
    with t1: tab_image(gm, gs, ge, conf, turbo, enhance)
    with t2: tab_video(gm, gs, ge, conf, turbo, enhance)

if __name__ == "__main__":
    main()
