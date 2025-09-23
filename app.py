import os
import io
import glob
import yaml
import time
import platform
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from ultralytics import __version__ as yolo_version
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
from streamlit_image_select import image_select
import torch

# Base do app para construir caminhos robustos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Road Sign Detection • YOLO",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo premium dark (inspirado no app LSTM, paleta ajustada)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
:root {
  --primary: #00C2FF; /* ciano-azulado */
  --accent: #8A2BE2;  /* roxo vibrante */
  --danger: #FF6B6B;
  --dark-bg: #0E1117;
  --card-bg: #171A23;
  --text: #E6E6E6;
  --muted: #9AA4AF;
  --shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.stApp { background: var(--dark-bg); color: var(--text); }
.main .block-container { max-width: none !important; padding-left: 1rem; padding-right: 1rem; }

h1, h2, h3, h4, h5 { font-family: 'Inter', sans-serif; color: var(--text); }
.main-hero { font-size: 2.4rem; font-weight: 700; text-align: left; margin: 0.5rem 0 1.25rem; }
.main-hero .title-gradient { background: linear-gradient(135deg, var(--primary), var(--accent)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.subtitle { text-align: left; color: var(--muted); margin-bottom: 2rem; }
.card { background: var(--card-bg); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 1rem; box-shadow: var(--shadow); }
.metric { background: linear-gradient(135deg, rgba(0,194,255,0.10), rgba(138,43,226,0.10)); border: 1px solid rgba(0,194,255,0.25); border-radius: 10px; padding: 0.9rem; }
.badge { display: inline-block; padding: 0.2rem 0.5rem; border-radius: 999px; background: rgba(0,194,255,0.15); border: 1px solid rgba(0,194,255,0.25); color: var(--text); font-size: 0.75rem; }
.button-primary .st-emotion-cache-7ym5gk { background: linear-gradient(135deg, var(--primary), var(--accent)) !important; color: #fff !important; border: 0 !important; }
hr { border-top: 1px solid rgba(255,255,255,0.08); }

/* Ícone de semáforo (emoji fallback) */
.tl-wrap { display:inline-flex; align-items:center; gap: 12px; }
.tl-icon { width: 28px; height: 64px; border-radius: 8px; background: #0b0e14; border: 1px solid rgba(255,255,255,0.08); display:flex; flex-direction:column; justify-content:space-around; padding:6px; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03); }
.tl-dot { width: 14px; height: 14px; border-radius: 50%; box-shadow: 0 0 12px rgba(0,0,0,0.6); margin:auto; }
.tl-red { background: #ff4d4f; box-shadow: 0 0 10px rgba(255,77,79,0.8); }
.tl-yellow { background: #ffd666; box-shadow: 0 0 10px rgba(255,214,102,0.8); }
.tl-green { background: #52c41a; box-shadow: 0 0 10px rgba(82,196,26,0.8); }

/* Chips de classe */
.chips { display:flex; gap:6px; flex-wrap:wrap; }
.chip { padding: 2px 8px; border-radius: 999px; font-size: 12px; border: 1px solid rgba(255,255,255,0.15); }
.chip-traffic { background: rgba(0,194,255,0.12); border-color: rgba(0,194,255,0.35); }
.chip-stop { background: rgba(255,107,107,0.12); border-color: rgba(255,107,107,0.35); }
.chip-speed { background: rgba(138,43,226,0.12); border-color: rgba(138,43,226,0.35); }
.chip-cross { background: rgba(82,196,26,0.12); border-color: rgba(82,196,26,0.35); }
</style>
""",
    unsafe_allow_html=True,
)

ALLOWED_CLASSES = ["Traffic Light", "Stop", "Speedlimit", "Crosswalk"]
CLASS_COLORS = {
    "Traffic Light": (0, 194, 255),
    "Stop": (255, 107, 107),
    "Speedlimit": (138, 43, 226),
    "Crosswalk": (82, 196, 26),
}

@st.cache_resource(show_spinner=False)
def load_model() -> YOLO | None:
    candidate_paths = [
        os.path.join(BASE_DIR, "modelos", "best.pt"),
        os.path.join(BASE_DIR, "resultados", "runs", "detect", "train", "weights", "best.pt"),
        os.path.join(BASE_DIR, "modelos", "last.pt"),
        os.path.join(BASE_DIR, "resultados", "runs", "detect", "train", "weights", "last.pt"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                return YOLO(path)
            except Exception as e:
                st.warning(f"Falha ao carregar modelo em {path}: {e}")
    st.error("Nenhum arquivo de pesos YOLO encontrado. Coloque 'best.pt' em 'modelos/'.")
    return None

@st.cache_data(show_spinner=False)
def load_training_artifacts():
    base = os.path.join(BASE_DIR, "resultados", "runs", "detect", "train")
    images = {
        "results": os.path.join(base, "results.png"),
        "confusion": os.path.join(base, "confusion_matrix.png"),
        "confusion_norm": os.path.join(base, "confusion_matrix_normalized.png"),
        "labels": os.path.join(base, "labels.jpg"),
        "batches": sorted(glob.glob(os.path.join(base, "train_batch*.jpg")))[:6],
        "val_preds": sorted(glob.glob(os.path.join(base, "val_batch*_pred.jpg")))[:6],
    }
    csv_path = os.path.join(base, "results.csv")
    args_path = os.path.join(base, "args.yaml")
    results_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
    args = None
    if os.path.exists(args_path):
        with open(args_path, "r", encoding="utf-8") as f:
            try:
                args = yaml.safe_load(f)
            except Exception:
                args = None
    return images, results_df, args

@st.cache_data(show_spinner=False)
def load_dataset_info():
    dataset_yaml = os.path.join(BASE_DIR, "dados", "road_signs_dataset.yaml")
    annotations_csv = os.path.join(BASE_DIR, "dados", "road_signs_annotations.csv")
    data_cfg = None
    if os.path.exists(dataset_yaml):
        with open(dataset_yaml, "r", encoding="utf-8") as f:
            try:
                data_cfg = yaml.safe_load(f)
            except Exception:
                data_cfg = None
    ann_df = pd.read_csv(annotations_csv) if os.path.exists(annotations_csv) else None
    return data_cfg, ann_df


def get_env_status():
    gpu = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu else platform.processor() or "CPU"
    torch_ver = torch.__version__
    cuda_ver = torch.version.cuda if gpu else "-"
    return {
        "device": "GPU" if gpu else "CPU",
        "device_name": device_name,
        "torch": torch_ver,
        "cuda": cuda_ver,
        "ultralytics": yolo_version,
        "python": platform.python_version(),
    }


def show_status(model, data_cfg, results_df):
    st.markdown("### 📊 Status do Sistema")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
<div class="metric">
  <b>🧠 Modelo YOLO</b><br/>
  <span class="badge">{ 'Carregado' if model else 'Indisponível' }</span>
</div>
""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
<div class="metric">
  <b>📁 Dataset</b><br/>
  <span class="badge">{ 'OK' if data_cfg else 'Não encontrado' }</span>
</div>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
<div class="metric">
  <b>📈 Métricas de Treino</b><br/>
  <span class="badge">{ 'Disponíveis' if results_df is not None else 'Sem resultados' }</span>
</div>
""",
            unsafe_allow_html=True,
        )


def page_home(model, data_cfg, results_df):
    st.markdown(
        '<div class="main-hero tl-wrap">\
          <div class="tl-icon"><div class="tl-dot tl-red"></div><div class="tl-dot tl-yellow"></div><div class="tl-dot tl-green"></div></div>\
          <span class="title-gradient">Road Sign Detection • YOLO</span>\
        </div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="subtitle">Detecção de sinais de trânsito com modelo YOLO treinado no seu dataset</div>', unsafe_allow_html=True)
    show_status(model, data_cfg, results_df)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        num_classes = len(data_cfg.get("names", [])) if data_cfg else 0
        st.markdown(f"""
<div class="card">
  <h4>📚 Número de classes</h4>
  <div class="badge">{num_classes}</div>
</div>
""", unsafe_allow_html=True)
    with col2:
        if results_df is not None and "epoch" in results_df.columns:
            last_epoch = int(results_df["epoch"].max())
        else:
            last_epoch = 0
        st.markdown(f"""
<div class="card">
  <h4>⏱️ Épocas treinadas</h4>
  <div class="badge">{last_epoch}</div>
</div>
""", unsafe_allow_html=True)
    with col3:
        if results_df is not None and "metrics/mAP50-95(B)" in results_df.columns:
            map_val = float(results_df["metrics/mAP50-95(B)"].dropna().iloc[-1])
            map_text = f"{map_val:.3f}"
        else:
            map_text = "N/D"
        st.markdown(f"""
<div class="card">
  <h4>🎯 mAP50-95</h4>
  <div class="badge">{map_text}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ✨ Destaques do Treino")
    images, _, _ = load_training_artifacts()
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(images["results"]):
            st.image(images["results"], caption="Resultados de treino", use_container_width=True)
    with c2:
        if os.path.exists(images["confusion_norm"]):
            st.image(images["confusion_norm"], caption="Matriz de Confusão (normalizada)", use_container_width=True)


def yolo_predict(model: YOLO, image: Image.Image, conf: float, iou: float, imgsz: int):
    img_np = np.array(image.convert("RGB"))
    results = model.predict(img_np, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    return results


def _gather_example_images():
    # principal
    preferred = sorted(glob.glob(os.path.join(BASE_DIR, "dados", "image_examples", "*.jpg")))
    preferred += sorted(glob.glob(os.path.join(BASE_DIR, "dados", "image_examples", "*.png")))
    if preferred:
        return preferred[:12]
    # legado
    legacy = sorted(glob.glob(os.path.join(BASE_DIR, "dados", "examples", "*.jpg")))
    legacy += sorted(glob.glob(os.path.join(BASE_DIR, "dados", "examples", "*.png")))
    if legacy:
        return legacy[:12]
    # fallback final
    base = os.path.join(BASE_DIR, "resultados", "runs", "detect", "train")
    fallback = sorted(glob.glob(os.path.join(base, "val_batch*_pred.jpg")))
    if not fallback:
        fallback = sorted(glob.glob(os.path.join(base, "train_batch*.jpg")))
    return fallback[:12]


def _draw_boxes(image_np: np.ndarray, boxes_xyxy: np.ndarray, classes: np.ndarray, confs: np.ndarray, names: dict, include: list[str]) -> np.ndarray:
    out = image_np.copy()
    for i in range(len(classes)):
        name = names.get(int(classes[i]), str(int(classes[i])))
        if name not in include:
            continue
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        color = CLASS_COLORS.get(name, (0, 194, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {confs[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)
    return out


def page_detect(model):
    st.markdown("## 🔎 Detecção")
    st.markdown("Faça upload, use a câmera ou selecione um exemplo.")

    st.info("Este modelo detecta apenas: Traffic Light, Stop, Speedlimit, Crosswalk.")

    # Presets
    cols_p = st.columns(3)
    preset = st.session_state.get("preset", "Equilibrado")
    with cols_p[0]:
        if st.button("⚡ Rápido"):
            preset = "Rápido"
    with cols_p[1]:
        if st.button("⚖️ Equilibrado"):
            preset = "Equilibrado"
    with cols_p[2]:
        if st.button("🎯 Preciso"):
            preset = "Preciso"
    st.session_state["preset"] = preset

    # Valores padrão por preset
    if preset == "Rápido":
        conf_default, iou_default, size_default = 0.35, 0.45, 640
    elif preset == "Preciso":
        conf_default, iou_default, size_default = 0.15, 0.55, 960
    else:
        conf_default, iou_default, size_default = 0.25, 0.50, 768

    col1, col2, col3 = st.columns([2,1,1])
    with col2:
        conf = st.slider("Confiança mínima", 0.05, 0.95, conf_default, 0.05)
    with col3:
        iou = st.slider("IoU", 0.1, 0.9, iou_default, 0.05)
    imgsz = st.select_slider("Tamanho", options=[640, 704, 768, 832, 896, 960], value=size_default)

    # Filtros por classe
    selected_classes = st.multiselect("Filtrar classes", ALLOWED_CLASSES, default=ALLOWED_CLASSES)
    st.markdown('<div class="chips"><span class="chip chip-traffic">Traffic Light</span><span class="chip chip-stop">Stop</span><span class="chip chip-speed">Speedlimit</span><span class="chip chip-cross">Crosswalk</span></div>', unsafe_allow_html=True)

    tab_up, tab_cam, tab_ex, tab_batch = st.tabs(["Upload", "Câmera", "Exemplos", "Lote"])

    def run_and_show(pil_img: Image.Image, key_prefix: str = "single"):
        start = time.time()
        results = yolo_predict(model, pil_img, conf, iou, imgsz)
        latency = (time.time() - start) * 1000
        if results:
            r0 = results[0]
            names = r0.names
            xyxy = r0.boxes.xyxy.cpu().numpy() if r0.boxes is not None else np.empty((0, 4))
            cls = r0.boxes.cls.cpu().numpy().astype(int) if r0.boxes is not None else np.empty((0,))
            confs = r0.boxes.conf.cpu().numpy() if r0.boxes is not None else np.empty((0,))

            img_np = np.array(pil_img.convert("RGB"))
            annotated = _draw_boxes(img_np, xyxy, cls, confs, names, selected_classes)
            st.image(annotated, caption=f"Detecções ({latency:.1f} ms)", use_container_width=True)

            # Botão de download da imagem anotada
            buf = io.BytesIO()
            Image.fromarray(annotated).save(buf, format="PNG")
            st.download_button("⬇️ Baixar imagem anotada", data=buf.getvalue(), file_name=f"deteccao_{key_prefix}.png", mime="image/png")

            # Histórico (apenas miniatura)
            hist = st.session_state.get("history", [])
            thumb = cv2.resize(annotated, (224, int(224 * annotated.shape[0] / max(annotated.shape[1], 1))))
            hist.insert(0, {"img": thumb})
            st.session_state["history"] = hist[:12]
        else:
            st.warning("Sem resultados de detecção.")

    # Upload tradicional
    with tab_up:
        uploaded = st.file_uploader("Imagem (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
        if uploaded is not None and model:
            image = Image.open(uploaded)
            run_and_show(image, key_prefix="upload")

    # Câmera via streamlit-webrtc
    with tab_cam:
        st.caption("Permita o uso da câmera do dispositivo para capturar uma imagem.")
        ctx = webrtc_streamer(key="camera", mode=WebRtcMode.SENDRECV, video_frame_callback=None, async_processing=False, media_stream_constraints={"video": True, "audio": False})
        captured_img = None
        if ctx and ctx.state.playing and ctx.video_receiver:
            try:
                frame = ctx.video_receiver.get_frame(timeout=0.5)
                if frame is not None:
                    img = frame.to_ndarray(format="bgr24")
                    captured_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            except Exception:
                pass
        if st.button("📸 Capturar e Detectar", type="primary") and model:
            if captured_img is not None:
                run_and_show(captured_img, key_prefix="camera")
            else:
                st.warning("Nenhum frame capturado ainda.")

    # Exemplos
    with tab_ex:
        examples = _gather_example_images()
        if examples:
            selected = image_select("Escolha um exemplo", images=[Image.open(p) for p in examples], captions=[os.path.basename(p) for p in examples])
            if selected is not None and model:
                if st.button("Detectar no Exemplo", key="detect_example", type="primary"):
                    img_obj = selected if isinstance(selected, Image.Image) else Image.open(selected)
                    run_and_show(img_obj, key_prefix="exemplo")
        else:
            st.info("Nenhuma imagem de exemplo encontrada.")

    # Lote
    with tab_batch:
        many = st.file_uploader("Imagens (múltiplas)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if many and model:
            prog = st.progress(0)
            gallery_cols = st.columns(3)
            for i, uf in enumerate(many):
                img = Image.open(uf)
                run_and_show(img, key_prefix=f"lote_{i}")
                prog.progress(int((i + 1) / len(many) * 100))
            st.success("Lote processado.")

    # Histórico da sessão
    st.markdown("---")
    st.markdown("### Histórico da Sessão")
    hist = st.session_state.get("history", [])
    if hist:
        cols = st.columns(6)
        for i, item in enumerate(hist[:12]):
            with cols[i % 6]:
                st.image(item["img"], use_container_width=True)
    else:
        st.caption("Você ainda não gerou detecções nesta sessão.")


def page_training():
    st.markdown("## 📈 Treinamento")
    images, results_df, args = load_training_artifacts()

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(images["confusion"]):
            st.image(images["confusion"], caption="Matriz de Confusão", use_container_width=True)
        if os.path.exists(images["labels"]):
            st.image(images["labels"], caption="Distribuição de labels", use_container_width=True)
    with col2:
        if os.path.exists(images["results"]):
            st.image(images["results"], caption="Resultados globais", use_container_width=True)

    st.markdown("---")

    if results_df is not None:
        st.markdown("### Evolução das Métricas por Época")
        metric_cols = [
            c for c in results_df.columns 
            if any(k in c for k in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss"])
        ]
        if metric_cols:
            fig = go.Figure()
            epochs = results_df.get("epoch", pd.Series(range(len(results_df))))
            for c in metric_cols:
                fig.add_trace(go.Scatter(x=epochs, y=results_df[c], mode="lines+markers", name=c))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#E6E6E6',
                xaxis_title="Época",
                yaxis_title="Valor",
                legend_title="Métrica",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Não foram encontradas colunas de métricas esperadas no results.csv.")
    else:
        st.warning("Arquivo results.csv não encontrado em resultados/runs/detect/train.")

    st.markdown("---")
    if images["batches"]:
        st.markdown("### Amostras de Treino")
        cols = st.columns(3)
        for i, path in enumerate(images["batches"]):
            with cols[i % 3]:
                st.image(path, use_container_width=True)
    if images["val_preds"]:
        st.markdown("### Predições no Conjunto de Validação")
        cols = st.columns(3)
        for i, path in enumerate(images["val_preds"]):
            with cols[i % 3]:
                st.image(path, use_container_width=True)


def page_data():
    st.markdown("## 🗂️ Dados do Projeto")
    data_cfg, ann_df = load_dataset_info()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Configuração do Dataset (YAML)")
        if data_cfg:
            st.json(data_cfg)
        else:
            st.warning("Arquivo dados/road_signs_dataset.yaml não encontrado.")
    with col2:
        st.markdown("### Anotações")
        if ann_df is not None:
            st.dataframe(ann_df.head(200), use_container_width=True)
            st.caption("Visualizando até 200 primeiras linhas do CSV de anotações.")
        else:
            st.warning("Arquivo dados/road_signs_annotations.csv não encontrado.")


def page_about():
    st.markdown("## ℹ️ Sobre o Projeto")
    st.markdown(
        """
<div class="card">
<p>Aplicativo profissional em Streamlit para <b>detecção de sinais de trânsito</b> com YOLO, com interface dark premium, páginas de detecção, análise de treino e dados do projeto.</p>
<p><b>Autor:</b> <a href="https://github.com/sidnei-almeida" target="_blank">sidnei-almeida</a><br/>
<b>Contato:</b> <a href="mailto:sidnei.almeida1806@gmail.com">sidnei.almeida1806@gmail.com</a></p>
</div>
""",
        unsafe_allow_html=True,
    )


def main():
    with st.sidebar:
        st.markdown("<h2 style='color:#00C2FF'>Navegação</h2>", unsafe_allow_html=True)
        selected = option_menu(
            menu_title=None,
            options=["Início", "Detecção", "Treinamento", "Dados", "Sobre"],
            icons=["house", "camera", "graph-up", "folder", "info-circle"],
            default_index=0,
            styles={
                "container": {"padding": "0", "background": "transparent"},
                "icon": {"color": "#00C2FF"},
                "nav-link": {"color": "#E6E6E6", "--hover-color": "#171A23"},
                "nav-link-selected": {"background-color": "rgba(0,194,255,0.12)", "color": "#00C2FF", "border-left": "4px solid #00C2FF", "border-radius": "6px"},
            },
        )

        # Status do ambiente (apenas na sidebar)
        st.markdown("---")
        st.markdown("<h4 style='margin-bottom:0.5rem;'>🖥️ Ambiente</h4>", unsafe_allow_html=True)
        env = get_env_status()
        st.markdown(
            f"""
<div class="card">
  <div><b>Dispositivo:</b> {env['device']}</div>
  <div><b>Nome:</b> {env['device_name']}</div>
  <div><b>Python:</b> {env['python']}</div>
  <div><b>Torch:</b> {env['torch']} (CUDA: {env['cuda']})</div>
  <div><b>Ultralytics:</b> {env['ultralytics']}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    model = load_model()
    data_cfg, _ = load_dataset_info()
    _, results_df, _ = load_training_artifacts()

    if selected == "Início":
        page_home(model, data_cfg, results_df)
    elif selected == "Detecção":
        page_detect(model)
    elif selected == "Treinamento":
        page_training()
    elif selected == "Dados":
        page_data()
    else:
        page_about()


if __name__ == "__main__":
    main()
