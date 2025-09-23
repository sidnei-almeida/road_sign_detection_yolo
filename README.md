# 🚦 Road Sign Detection • YOLO

Detecção de placas de trânsito com YOLO, acompanhada de um aplicativo Streamlit elegante (tema dark) para inferência, visualização de métricas de treino e exploração de dados.

- **Autor**: [sidnei-almeida](https://github.com/sidnei-almeida)
- **Contato**: <sidnei.almeida1806@gmail.com>

---

## ✨ Destaques
- App Streamlit premium com tema dark e paleta ciano/roxo
- Página de **Detecção** com upload de imagens, **câmera (streamlit-webrtc)** e **seleção de exemplos (streamlit-image-select)**
- Página de **Treinamento** com gráficos (results.csv) e artefatos (matriz de confusão, batches, validação)
- Página de **Dados** exibindo `dados/road_signs_dataset.yaml` e amostra do CSV de anotações

> Observação: o modelo atual detecta apenas: **Traffic Light**, **Stop**, **Speedlimit**, **Crosswalk**.

---

## 🚀 Como executar

Pré-requisitos: Python 3.10+ e dependências do `requirements.txt`.

```bash
# clonar e entrar no projeto
git clone https://github.com/sidnei-almeida/road_sign_detection_yolo.git
cd road_sign_detection_yolo

# (opcional) criar venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows PowerShell

# instalar dependências
pip install -r requirements.txt

# rodar o app
bash run_app.sh
# ou
streamlit run app.py
```

Coloque os pesos do modelo em `modelos/best.pt` (ou utilize `resultados/runs/detect/train/weights/best.pt`).

---

## 🧱 Estrutura

```
road_sign_detection_yolo/
├─ app.py                         # App Streamlit
├─ .streamlit/config.toml         # Tema dark customizado
├─ dados/
│  ├─ road_signs_dataset.yaml     # Config do dataset YOLO
│  ├─ road_signs_annotations.csv  # Anotações (amostra/EDA)
│  └─ image_examples/             # Imagens para a aba Exemplos (PNG/JPG)
├─ modelos/
│  ├─ best.pt                     # Pesos do modelo (colocar aqui)
│  └─ last.pt
├─ resultados/runs/detect/train/  # Artefatos de treino do YOLO
│  ├─ results.csv                 # Métricas por época
│  ├─ results.png                 # Resumo
│  ├─ confusion_matrix.png        # Matriz de confusão
│  ├─ confusion_matrix_normalized.png
│  ├─ train_batch*.jpg            # Lotes de treino
│  ├─ val_batch*_pred.jpg         # Predições de validação
│  └─ weights/best.pt             # Pesos
└─ notebooks/                     # EDA, Preprocessamento, Treino
```

---

## 📈 App – Páginas
- **Início**: status do sistema, resumo de classes e mAP, destaques do treino
- **Detecção**: upload | câmera | exemplos, sliders de confiança e IoU
- **Treinamento**: gráficos interativos a partir do `results.csv` + imagens principais
- **Dados**: visualização do YAML do dataset e amostra de anotações
- **Sobre**: informações do projeto e contato

---

## 🧪 Exemplos
- Coloque imagens em `dados/image_examples/` para aparecerem na aba Exemplos.
- Se a pasta estiver vazia, o app tenta usar `dados/examples/` (legado) ou imagens de validação de `resultados/runs/detect/train`.

---

## 📬 Contato
- GitHub: [sidnei-almeida](https://github.com/sidnei-almeida)
- E-mail: <sidnei.almeida1806@gmail.com>

```text
Se este projeto foi útil para você, deixe uma estrela no repositório ⭐
```
