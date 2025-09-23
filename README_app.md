# 🚦 App de Detecção de Placas de Trânsito

Aplicativo Streamlit profissional para detecção de placas de trânsito em tempo real usando YOLOv8.

## 🚀 Funcionalidades

- **Detecção em Tempo Real**: Upload de imagens e detecção instantânea de placas
- **4 Classes de Sinais**: Limite de velocidade, faixa de pedestres, semáforo e placa de parada
- **Interface Interativa**: Ajuste de limiar de confiança e visualização de resultados
- **Análise do Modelo**: Gráficos interativos de treinamento e métricas de performance
- **Design Profissional**: Interface moderna e responsiva

## 📦 Instalação

1. **Clone o repositório**:
```bash
git clone <repository-url>
cd road_sign_detection_yolo
```

2. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

3. **Execute o aplicativo**:
```bash
streamlit run app.py
```

## 🎯 Como Usar

1. **Acesse a aba "Detecção"**
2. **Faça upload de uma imagem** com placas de trânsito
3. **Ajuste o limiar de confiança** na barra lateral (opcional)
4. **Clique em "Detectar Placas"**
5. **Visualize os resultados** com bounding boxes e confiança

## 📊 Abas Disponíveis

### 🔍 Detecção
- Upload e processamento de imagens
- Visualização de resultados com bounding boxes
- Tabela de detecções com confiança

### 📊 Análise do Modelo
- Métricas finais de performance
- Informações sobre as classes detectadas
- Estatísticas do modelo

### 📈 Treinamento
- Gráficos interativos de métricas
- Curvas de loss durante o treinamento
- Estatísticas detalhadas do processo

### ℹ️ Sobre
- Informações do projeto
- Tecnologias utilizadas
- Performance do modelo

## 🛠️ Tecnologias

- **Frontend**: Streamlit
- **ML**: YOLOv8 (Ultralytics)
- **Visualização**: Plotly
- **Processamento**: OpenCV, PIL
- **Dados**: Pandas, NumPy

## 📈 Performance

- **mAP@0.5**: ~92%
- **mAP@0.5:0.95**: ~77%
- **Precision**: ~94%
- **Recall**: ~88%

## 🎨 Características da Interface

- Design responsivo e moderno
- Cores profissionais (azul e gradientes)
- Gráficos interativos com Plotly
- Feedback visual em tempo real
- Sidebar com configurações
- Sistema de abas organizado

## 🔧 Configurações

- **Limiar de Confiança**: 0.1 - 1.0 (padrão: 0.5)
- **Formatos Suportados**: PNG, JPG, JPEG
- **Resolução**: Automática (otimizada para 640x640)

## 📝 Notas

- O modelo deve estar na pasta `modelos/best.pt`
- Os dados de treinamento devem estar em `resultados/runs/detect/train/results.csv`
- A configuração do dataset deve estar em `dados/road_signs_dataset.yaml`
